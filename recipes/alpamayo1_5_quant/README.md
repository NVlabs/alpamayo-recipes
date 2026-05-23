# FP8 / AutoQuant (FP8 + NVFP4) Quantization

This Recipe defines a reproducible post-training quantization (PTQ) procedure for quantizing Alpamayo 1.5 to FP8 or AutoQuant (FP8 + NVFP4)

## Prerequisites
This recipe is tested in the following settings. Other settings may also work but not guaranteed. 
- NVIDIA B300 GPU with CUDA 13
- Python 3.12
- Python Libraries: torch==2.12.0, torchvision==0.27.0, nvidia-modelopt==0.43.0

**NVIDIA Model Optimizer (ModelOpt)** is a library comprising state-of-the-art model optimization techniques including quantization and sparsity to compress models. It accepts a torch or ONNX model as input and provides Python APIs for users to easily stack different model optimization techniques to produce optimized & quantized checkpoints. In this recipe, we utilize ModelOpt to quantize Alpamayo 1.5.

## Getting started
### Download Alpamayo 1.5
```bash
git clone https://github.com/NVlabs/alpamayo1.5.git
cd alpamayo1.5
git checkout 2eff7037e47afb96a578b3d1bca453a373cd781e
```

### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Set up the environment
```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
```
To upgrade torch and torchvision, please run the following command to override the existing version, and install NVIDIA ModelOpt:
```bash
pip install torch==2.12.0 torchvision==0.27.0 nvidia-modelopt==0.43.0
```

### Authenticate with HuggingFace
The model and dataset require access to gated resources. Request access here: <br> 
🤗 [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) <br> 
🤗 [Alpamayo-1.5-10B Model](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Get your token at: https://huggingface.co/settings/tokens. Then authenticate: 
```bash
hf auth login
```

## Quantization

### Patch Alpamayo 1.5 repo

We need the following patches to quantize Alpamayo 1.5.

1. Move `eval.py`, `quantize_utils.py`, `1005_7cam_gold_eval_metadb_public.parquet`, and `0417_5k_train_set_for_calibration_25.10.parquet` to `alpamayo1.5/src/alpamayo1_5/`.
2. *[AutoQuant Only]* If you want to do AutoQuant as described in the next section, please copy the following code snippet to the line 692 of `alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py`.
    <details>
    <summary>Show code</summary>

    ```python
        def teacher_forced_flow_loss_forward(
            self,
            data: dict[str, Any],
        ) -> dict[str, torch.Tensor]:
            """Differentiable forward that returns the flow-matching training targets.

            Bypasses autoregressive reasoning generation and diffusion sampling.
            The VLM runs in a single non-sampling forward pass (with ``<traj_future_start>``
            appended to the prompt) to build the prompt KV cache; the expert then runs once
            on a linearly-interpolated noisy action and returns the predicted velocity field.

            Args:
                data: dict with ``tokenized_data`` (input_ids + other processor outputs),
                    ``ego_history_xyz``, ``ego_history_rot``, ``ego_future_xyz``,
                    ``ego_future_rot``.

            Returns:
                dict with keys ``v_pred`` and ``v_target``, both shape
                ``(B, n_diffusion_tokens, action_dim)``. Callers compute MSE between them.
            """
            ego_history_xyz = data["ego_history_xyz"]
            ego_history_rot = data["ego_history_rot"]
            ego_future_xyz = data["ego_future_xyz"]
            ego_future_rot = data["ego_future_rot"]
            B, n_traj_group, _, _ = ego_history_xyz.shape
            assert n_traj_group == 1, "Only one trajectory group is supported."

            tokenized_data = dict(data["tokenized_data"])
            input_ids = tokenized_data.pop("input_ids")
            traj_data_vlm = {
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            }
            input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
            device = input_ids.device

            # Append <traj_future_start> so the expert attends through the full prompt
            # that inference would have generated up to the action block.
            traj_future_start_id = self.tokenizer.convert_tokens_to_ids(
                to_special_token("traj_future_start")
            )
            start_col = torch.full(
                (input_ids.shape[0], 1),
                traj_future_start_id,
                dtype=input_ids.dtype,
                device=device,
            )
            input_ids = torch.cat([input_ids, start_col], dim=1)
            if "attention_mask" in tokenized_data and tokenized_data["attention_mask"] is not None:
                am = tokenized_data["attention_mask"]
                tokenized_data["attention_mask"] = torch.cat(
                    [am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1
                )

            vlm_outputs = self.vlm(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                **tokenized_data,
            )
            prompt_cache = vlm_outputs.past_key_values
            prefill_seq_len = prompt_cache.get_seq_length()
            rope_deltas = self.vlm.model.rope_deltas

            n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
            offset = torch.full((B,), prefill_seq_len, device=device, dtype=torch.long)

            position_ids = torch.arange(n_diffusion_tokens, device=device)
            position_ids = einops.repeat(position_ids, "l -> 3 b l", b=B).clone()
            delta = rope_deltas + offset[:, None]
            position_ids += delta.to(position_ids.device)

            # No padding between prompt cache and action block: full attention mask.
            attention_mask = torch.zeros(
                (B, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
                dtype=torch.float32,
                device=device,
            )

            forward_kwargs = {}
            if self.config.expert_non_causal_attention:
                forward_kwargs["is_causal"] = False

            # Build flow-matching target: x_1 = GT action, x_0 ~ N(0, I).
            x_1 = self.action_space.traj_to_action(
                traj_history_xyz=ego_history_xyz[:, 0],
                traj_history_rot=ego_history_rot[:, 0],
                traj_future_xyz=ego_future_xyz[:, 0],
                traj_future_rot=ego_future_rot[:, 0],
            )  # (B, n_diffusion_tokens, 2)
            x_1 = x_1.to(device=device, dtype=torch.float32)

            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, 1, 1, device=device, dtype=x_1.dtype)
            x_t = (1.0 - t) * x_0 + t * x_1
            v_target = x_1 - x_0

            # Cast to action-module dtype to match action_in_proj / expert weights.
            proj_dtype = next(self.action_in_proj.parameters()).dtype
            x_t_cast = x_t.to(dtype=proj_dtype)
            t_cast = t.to(dtype=proj_dtype)

            future_token_embeds = self.action_in_proj(x_t_cast, t_cast)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(B, n_diffusion_tokens, -1)

            expert_out = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
            v_pred = self.action_out_proj(last_hidden).view(
                B, *self.action_space.get_action_space_dims()
            )

            return {"v_pred": v_pred.to(torch.float32), "v_target": v_target}
    ```
    </details>

### Quantization-related arguments
The quantization path is controlled by the following arguments:

- `--parquet <path>`: evaluation clip source. `1005_7cam_gold_eval_metadb_public.parquet` is used by default.
- `--quant_format fp8`: enables FP8 PTQ.
- `--quant_algo <algo>`: keeps default `max` for FP8 runs.
- `--quant_weight_only` (optional): enables FP8 weight-only PTQ.
- `--calib_parquet <path>`: calibration clip source. `0417_5k_train_set_for_calibration_25.10.parquet` is used by default.
- `--num_of_calib_clips <N>`: number of calibration clips (1 to 5000). `100` is used by default.

Please refer to `eval.py` to find the usage of more arguments.

### FP8 quantization

Under the root of `alpamayo1.5` repo, run an example command below to quantize alpamayo1.5 in FP8 and save the quantized model:
```bash
PYTHONPATH=src python src/alpamayo1_5/eval.py --quant_format=fp8 --num_of_calib_clips=100 --save_model_dir=./models
```

### AutoQuant (FP8 + NVFP4)

Under the root of `alpamayo1.5` repo, run an example command below to quantize alpamayo1.5 in AutoQuant (FP8 + NVFP4) with 6.5 effective bits and save the quantized model:
```bash
PYTHONPATH=src python src/alpamayo1_5/eval.py --quant_format=auto --auto_quantize_bits=6.5 --num_of_calib_clips=100 --save_model_dir=./models
```

### Expected runtime behaviors and outputs
During a correct run, logs would show:
- Calibration clips are loaded from `--calib_parquet`.
- Calibration loop progress (`calibration: ...%`) is executed.
- Quantization summary is printed.
- Evaluation starts and reports per-clip metrics and final averages.

After the evaluation, you will see the following outputs:
- Average minADE
- Average evaluation time per clip
