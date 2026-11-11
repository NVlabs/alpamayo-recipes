# Alpamayo2-Edge on DRIVE AGX Thor with TensorRT Edge-LLM

The released [`nvidia/Alpamayo2-Edge`](https://huggingface.co/nvidia/Alpamayo2-Edge) checkpoint
running end to end on a DRIVE AGX Thor: 22 camera frames + ego history -> chain-of-causation text +
6 trajectory candidates, on the order of 200 ms per request in FP8.

[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) `v0.11.1` (submodule
`tensorrt-edge-llm/`, unmodified) does the export, the FP8 quantization and the engine builds. The
recipe adds what the stock tools do not cover for this model: `host/shim.py` re-keys the release
checkpoint into the layout the exporter reads, `host/export.py` and `host/action_graph.py` export
the trajectory expert (the stock action exporter is Alpamayo-R1 only), and `thor/run.py` with
`thor/runtime/` drives the engines from Python: one prefill shared by all candidates, encoded frames
cached per image, speculative decoding with a DFlash2 draft engine, the expert attending the
language model's paged K/V pool in place through Edge-LLM's attention plugin, FP8 weights,
activations and K/V cache.

## Tested with

Other versions are untested.

| | |
|---|---|
| board | DRIVE AGX Thor, DRIVE OS 7.2.7.0, CUDA 13.4, TensorRT 11.3.1.9 (aarch64 SDK) |
| board Python | `pyproject.toml` + `uv.lock` (Python 3.12, torch 2.10.0+cu130, tokenizers 0.22.2) + the SDK's TensorRT wheel |
| host | x86-64, Python 3.12, `uv`; `tensorrt-edge-llm[tools]` (torch 2.13.0+cu130, transformers 5.14.1) |
| host, FP8 quantization | the same extras with transformers 5.17.0 (the first release with `cosmos3_edge`), ModelOpt 0.45.0, one GPU with >= 16 GB |
| model | `nvidia/Alpamayo2-Edge` revision `aaaa32c66af38a40aaeabb55effb44858c15e79d` |
| Edge-LLM | tag `v0.11.1` (`358de11c`) |

## Host (x86)

```bash
export WORK=$PWD/work                                 # any disk with >= 40 GB free
export RECIPE=$PWD/recipes/alpamayo2edge_edgellm      # from a checkout of this repository
git submodule update --init --recursive $RECIPE/tensorrt-edge-llm
mkdir -p $WORK && cd $WORK

# 1. checkpoint (needs `hf auth login` + model access)
hf download nvidia/Alpamayo2-Edge --revision aaaa32c66af38a40aaeabb55effb44858c15e79d --local-dir downloads/Alpamayo2-Edge

# 2. environments: Edge-LLM's own pins; a second one for the quantizer (transformers 5.17.0)
uv venv --python 3.12 env/host && uv pip install --python env/host/bin/python -e "$RECIPE/tensorrt-edge-llm[tools]" pillow
uv venv --python 3.12 env/quant && uv pip install --python env/quant/bin/python "$RECIPE/tensorrt-edge-llm[tools]" && uv pip install --python env/quant/bin/python transformers==5.17.0
source env/host/bin/activate

# 3. inputs: one clip = frames resized per camera budget, prompt rendered and tokenized, ego history;
#    from a PhysicalAI-AV clip through the public alpamayo2_edge package (its examples/nav_samples.json
#    names one; commit 02f55a7 is the first with the Cosmos3-Edge checkpoint), or from an Edge
#    inference sample on disk (sample.json + tensors.safetensors)
uv venv --python 3.12 env/inputs && uv pip install --python env/inputs/bin/python "alpamayo2-edge[dataset] @ git+https://github.com/NVlabs/alpamayo2-edge@02f55a7"
env/inputs/bin/python $RECIPE/host/make_inputs.py --clip ea7bbd31-b7a5-4972-8dbd-7089e6b53de4 4000000 "Turn right in 30m" downloads/Alpamayo2-Edge inputs/<clip>

# 4. shim: release layout -> exporter layout (+ expert.safetensors), and the HF layout for the quantizer
python $RECIPE/host/shim.py downloads/Alpamayo2-Edge shim
python $RECIPE/host/shim.py downloads/Alpamayo2-Edge shim-hf --hf

# 5. FP8: Edge-LLM's quantizer calibrated on the clip's camera frames, with FP8 K/V-cache scales;
#    the ViT's patch embedding and first two layers stay FP16
env/quant/bin/python $RECIPE/host/quantize.py shim-hf quant/fp8 inputs/<clip> --kv-fp8
python $RECIPE/host/shim.py quant/fp8 shim-fp8 --expert shim --fp16-modules "shim:visual.embeddings.patch_embedding,visual.encoder.layers.0.,visual.encoder.layers.1."
env/quant/bin/python $RECIPE/host/quantize.py --draft shim-hf $DRAFT_SNAPSHOT quant/draft-fp8 inputs/<clip>

# 6. reduced output vocabulary: the 2211 token ids the chain of causation uses (host/coc_vocab.json),
#    padded to 8192 rows for the language model's head and 4096 for the drafter's
python $RECIPE/host/vocab.py shim-fp8 $RECIPE/host/coc_vocab.json 8192 vocab/llm
python $RECIPE/host/vocab.py shim-fp8 $RECIPE/host/coc_vocab.json 4096 vocab/draft

# 7. ONNX: stock VLM export, the text engine again with the drafter's tap outputs and the reduced
#    head, the expert as one graph (5 Euler steps, K/V capacity 3584 rows, FP8 GEMMs with the ranges in
#    host/expert_scales.json) over the FP8 pool, the FP8 DFlash2 draft engine
tensorrt-edgellm-export shim-fp8 onnx/fp8 --task reasoning --target-sm 110
python $RECIPE/host/export.py llm shim-fp8 onnx/fp8/llm --vocab vocab/llm
python $RECIPE/host/action_graph.py shim-fp8 onnx/fp8/action 5 3584 --fp8-kv --fp8 $RECIPE/host/expert_scales.json
python $RECIPE/host/export.py draft shim quant/draft-fp8 onnx/draft --vocab vocab/draft
```

`$DRAFT_SNAPSHOT` is a DFlash2 block-16 drafter for this VLM (`config.json` + `draft.safetensors`);
it is not part of the model release. Without one, skip the draft steps and run greedy decoding (numbers
below). FP16 engines: skip step 5, export from `shim` and `$DRAFT_SNAPSHOT` without `--fp8-kv` and `--fp8`.
`host/expert_scales.json` holds the expert projections' input ranges (max |input| over ten validation clips
through the PyTorch reference); the input and output projections and the norms stay FP16. The reduced vocabulary
is Edge-LLM's `reduce-vocab` format: the transformer and the input embedding keep all 135168 tokens, only
the head rows go; greedy decoding is unchanged while the argmax stays in the list (no chain of the clips we
tested leaves it). Full heads: drop step 6 and the two `--vocab` flags.

Copy `onnx/`, `inputs/` and the recipe folder (with the submodule) to the board's work directory.

## DRIVE AGX Thor

```bash
export WORK=~/work TENSORRT_ROOT=~/TensorRT-11.3.1.9      # TensorRT aarch64 SDK (include/ lib/ bin/ python/)
bash $WORK/recipe/thor/setup.sh                            # once per board: venv from uv.lock + the TensorRT wheel, env.sh
source $WORK/env.sh
export EDGELLM_PLUGIN_PATH=$WORK/edgellm/build/libNvInfer_edgellm_plugin.so

python $WORK/recipe/thor/build.py $WORK/onnx/fp8 16                                          # Edge-LLM + llm/ visual/ engines (about 15 min)
python $WORK/recipe/thor/build_action.py $WORK/onnx/fp8/action $WORK/engines/fp8/action      # about 10 min
python $WORK/recipe/thor/build_draft.py $WORK/onnx/draft $WORK/engines/draft 3584            # about 5 min
python $WORK/recipe/thor/run.py $WORK/engines/fp8 $WORK/inputs/<clip> --steps 5 --rolling --draft-engine $WORK/engines/draft
```

`run.py` writes `$WORK/results/fp8/<clip>/output.json` (chain of causation, 64 acceleration and
curvature pairs per candidate) and `profile.json` (milliseconds per stage). Options: `--candidates`
(default 6), `--steps` (Euler steps, default 10), `--rolling` (time only the newest frame of each
camera; the others are encoded before the clock starts, as a streaming runtime has them from earlier
ticks), `--draft-engine DIR`, `--seed`, `--warmup` (untimed runs first, default 1). Trajectories on the host:

```bash
python $RECIPE/host/compare.py inputs/<clip> results/fp8/<clip> downloads/Alpamayo2-Edge/config.json
```

## Results

Indicative timings from one DRIVE AGX Thor with the stage timers of `thor/run.py` (stream
synchronization at every stage boundary, Python driver), warm, 6 candidates, 5 Euler steps, the
engines above. They are not a benchmark measurement; expect them to move with the board, the
DRIVE OS and TensorRT versions and the clip. Rounded, on the public example clip of step 3 (22 frames,
about 3300 prompt tokens, 4 drafter rounds):

| stage | about |
|---|---|
| ViT, 6 newest frames | 20 ms |
| prefill | 75 ms |
| chain of causation, per drafter round (draft + verify) | 12 ms |
| expert, 5 Euler steps x 6 candidates | 70 ms |

A request is ViT, prefill and expert once, plus one chain-of-causation round per 12 ms (greedy
decoding instead: about 7 ms per token); this clip takes about 220 ms. Its chain reads "Decelerate
and turn right at the intersection to follow our route, as the roadway is clear." and the six
candidates each lie within 0.3 to 0.45 m ADE of one of the six the public PyTorch inference
(`alpamayo2_edge.inference`, 5 Euler steps, sampled chains) produces for the same clip, about the
spread among those six themselves. Over ten internal validation clips, 1 to 7 rounds, roughly 180 to
250 ms per request; mean minADE6 1.547 m, against 1.570 m with FP16 engines; speculative decoding
reproduces the greedy chain on every clip.

## Layout

```
recipes/alpamayo2edge_edgellm/
├── README.md  LICENSE                    Apache-2.0
├── pyproject.toml  uv.lock               board Python environment (thor/setup.sh installs it)
├── tensorrt-edge-llm/                    submodule: TensorRT Edge-LLM v0.11.1, unmodified
├── host/                                 x86 host, in the order of the steps above
│   ├── make_inputs.py                    clip -> prompt tokens, resized frames, ego history
│   ├── shim.py                           release checkpoint -> exporter layout / HF layout, FP16 module overrides
│   ├── quantize.py                       Edge-LLM FP8 quantizer on the clip's frames (--kv-fp8); --draft: the drafter
│   ├── expert_scales.json                expert FP8 input ranges (action_graph.py --fp8)
│   ├── vocab.py  coc_vocab.json          reduced output vocabulary (vocab_map.safetensors) from the chain's token ids
│   ├── export.py                         tapped LLM export, DFlash2 draft export (--vocab: reduced heads)
│   ├── action_graph.py  onnx_graph.py    the expert as one ONNX graph over the paged pool (FP8 GEMMs); ONNX builder
│   └── compare.py                        trajectories from a result
└── thor/                                 DRIVE AGX Thor
    ├── setup.sh                          venv, env.sh
    ├── build.py                          Edge-LLM build + llm/ visual/ engines
    ├── build_action.py  build_draft.py   action and draft engines (TensorRT Python API)
    ├── run.py                            one clip through the engines; output.json, profile.json
    └── runtime/
        ├── engine.py                     TensorRT contexts and bindings
        ├── vision.py  llm.py  action.py  ViT (per-image cache), language model (paged K/V), expert
        ├── rope.py                       M-RoPE positions and cos/sin tables
        └── speculative.py  drafter_trt.py   drafter rounds; the draft engine
```
