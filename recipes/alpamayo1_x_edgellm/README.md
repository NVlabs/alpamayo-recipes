# Alpamayo 2B TensorRT recipe for DRIVE Thor

Export, build and run the Alpamayo-R1 2B distill on DRIVE AGX Thor with TensorRT and a Python
runtime (`run.py`): six engines, a Domino drafter with a speculative verifier, in FP8 (about 175 ms
per request) or FP16 (about 220 ms). A C++ runtime for these engines is planned.

| Concern | Value |
|---|---|
| Target | DRIVE AGX Thor (SM 11.0, 1575 MHz), DRIVE OS 7.0.5 or 7.2.6, CUDA 13, TensorRT 11.2 or 11.3 |
| Weights | target checkpoint (vision, language, action expert), Domino drafter |
| Request | 4 cameras x 4 frames (576x320), 16-point ego history, 3006 prompt rows |
| Vision | newest frame of the 4 cameras per request; older frames are reused from earlier requests of the clip |
| Language | full-attention prefill; Domino drafts 16 tokens, the verifier checks 17 rows per round; greedy |
| Action | 6 candidates x 64 steps, 5 flow-matching steps, fixed noise (`action-noise.bin` in the bundle) |
| KV cache | one paged FP16 pool per layer (TensorRT Edge-LLM `AttentionPlugin` layout, 40 pages x 128 rows): pages 0..24 hold the language rows, pages 25..36 the six action candidates' private rows |
| Attention plugin | Edge-LLM's `AttentionPlugin` (tree mode) in `verify_q17`, `verify_q1` and `action`; the Alpamayo plugin is `AlpamayoFusedGateUpPlugin` only |

| Engine | FP8 (default) | FP16 (`--precision fp16`) |
|---|---|---|
| `visual` | FP8 GEMMs, FP8-rounded Q/K attention | FP16 |
| `prefill` | FP16 attention, FP8 FFN (`AlpamayoFusedGateUpPlugin`: gate/up GEMM + SwiGLU, FP8 out) | FP16 |
| `verify_q17` | FP16 attention (`AttentionPlugin`, tree mode, 17 rows), FP8 FFN and LM head | FP16, `AttentionPlugin` |
| `verify_q1` | FP16, `AttentionPlugin` (appends the trajectory start token when needed) | same |
| `domino` | FP8 projections and FFN, FP16 attention | FP16 |
| `action` | FP8 FFN, FP16 attention (`AttentionPlugin`, tree mode, 6 sequences x 64 rows) | FP16, `AttentionPlugin` |

FP8 weight scales are amax/448 of the checkpoint (the action FFN's are carried in `fp8-scales.json`:
the checkpoint's max over all layers / 448); activation scales are per-tensor ModelOpt max
calibrations (scale = amax/448) frozen in the bundle's `fp8-scales.json` and carried in the ONNX
Q/DQ nodes and plugin inputs.

Every step runs on the DRIVE Thor, from `recipes/` in a checkout of this repository.
Placeholders: `MODEL` = the model bundle (below); `TENSORRT_ROOT` = TensorRT 11 SDK (`include/`,
`lib/`, `python/`); `WORK` = a new writable directory with about 25 GB free per precision
(environment 6 GB, ONNX 9 GB, engines 8 GB FP8 / 12 GB FP16).

The model bundle is one directory holding the weights and the files derived from them:

| File | Content |
|---|---|
| `config.json`, `model*.safetensors` (+ `model.safetensors.index.json` when sharded) | target checkpoint: vision, language and action expert |
| `domino-drafter.safetensors` | Domino drafter |
| `fp8-scales.json` | FP8 activation scales per engine, and the action FFN weight scales |
| `domino-vocab.json` | the drafter's 4096-token compact vocabulary (`compact_to_global`) |
| `action-noise.bin` | 384 x 2 FP32 initial trajectory noise |
| `MANIFEST.json` | size and SHA-256 of every file above; `python -m alpamayo1_x_edgellm.bundle $MODEL` writes it |

The weights are not distributed with this recipe. Assemble the bundle from the target checkpoint
directory (`CHECKPOINT`), the drafter file (`DRAFTER`) and the directory holding the three derived
files (`DERIVED`), then write the manifest (with the step 0 environment active):

```bash
mkdir $MODEL
cp $CHECKPOINT/config.json $CHECKPOINT/model*.safetensors $MODEL/
cp $CHECKPOINT/model.safetensors.index.json $MODEL/ 2>/dev/null || true   # sharded checkpoints only
cp $DRAFTER $MODEL/domino-drafter.safetensors
cp $DERIVED/fp8-scales.json $DERIVED/domino-vocab.json $DERIVED/action-noise.bin $MODEL/
python -m alpamayo1_x_edgellm.bundle $MODEL
sha256sum $MODEL/MANIFEST.json   # the validated bundle: 20a191a68d27e30e99688cb1eb37a1057700f310e705620eb2cc13c7cb7b30e4
```

Every exporter and `run.py` check the manifest before reading the bundle, and the result JSON
records its hash.

0. Environment, once. Python 3.12 with `venv`, CMake, CUDA 13 (`nvcc` on `PATH`), and a Hugging Face
   account with access to the gated `nvidia/Cosmos-Reason2-2B` and `nvidia/PhysicalAI-Autonomous-Vehicles`:

   ```bash
   git submodule update --init --recursive alpamayo1_x_edgellm/tensorrt-edge-llm
   python3.12 -m venv $WORK/venv && source $WORK/venv/bin/activate
   export UV_CACHE_DIR=$WORK/uv-cache CUDA_CACHE_PATH=$WORK/cuda-cache
   pip install uv==0.9.24
   uv sync --project alpamayo1_x_edgellm --frozen --active --inexact --no-install-project
   uv pip install $TENSORRT_ROOT/python/tensorrt-11.*-cp312-none-linux_aarch64.whl
   export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib   # set, not extended: system CUDA libraries break torch's cuBLAS
   hf auth login
   ```

   GPU memory, once per boot: DRIVE Thor allocates CUDA memory from the 2 MiB hugepage pool
   (6 GiB by default); the run needs 20 GiB (10240 pages):

   ```bash
   cat /proc/sys/vm/nr_hugepages                    # current pages; note it to restore later
   echo 10240 | sudo tee /proc/sys/vm/nr_hugepages
   grep HugePages_Total /proc/meminfo               # expect 10240
   ```

1. Download the Cosmos-Reason2 tokenizer and processor files, pinned to one revision of
   [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) (NVIDIA Open
   Model License):

   ```bash
   python -m alpamayo1_x_edgellm.export.metadata --output-dir $WORK/metadata
   ```

2. Export the six ONNX graphs:

   ```bash
   P="--precision fp8"   # FP16: P="--precision fp16"
   for stage in visual prefill verify domino action; do
     python -m alpamayo1_x_edgellm.export.$stage --model $MODEL $P --output-dir $WORK/onnx
   done
   ```

   For both precisions, use one `$WORK/onnx` and `$WORK/engines` per precision (for example
   `onnx-fp16`, `engines-fp16`). `run.py` reads the precision from the engines.

3. Prepare the requests: the first 20 clips of `dataset/1005_7cam_gold_eval_metadb_public.parquet` (repository root),
   16 frames per camera at 10 Hz ending at t0 = 5.1 s, and 13 sliding 4-frame windows per clip
   (260 requests):

   ```bash
   python -m alpamayo1_x_edgellm.prepare --output-dir $WORK/jobs
   ```

4. Build the plugins and engines:

   ```bash
   python -m alpamayo1_x_edgellm.build --onnx-dir $WORK/onnx --output-dir $WORK/engines --tensorrt-root $TENSORRT_ROOT
   ```

   About 20 minutes: TensorRT Edge-LLM's plugin library from the submodule (`AlpamayoFusedGateUpPlugin`
   and Edge-LLM's `AttentionPlugin` with its CuTe DSL `fmha_decode_ctm` kernels, extracted from
   `tensorrt-edge-llm/kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_110_cuda13.tar.gz`), then the six
   engines. DRIVE OS has no cuRAND headers: `build.py` copies `curand*.h` from the environment's
   `nvidia-curand` wheel to `$WORK/engines/curand-include` (or pass `--curand-include DIR`); nvcc's
   temporary files go to `$WORK/engines/tmp`.

5. Run:

   ```bash
   python -m alpamayo1_x_edgellm.run --engines $WORK/engines --model $MODEL --vlm-metadata $WORK/metadata \
       --jobs $WORK/jobs/jobs.json --output $WORK/results.json
   ```

   Writes one row per request (minADE6, generated tokens, rounds, `request_ms`) and prints the
   summary. `request_ms` runs from the token handoff to the trajectory.

Expected results (260 requests; latency over 240, each clip's first excluded), DRIVE OS 7.0.5,
TensorRT 11.2; DRIVE OS 7.2.6 with TensorRT 11.3 builds and runs the same steps but has not been
re-measured with this checkpoint:

| Metric | FP8 | FP16 |
|---|---|---|
| minADE6, mean | 1.509 m | 1.527 m |
| minADE6, windows ending at t0 | 1.296 m | 1.313 m |
| Accepted tokens per round | 8.30 | 8.44 |
| Request latency, median / p90 | 171.2 / 173.2 ms | 220.6 / 225.1 ms |

Of the FP8 request, about 10 ms is host work feeding the paged pool (prefill K/V copied into pages,
per-request action page table and private pages).

Files: `prepare.py`, `export/{graph,visual,prefill,verify,domino,action}.py` (one exporter per
engine; `graph.py` is the shared ONNX builder), `export/metadata.py`, `build.py`,
`run.py`, `bundle.py`, and `core.py`.
