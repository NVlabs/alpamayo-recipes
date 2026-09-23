## Optimization Categories

```
[01 Data Loading] ──CPU──► [02 Transport & Step Overhead] ──H2D(CPU→GPU)──► [03 Forward & Backward] ──GPU──► [04 Parallelism & Memory]
```

- **01 Data Loading**: storage → decode → CPU preprocessing → worker CPU scheduling & memory
- **02 Transport & Step Overhead**: packing / H2D copy / GC freeze / fused collective reporting
- **03 Forward & Backward**: forward-backward / attention kernels / the compiler
- **04 Parallelism & Memory**: parallel layout / FSDP / cross-GPU communication / memory management

Two more categories are infrastructure:

- **05 Measurement** (runs through 01-04): per-rank telemetry, CUDA-event decomposition, MFU/HFU/OFU, regression bands on a fixed reference run — answers "how do you know which stage is slow"
- **06 Methodology** (runs through 01-04): same-window paired runs, bitwise-invariance gates, dose-response ladders, the falsification loop — answers "how do you prove a change actually works and didn't break correctness"

## Note: Terminology

**Stall**

How long the slowest rank (GPU) in a training step has to wait for its data.

- Severe stall: within one step, the spread in when different ranks' data becomes ready exceeds 2 seconds — meaning one rank (whose data isn't ready) makes every other rank (GPU) sit idle for more than 2 seconds, which is pure waste.
- Stall burden: how large a share of all steps hit a severe stall, e.g. "9.1% (89/980 steps)" means 89 out of 980 training steps hit a severe stall.

**S1 / S2**

The two training stages in the Alpamayo training pipeline.

- S1 (Stage 1): full SFT training, training the whole model (including the backbone)
- S2 (Stage 2): expert-head training, the backbone stays frozen and only a small expert module is trained

S1 and S2 differ a lot in compute per step: S2's step is cheap, so the CPU-side fixed overhead (worker decode, memory allocation, etc.) is a larger share of the step and exposes data-side bottlenecks more easily; the optimal worker count, batch size, etc. also differ between the two, so gains in this document are usually reported separately for S1 and S2.

---

This directory will keep adding training-optimization write-ups over time; some content is still being organized — stay tuned.
