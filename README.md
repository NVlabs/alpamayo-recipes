# NVIDIA Alpamayo Developer Hub

**Open platform for reasoning-based autonomous driving.** Vision Language Action models, closed-loop simulation, reinforcement learning, reasoning-based auto-labeling, and open driving datasets for transparent driving.

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00088-b31b1b.svg)](https://arxiv.org/abs/2511.00088)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/nvidia)
[![NVIDIA Forum](https://img.shields.io/badge/NVIDIA%20Forum-Alpamayo-76B900.svg)](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Challenges](https://img.shields.io/badge/🏆%20Challenges-2026-orange.svg)](#participate-in-challenges)

</div>

<div align="center">

<a href="https://www.youtube.com/watch?v=KGCTwoAlhsM">
  <img src="https://img.youtube.com/vi/KGCTwoAlhsM/maxresdefault.jpg" width="560" alt="How Autonomous Vehicles Learn to Reason With NVIDIA Alpamayo"/>
</a>

</div>

---

## What's New

- [**NVIDIA Alpamayo 2 Super**](https://github.com/NVlabs/alpamayo2) is an open 34B reasoning Vision Language Action model for end-to-end autonomous driving, built on the NVIDIA Cosmos 3 backbone. Model weights and code can be found on [Hugging Face](https://huggingface.co/nvidia/Alpamayo2-Super) and [GitHub](https://github.com/NVlabs/alpamayo2).
- [**NVIDIA AlpaGym**](https://github.com/NVlabs/alpagym) is a closed-loop reinforcement learning framework for training end-to-end autonomous driving policies in simulation.
- [**NVIDIA CoC Auto-Labeling Pipeline**](https://github.com/NVlabs/alpamayo-coc-autolabeler) is an open-source pipeline that generates Chain-of-Causation (CoC) reasoning labels for driving clips automatically.

---

## What is Alpamayo?

**NVIDIA Alpamayo** is an open platform designed to accelerate the development of safe, transparent, and reasoning-based autonomous vehicles. It consists of vision-language-action (VLA) models, simulation frameworks, reinforcement learning infrastructure, and physical AI datasets and tools. The VLA models scale from the 10B-parameter Alpamayo 1 Nano and Alpamayo 1.5 Nano to the 34B-parameter Alpamayo 2 Super, each generating driving trajectories alongside CoC reasoning traces that make every decision transparent and auditable. AlpaSim provides closed-loop policy simulation, and AlpaGym enables reinforcement learning on top to learn from edge-case failures across massively parallel GPU simulations. The Physical AI AV Dataset spans 2500+ cities across 25 countries and provides real-world long-tail training data. The CoC Auto-Labeling Pipeline annotates data with meta-actions and reasoning labels automatically.

---

## Why Alpamayo?

| | |
|---|---|
| **Open platform** | Model weights, inference code, closed-loop simulation, reinforcement learning tools, and driving datasets, all published openly for reasoning-based autonomous driving |
| **Reasoning-based** | CoC traces accompany every predicted trajectory, making each driving decision transparent and auditable |
| **Closed-loop by design** | AlpaGym and AlpaSim train and validate driving policies through continuous decision and observation cycles, surfacing the compounding, long-tail failures that open-loop log replay and imitation learning miss |
| **Scales with GPUs** | Simulation, reinforcement learning, and inference scale from a single GPU to multi-node across the family |

---

## Which Repository should I use?

This repository holds the post-training recipes and utility scripts. The models, simulation frameworks, and datasets live in the sibling repositories linked in each section below.

| | **Purpose** | **When to use** | **Location** |
|---|---|---|---|
| **Inference** | Run inference and reasoning with the Alpamayo VLA models | You want to run or evaluate a version of an Alpamayo VLA model | [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo), [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5), [NVlabs/alpamayo2](https://github.com/NVlabs/alpamayo2) |
| **Recipes** | Post-train, quantize or distill Alpamayo VLA models | You want to adapt an Alpamayo model to your data, compute budget, or sensor rig | This repository |
| **Simulation and RL** | Test and train driving policies in closed-loop simulation | You want to validate policies or run RL beyond open-loop log replay | [NVlabs/alpasim](https://github.com/NVlabs/alpasim), [NVlabs/alpagym](https://github.com/NVlabs/alpagym) |
| **Data and Labeling** | Generate reasoning labels | You want to auto-label driving clips or start from open data | [NVlabs/alpamayo-coc-autolabeler](https://github.com/NVlabs/alpamayo-coc-autolabeler) |
| **Physical AI AV Dataset** | Open multi-sensor driving dataset with CoC labels | You want real-world training data to train or test your models | [NVlabs/physical_ai_av](https://github.com/NVlabs/physical_ai_av) |

---

## Recipes

End-to-end recipes for the Alpamayo VLA models, covering supervised fine-tuning (SFT), open-loop reinforcement learning (RL), and quantization.
**Each recipe includes:**
- **Data preparation** using the utility scripts to download and curate the Physical AI Autonomous Vehicles dataset
- **Training configuration** with model, optimizer, and hyperparameter settings
- **A per-folder README** with installation, run instructions, and hardware requirements

| Recipe | Description | Stage |
|--------|-------------|-------|
| [`recipes/alpamayo1_sft/`](recipes/alpamayo1_sft/README.md) | Alpamayo 1 supervised fine-tuning with Hugging Face Trainer and DeepSpeed | SFT |
| [`recipes/alpamayo1_5_sft/`](recipes/alpamayo1_5_sft/README.md) | Alpamayo 1.5 supervised fine-tuning with Hugging Face Trainer and DeepSpeed | SFT |
| [`recipes/alpamayo1_x_rl/`](recipes/alpamayo1_x_rl/README.md) | Alpamayo 1 and 1.5 open-loop reinforcement learning post-training with Cosmos-RL and GRPO | RL (GRPO) |
| [`recipes/alpamayo1_5_quant/`](recipes/alpamayo1_5_quant/README.md) | Alpamayo 1.5 quantization with Model Optimizer Toolkit — FP8 and NVFP4 + FP8 Mixed Precision | Quantization |

**What you can adapt:** fine-tune on your own fleet data, adjust the camera count and sensor configuration, and swap reward functions for the RL recipe. See each recipe README for supported inputs and hardware requirements.

---

## Utility Scripts

Helper scripts for preparing data and converting checkpoints across the Alpamayo recipes.

| Script | Purpose |
|--------|---------|
| `scripts/download_pai.py` | Download the Physical AI Autonomous Vehicles dataset from Hugging Face |
| `scripts/curate_pai_samples.py` | Curate a subset of Physical AI Autonomous Vehicles samples |
| `scripts/convert_checkpoint.py` | Convert between Alpamayo 1 and 1.5 checkpoints |
| `scripts/convert_release_config_to_training.py` | Convert a release checkpoint to training format |
| `scripts/convert_cosmos_rl_checkpoint.py` | Convert a Cosmos-RL checkpoint to Hugging Face format |

---

## Alpamayo VLA Models

Vision Language Action models that generate driving trajectories alongside CoC reasoning traces for end-to-end autonomous driving.

| Model | Params | Backbone | Highlights | Resources |
|-------|--------|----------|------------|-----------|
| **Alpamayo 1 Nano** | 10B | Cosmos-Reason | Four-camera reasoning VLA with trajectory and reasoning output | [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo), [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) |
| **Alpamayo 1.5 Nano** | 10B | Cosmos-Reason2 | RL post-trained, flexible camera count, navigation guidance, visual question answering | [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) ,[nvidia/Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) |
| **Alpamayo 2 Super** | 34B | Cosmos 3 | 360-degree surround-view camera inputs, meta-action outputs, CoC autolabeling, visual question answering, 2D grounding, multi-task teacher model | [NVlabs/alpamayo2](https://github.com/NVlabs/alpamayo2) , [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super)  |

**White paper:** [Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail](https://arxiv.org/abs/2511.00088)

<details>
<summary><strong>Alpamayo 2 Super</strong> — 34B · Cosmos 3 backbone</summary>

NVIDIA Alpamayo 2 Super is an open 34B-parameter reasoning Vision Language Action (VLA) model for end-to-end autonomous driving, built on the NVIDIA Cosmos 3 backbone at 3x the scale of prior Alpamayo generations. Full 360-degree surround perception and meta-action outputs encoding high-level behavioral decisions alongside trajectory predictions make it multi-task by design, handling planning, CoC auto-labeling, grounded visual question answering, and model evaluation in one architecture.

**Model Specifications:**
- 34B parameters, built on the Cosmos 3 backbone, 3x the scale of Alpamayo 1 Nano and 1.5 Nano
- 360-degree surround perception
- Meta-action outputs encoding high-level behavioral decisions alongside trajectory predictions
- Trained on 110,000+ hours of driving data
- Multi-task: planning, CoC auto-labeling, and grounded visual Q&A in one architecture

**What You Can Build:**
- Trajectory planning with explicit meta-actions
- CoC auto-labeling from driving clips
- Evaluation and judging of smaller on-board models
- A teacher model for distillation and quantization into student models that meet in-vehicle latency and safety requirements on DRIVE AGX Thor

**Resources:** [NVlabs/alpamayo2](https://github.com/NVlabs/alpamayo2) · [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super)

</details>

<details>
<summary><strong>Alpamayo 1.5 Nano</strong> — 10B · Cosmos-Reason2 backbone</summary>

NVIDIA Alpamayo 1.5 Nano is an open 10B-parameter reasoning Vision Language Action (VLA) model for end-to-end autonomous driving, reinforcement-learning post-trained on the NVIDIA Cosmos-Reason2 backbone. RL post-training sharpens reasoning quality and tightens consistency between what the model decides and what it does. Flexible multi-camera support with variable camera count adapts the model to different sensor rigs without retraining from scratch. Visual question answering and natural-language navigation make the vehicle steerable and explainable in real time.

**Model Specifications:**
- 10B parameters, built on the Cosmos-Reason2 backbone
- RL post-trained for reasoning quality and reasoning-trajectory consistency
- Flexible multi-camera support with variable camera count
- Visual question answering and natural-language navigation guidance
- SFT and RL post-training scripts included

**Resources:** [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) · [nvidia/Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

</details>

<details>
<summary><strong>Alpamayo 1 Nano</strong> — 10B · Cosmos-Reason backbone</summary>

NVIDIA Alpamayo 1 Nano is an open 10B-parameter reasoning Vision Language Action (VLA) model for end-to-end autonomous driving, built on the NVIDIA Cosmos-Reason backbone. It was the first open industry-scale reasoning VLA for autonomous driving, released as Alpamayo-R1 and introduced as Alpamayo 1 at CES 2026. It generates driving trajectories alongside CoC reasoning traces from multi-camera video and egomotion history.

**Model Specifications:**
- 10B parameters, Cosmos-Reason backbone (8.2B) with a diffusion-based action expert (2.3B)
- Inputs: four-camera video, egomotion history, and text commands
- Outputs: driving trajectories and natural-language CoC reasoning traces
- Trained on more than 1B images from 80,000 hours of driving
- Runs on a single GPU with 24GB of memory

**Resources:** [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo) · [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)

</details>

---

## Simulation and Reinforcement Learning

Closed-loop simulation and reinforcement learning for autonomous driving policies, where every policy decision reshapes the simulation state.

| Framework | Role | Highlights | Resources |
|-----------|------|------------|-----------|
| **AlpaSim** | Closed-loop simulation platform | Microservice architecture, NuRec and OmniDreams rendering backends, GPU-scalable | [NVlabs/alpasim](https://github.com/NVlabs/alpasim) |
| **AlpaGym** | Closed-loop RL framework | Trains driving policies in AlpaSim, Cosmos RL backend, GRPO default, swappable rewards and algorithms | [NVlabs/alpagym](https://github.com/NVlabs/alpagym) |

<details>
<summary><strong>AlpaSim</strong> — Closed-loop simulation platform</summary>

NVIDIA AlpaSim is an open-source closed-loop simulation platform for developing and testing end-to-end autonomous driving policies. Its microservice architecture orchestrates Driver, Renderer, TrafficSim, Controller, and Physics services as independent processes, each assignable to different GPUs for parallelized simulation across large fleets of concurrent policy runs.

**What It Provides:**
- Closed-loop simulation where every policy decision reshapes the simulation state, going beyond open-loop log replay
- GPU-scalable microservice architecture across independent services
- Two rendering backends: Omniverse NuRec for neural reconstruction of real-world scenes, and OmniDreams for generative world model rendering of novel and long-tail scenarios
- The orchestration layer for AlpaGym reinforcement learning, scaling from single-GPU development to multi-node training

**Resources:** [NVlabs/alpasim](https://github.com/NVlabs/alpasim)

</details>

<details>
<summary><strong>AlpaGym</strong> — Closed-loop RL framework</summary>

NVIDIA AlpaGym is an open-source closed-loop reinforcement learning (RL) framework for training end-to-end autonomous driving policies at GPU scale. It trains policies through continuous closed-loop decision and observation cycles inside AlpaSim, where every braking, steering, and navigation action reshapes the environment and surfaces the compounding, long-tail failures that log-based imitation learning misses.

**What It Provides:**
- Closed-loop reinforcement learning across parallel GPU simulations
- Built on AlpaSim with NVIDIA Cosmos RL as the backend and GRPO as the default RL algorithm
- Swappable policy models, reward configurations, and RL algorithms
- Standard reward functions for collision avoidance, offroad detection, and progress
- Physical AI NuRec dataset support to begin RL training without additional scene reconstruction or annotation

**Resources:** [NVlabs/alpagym](https://github.com/NVlabs/alpagym)

</details>

---

## Data, Labels, and Benchmarks

### Open Datasets

Open multi-sensor autonomous driving datasets and synthetic scenarios for training and validating reasoning-based driving systems.

| Dataset | Usage | License | Description |
|---------|-------|---------|-------------|
| [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | Training | NVIDIA AV Dataset License | 1,700+ hours of multi-sensor driving data across 25 countries, 306K clips, with 7 synchronized cameras, LiDAR, and radar |
| [PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | Simulation / RL | NVIDIA AV Dataset License | Neural reconstruction scenarios paired to AV clips, the default AlpaGym RL starting dataset, with ClipGT annotations |
| [PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) | SDG | CC-BY-4.0 | 81K synthetic videos with LiDAR and HD-map annotations |
| [PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic) | SDG | CC-BY-4.0 | Cosmos-generated synthetic driving scenarios |

CoC reasoning labels are being added to the Physical AI Open Dataset on Hugging Face. See the [Physical AI Autonomous Vehicles collection](https://huggingface.co/collections/nvidia/physical-ai-autonomous-vehicles).

### Reasoning Labels

Automated reasoning-label generation and open datasets for reasoning-based autonomous driving.

| Tool | Role | Highlights | Resources |
|------|------|------------|-----------|
| **CoC Auto-Labeling Pipeline** | CoC label generation | Three-step keyframe-to-trace pipeline, VLM-based, causally grounded reasoning labels | [NVlabs/alpamayo-coc-autolabeler](https://github.com/NVlabs/alpamayo-coc-autolabeler) |

<details>
<summary><strong>CoC Auto-Labeling Pipeline</strong> — CoC label generation</summary>

NVIDIA CoC Auto-Labeling Pipeline is an open-source auto-labeling pipeline that generates CoC reasoning labels for driving clips automatically. It identifies decision-making keyframes from high-level motion data, runs a VLM pipeline on those keyframes, and organizes cause and effect into decision-grounded, causally linked traces. The output is a strong foundation for training reasoning-based driving models such as Alpamayo 2 Super.

**Three-Step Process:**
1. **Identify the decision-making moment.** Keyframes are defined by critical objects, traffic lights, yield and stop signs, road events, lane lines, and ODD events, detected through sudden changes in egomotion.
2. **Label the explicit driving decision.** Longitudinal and lateral decisions are drawn from a closed action set that eliminates vague behavior descriptions.
3. **Organize cause and effect into a CoC trace.** Causal factors are linked to the resulting decision.

**Highlights:**
- Automated CoC labeling at scale
- Causally grounded traces that reflect the actual reason for each decision
- Closed action set for precise, unambiguous driving decisions
- Used to generate CoC reasoning labels in the NVIDIA Physical AI Open Dataset

**Resources:** [NVlabs/alpamayo-coc-autolabeler](https://github.com/NVlabs/alpamayo-coc-autolabeler)

</details>


### 🏆 Participate in Challenges

> Test your models against the Alpamayo ecosystem benchmarks — open to the community.

| Challenge | Description |
|-----------|-------------|
| [**Physical AI AV Out-of-Distribution Reasoning Challenge 2026**](https://huggingface.co/spaces/nvidia/PhysicalAI-AV-OOD-Reasoning-Challenge-2026) | Evaluate VLA model reasoning on rare and long-tail driving scenarios from the Physical AI dataset |
| [**AlpaSim End-to-End Closed-Loop Challenge 2026**](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026) | Benchmark end-to-end driving policies in closed-loop simulation with AlpaSim |X

---

## Technical Blogs

- [Generate Trajectories, Reasoning Traces, and Auto-labels with NVIDIA Alpamayo 2 Super](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) — NVIDIA Developer Blog
- [Building Autonomous Vehicles That Reason with NVIDIA Alpamayo](https://developer.nvidia.com/blog/building-autonomous-vehicles-that-reason-with-nvidia-alpamayo/) — NVIDIA Developer Blog
- [How to Post-Train Autonomous Vehicle Models in Closed-Loop with NVIDIA Alpamayo](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/) — NVIDIA Developer Blog
- [NVIDIA Alpamayo 2](https://huggingface.co/blog/nvidia/nvidia-alpamayo-2) — Hugging Face Blog
- [NVIDIA Launches Alpamayo 2 Super Open Reasoning Model for Robotaxis](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis) — NVIDIA Newsroom
- [Alpamayo Summit](https://www.nvidia.com/en-us/on-demand/search/?facet.event_name[]=Alpamayo%20Summit&facet.event_year[]=2026) — NVIDIA On Demand
- [Contributing Guidelines](CONTRIBUTING.md)

---

## Contributing

Contributions are welcome, including examples, recipes, and tooling. Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

---

## Support

📣 **Usage questions and discussion**: join us on the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Bugs, documentation issues, and feature requests**: file a [GitHub issue](../../issues/new/choose) using the appropriate template. The relevant NVIDIA responder is auto-assigned.

---

## Security

To report a vulnerability, please contact [security@nvidia.com](mailto:security@nvidia.com) or use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). Do not file security issues publicly.

---

## License

Code in this repository is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details. Model weights and datasets are governed by their respective licenses listed above.

---

**NVIDIA Alpamayo.** Open family for reasoning-based autonomous driving.
