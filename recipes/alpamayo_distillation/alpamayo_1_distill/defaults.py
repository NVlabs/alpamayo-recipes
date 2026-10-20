"""Portable defaults for the Alpamayo 1 distillation recipe.

Filesystem locations for checkpoints, datasets, and outputs are deliberately
not defined here.  Callers provide those locations through CLI arguments.  The
model identifiers below are public Hugging Face repository IDs and can also be
overridden with local paths.
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "config"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config_cosmos_2b.json"
DEFAULT_GOLD_CLIP_IDS_PATH = CONFIG_DIR / "clip_ids_eval_gold.txt"
PAI_GOLD_EVAL_SIZE = 644
PAI_TRAIN_SPLIT_SIZE = 153_625
PAI_GOLD_TRAIN_OVERLAP = 325
PAI_TRAIN_SIZE = 153_300
PAI_FULL_VAL_SIZE = 90_928

DEFAULT_TEACHER_MODEL = "nvidia/Alpamayo-R1-10B"
DEFAULT_COSMOS_STUDENT_MODEL = "nvidia/Cosmos-Reason2-2B"
DEFAULT_TEACHER_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

SUPPORTED_STUDENT_BACKEND = "qwenvl3"
SUPPORTED_STUDENT_TEXT_ARCHITECTURE = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
SUPPORTED_STUDENT_VISION_ARCHITECTURE = {
    "model_type": "qwen3_vl",
    "depth": 24,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "out_hidden_size": 2048,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [5, 11, 17],
}
SUPPORTED_ACTION_EXPERT_ARCHITECTURE = {
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
SUPPORTED_ACTION_STACK_PARAMETER_COUNT = 116_128_002
