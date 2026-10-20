# Constants for the VLA models

IGNORE_LABEL_ID = -100
# lagecy name for the same purpose as IGNORE_LABEL_ID, make sure two have some value.
IGNORE_INDEX = -100

EXTRA_TOKEN_KEYS = dict(
    vision_keys=["image_pre_tkn"],  # placeholder for continuous camera/ts info
    traj_keys=[
        "traj_history_start",
        "traj_history_pre_tkn",
        "traj_history",
        "traj_history_end",
        "traj_future_start",
        "traj_future_pre_tkn",
        "traj_future",
        "traj_future_end",
    ],
    cot_keys=["cot_start", "cot_end"],
    meta_action_keys=["meta_action_start", "meta_action_end"],
    vectorized_wm_keys=[
        "vectorized_wm_start",
        "vectorized_wm_pre_tkn",
        "vectorized_wm",
        "vectorized_wm_end",
    ],
    route_keys=["route_start", "route_pad", "route_end"],
)


def name_to_token(name: str) -> str:
    """Convert a str name to a token."""
    return "<|" + name + "|>"


_keys = [key for sublist in EXTRA_TOKEN_KEYS.values() for key in sublist]
EXTRA_TOKENS = {k: name_to_token(k) for k in _keys}

#######################################################################################
# deprecated
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}
# List of all special token keys
SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    # Following three are just placeholders whose embeddings are not used/trained
    "traj_history",
    "traj_future",
    "image_pad",
    # The following four are used for vectorized world model
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    # The following three are used for route model
    "route_start",
    "route_pad",
    "route_end",
    # The following four are used for VQA model
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS = {k: "<|" + k + "|>" for k in SPECIAL_TOKENS_KEYS}
