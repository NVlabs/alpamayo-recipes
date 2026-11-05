import os
from enum import IntEnum
from typing import Final

CROSS_LEFT_CAMERA_NAME: Final = "camera_cross_left_120fov"
CROSS_RIGHT_CAMERA_NAME: Final = "camera_cross_right_120fov"
FRONT_TELE_CAMERA_NAME: Final = "camera_front_tele_30fov"
FRONT_WIDE_CAMERA_NAME: Final = "camera_front_wide_120fov"
REAR_LEFT_CAMERA_NAME: Final = "camera_rear_left_70fov"
REAR_RIGHT_CAMERA_NAME: Final = "camera_rear_right_70fov"
REAR_TELE_CAMERA_NAME: Final = "camera_rear_tele_30fov"

# Camera indices will be used for computing camera embeddings.
CAMERA_NAMES_TO_INDICES = {
    CROSS_LEFT_CAMERA_NAME: 0,
    FRONT_WIDE_CAMERA_NAME: 1,
    CROSS_RIGHT_CAMERA_NAME: 2,
    REAR_LEFT_CAMERA_NAME: 3,
    REAR_TELE_CAMERA_NAME: 4,
    REAR_RIGHT_CAMERA_NAME: 5,
    FRONT_TELE_CAMERA_NAME: 6,
}
CAMERA_NAMES = tuple(name for name in CAMERA_NAMES_TO_INDICES)
CAMERA_INDICES_TO_NAMES = {idx: name for name, idx in CAMERA_NAMES_TO_INDICES.items()}
CAMERA_NAMES_TO_DISPLAY_NAMES = {
    CROSS_LEFT_CAMERA_NAME: "Front left camera",
    CROSS_RIGHT_CAMERA_NAME: "Front right camera",
    FRONT_WIDE_CAMERA_NAME: "Front camera",
    FRONT_TELE_CAMERA_NAME: "Front telephoto camera",
    REAR_LEFT_CAMERA_NAME: "Rear left camera",
    REAR_TELE_CAMERA_NAME: "Rear camera",
    REAR_RIGHT_CAMERA_NAME: "Rear right camera",
}
CAMERA_INDICES_TO_DISPLAY_NAMES = {
    idx: CAMERA_NAMES_TO_DISPLAY_NAMES[name] for name, idx in CAMERA_NAMES_TO_INDICES.items()
}

INFERENCE_PLATFORM_IN_CAR: Final = "IN_CAR"
INFERENCE_PLATFORM_LOCAL_REPLAY: Final = "LOCAL_REPLAY"

# fps for v2 data.
VIDEO_FPS = 30

# minimum start index.
# NOTE V2 data might have issue in the first few frames, set a minimum to exclude them.
MIN_START_INDEX = 3

# Roughly-estimated car dimensions (length, width, height) in meters.
# For full accuracy, the actual ego-vehicle dimensions for each clip can be found
# in the rig.json file, as rig["rig"]["vehicle"]["value"]["body"].
EGO_VEHICLE_LWH = (4.0, 3.0, 2.0)
# position of ego coordinate frame along the length of the car
# (distance of rear axle from back bumper / the length of the car)
# true value can be computed from rig.json file using
# -rig["rig"]["vehicle"]["value"]["body"]["boundingBoxPosition"][0] / ego_length
# we assume it is centered along the width, and so far are not using the z offset
EGO_VEHICLE_LENGTH_OFFSET = 0.21

# Mapping between raw object sub-categories loaded from the data and grouped object
# class categories used for training the detector
CLASS_TO_SUBCATEGORY_INV = {
    "automobile": "Car",
    "trailer": "Car",
    "truck": "Truck",
    "heavy_truck": "Truck",
    "bus": "Truck",
    "train_or_tram_car": "Truck",
    "other_vehicle": "Truck",
    "pedestrian": "Pedestrian",
    "person": "Pedestrian",
    "person_group": "Pedestrian",
    "bicycle": "Cyclist",
    "stroller": "Cyclist",
    "bicycle_with_rider": "Cyclist",
    "motorcycle_with_rider": "Cyclist",
    "motorcycle": "Cyclist",
    "rider": "Cyclist",
    "cycle": "Others",
    "protruding_object": "Others",
    "animal": "Others",
    "unknown": "Others",
}
CLASS_TO_SUBCATEGORY = {
    "Car": "automobile",
    "Truck": "truck",
    "Pedestrian": "pedestrian",
    "Cyclist": "bicycle",
    "Others": "unknown",
}
CLASSES = list(CLASS_TO_SUBCATEGORY.keys())
CLASSES_SUB = list(CLASS_TO_SUBCATEGORY_INV.keys())

LIDAR_NAME: Final = "lidar_gt_top_p128"
"""Identifier for the LiDAR sensor in the dataset."""

# Alpackage storage settings. Each can be overridden with an environment variable of the same
# name prefixed with ALPAMAYO_, e.g. ALPAMAYO_S3_ENDPOINT_URL.
ALPACKAGES_BUCKET: Final = os.environ.get("ALPAMAYO_ALPACKAGES_BUCKET", "alpackages")
"""The name of the S3/SwiftStack bucket/container where alpackages are stored."""

ALPACKAGE_ENDPOINT_URL: Final = os.environ.get(
    "ALPAMAYO_ALPACKAGE_ENDPOINT_URL",
    f"s3://{ALPACKAGES_BUCKET}/pt/{{alpackage_name}}_{{alpackage_sha256}}.tar",
)
"""The endpoint URL for the alpackages in the S3/SwiftStack bucket/container."""

S3_REGION: Final = os.environ.get("ALPAMAYO_S3_REGION", "us-east-1")
"""The region of the S3/SwiftStack bucket/container."""

S3_ENDPOINT_URL: Final = os.environ.get("ALPAMAYO_S3_ENDPOINT_URL")
"""The endpoint URL for the S3/SwiftStack bucket/container. Unset uses the S3 client's default."""

MAX_IMAGE_CONTAINER_BATCH_SIZE: Final[int] = 10
"""Max Number of historical images to be sent back to NDAS Roadcast for logging"""

MAX_EGO_CONTAINER_BATCH_SIZE: Final[int] = 400
"""Max Number of historical egomotion datapoints to be sent back to NDAS Roadcast for logging"""

MAX_PROFILE_ARRAY_SIZE: Final[int] = 10
"""Max Number of profiling timestamps to be sent back to NDAS Roadcast for logging"""


MAX_UNSELECTED_TRAJECTORIES: Final[int] = 7
"""Max Number of alternative trajectories allowed by NDAS"""


# Integer enums for laneline styles
class LaneLineStyle(IntEnum):
    """Integer enums for laneline styles."""

    SOLID_SINGLE = 0
    SOLID_GROUP = 1
    DASHED_SINGLE = 2
    DASHED_GROUP = 3
    ZIGZAG = 4
    PHYSICAL = 5
    VIRTUAL = 6
    UNKNOWN = 7
    SOLID_DASHED = 8
    DASHED_SOLID = 9


# Integer enums for laneline colors
class LaneLineColor(IntEnum):
    """Integer enums for laneline colors."""

    WHITE = 0
    YELLOW = 1
    OTHER = 2


# Integer enums for road boundary styles
class RoadBoundaryStyle(IntEnum):
    """Integer enums for road boundary styles."""

    UNKNOWN = 0
    BARRIER = 1
    CURB = 2
    SEAM = 3
    GUIDEPOST = 4
    WALL = 5


# Point styles for laneline using integer enums
STYLES_FOR_LANELINE = {
    "solid_single": LaneLineStyle.SOLID_SINGLE,
    "dot_solid_single": LaneLineStyle.SOLID_SINGLE,
    "area": LaneLineStyle.SOLID_SINGLE,
    "solid_group": LaneLineStyle.SOLID_GROUP,
    "dot_solid_group": LaneLineStyle.SOLID_GROUP,
    "short_dashed_single": LaneLineStyle.DASHED_SINGLE,
    "long_dashed_single": LaneLineStyle.DASHED_SINGLE,
    "dot_dashed_single": LaneLineStyle.DASHED_SINGLE,
    "thick_dashed": LaneLineStyle.DASHED_SINGLE,
    "short_dashed_group": LaneLineStyle.DASHED_GROUP,
    "long_dashed_group": LaneLineStyle.DASHED_GROUP,
    "zipzag_group": LaneLineStyle.ZIGZAG,
    "road_boundary": LaneLineStyle.PHYSICAL,
    "tall_curb": LaneLineStyle.PHYSICAL,
    "low_curb": LaneLineStyle.PHYSICAL,
    "fence": LaneLineStyle.PHYSICAL,
    "barrier": LaneLineStyle.PHYSICAL,
    "wall": LaneLineStyle.PHYSICAL,
    "obstacle_bars": LaneLineStyle.PHYSICAL,
    "other_physical": LaneLineStyle.PHYSICAL,
    "virtual": LaneLineStyle.VIRTUAL,
    "other_marking": LaneLineStyle.UNKNOWN,
    "unknown": LaneLineStyle.UNKNOWN,
    "other": LaneLineStyle.UNKNOWN,
    "solid_dashed": LaneLineStyle.SOLID_DASHED,
    "dashed_solid": LaneLineStyle.DASHED_SOLID,
}

# Point colors for laneline using integer enums
COLORS_FOR_LANELINE = {
    "white": LaneLineColor.WHITE,
    "yellow": LaneLineColor.YELLOW,
    "blue": LaneLineColor.OTHER,
    "red": LaneLineColor.OTHER,
    "green": LaneLineColor.OTHER,
    "gray": LaneLineColor.OTHER,
    "orange": LaneLineColor.OTHER,
    "mixed": LaneLineColor.OTHER,
    "unknown": LaneLineColor.OTHER,
}

# Draw colors for laneline using integer enums
COLORS_FOR_LANELINE_VS = {
    LaneLineColor.WHITE: "white",
    LaneLineColor.YELLOW: "yellow",
    LaneLineColor.OTHER: "blue",
}

# Draw line styles for laneline using integer enums
STYLES_FOR_LANELINE_VS = {
    LaneLineStyle.SOLID_SINGLE: "-",
    LaneLineStyle.SOLID_GROUP: "-",
    LaneLineStyle.DASHED_SINGLE: "--",
    LaneLineStyle.DASHED_GROUP: "--",
    LaneLineStyle.ZIGZAG: "-",
    LaneLineStyle.PHYSICAL: "-",
    LaneLineStyle.VIRTUAL: ":",
    LaneLineStyle.UNKNOWN: "-",
    LaneLineStyle.SOLID_DASHED: "-.",
    LaneLineStyle.DASHED_SOLID: "-.",
}

# Mapping for road boundary styles
STYLES_FOR_ROADBOUNDARY = {
    "other_physical": RoadBoundaryStyle.UNKNOWN,
    "fence": RoadBoundaryStyle.UNKNOWN,
    "barrier": RoadBoundaryStyle.BARRIER,
    "low_curb": RoadBoundaryStyle.CURB,
    "tall_curb": RoadBoundaryStyle.CURB,
    "road_boundary": RoadBoundaryStyle.SEAM,
    "obstacle_bars": RoadBoundaryStyle.GUIDEPOST,
    "wall": RoadBoundaryStyle.WALL,
}


class MetaAction(IntEnum):
    NotAvailable = 0
    Accelerate = 1
    Decelerate = 2
    KeepSpeed = 3
    Wait = 4
    Straight = 5
    LeftTurn = 6
    RightTurn = 7
    KeepLane = 8
    LeftLaneChange = 9
    RightLaneChange = 10
    Reverse = 11


META_ACTION_STRINGS = {
    MetaAction.NotAvailable: "Not Available",
    MetaAction.Accelerate: "Accelerate",
    MetaAction.Decelerate: "Decelerate",
    MetaAction.KeepSpeed: "Keep Speed",
    MetaAction.Wait: "Wait",
    MetaAction.Straight: "Straight",
    MetaAction.LeftTurn: "Left Turn",
    MetaAction.RightTurn: "Right Turn",
    MetaAction.KeepLane: "Keep Lane",
    MetaAction.LeftLaneChange: "Left Lane Change",
    MetaAction.RightLaneChange: "Right Lane Change",
    MetaAction.Reverse: "Reverse",
}


# Make sure values start from 0 and are consecutive as they will be used as indices.
META_ACTION_GROUPS = {
    MetaAction.Accelerate.value: 0,
    MetaAction.Decelerate.value: 0,
    MetaAction.KeepSpeed.value: 0,
    MetaAction.Reverse.value: 0,
    MetaAction.Wait.value: 0,
    MetaAction.Straight.value: 1,
    MetaAction.LeftTurn.value: 1,
    MetaAction.RightTurn.value: 1,
    MetaAction.KeepLane.value: 2,
    MetaAction.LeftLaneChange.value: 2,
    MetaAction.RightLaneChange.value: 2,
}
NUM_META_ACTION_GROUPS = len(set(META_ACTION_GROUPS.values()))
META_ACTION_GROUP_STRINGS = {
    0: "longitudinal",
    1: "lateral",
    2: "lane",
}
