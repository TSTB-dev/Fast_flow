MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

JELLY_CATEGORIES = [
    'glove1',
    'glove95',
    'glove100',
    'hair1',
    'hair95',
    'hair100',
    'larva4',
    'larva40',
    'larva56',
    'metal99',
    'metal2',
    'metal92',
    'all'
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

BATCH_SIZE = 8
NUM_EPOCHS = 1
LR = 1e-3
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 1
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 1