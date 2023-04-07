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
BACKBONE_DEIT_224 = "deit_base_distilled_patch16_224"
BACKBONE_DEITS_224 = "deit_tiny_distilled_patch16_224"
BACKBONE_MOBILEVIT_V2 = "deit_small_distilled_patch16_224"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_RESNET50 = "resnet50"
BACKBONE_RESNET101 = "resnet101"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"
BACKBONE_WIDE_RESNET101 = "wide_resnet101_2"
BACKBONE_EFFICIENTNET = "efficientnet_b4"
BACKBONE_CONVNEXT_BASE = "convnext_base"
BACKBONE_CONVNEXT_SMALL = "convnext_small"
BACKBONE_CONVNEXT_TINY = "convnext_tiny"
BACKBONE_CONVNEXT_TINY_IN22K = "convnext_tiny_in22k"
BACKBONE_CONVNEXT_SMALL_IN22K = "convnext_small_in22k"
BACKBONE_SWIN_BASE_PATCH4_WINDOWS7_224 = "swin_base_patch4_window7_224"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_DEIT_224,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_RESNET50,
    BACKBONE_WIDE_RESNET50,
    BACKBONE_DEITS_224,
    BACKBONE_MOBILEVIT_V2,
    BACKBONE_EFFICIENTNET,
    BACKBONE_WIDE_RESNET101,
    BACKBONE_RESNET101,
    BACKBONE_CONVNEXT_BASE,
    BACKBONE_CONVNEXT_SMALL,
    BACKBONE_CONVNEXT_TINY,
    BACKBONE_CONVNEXT_TINY_IN22K,
    BACKBONE_CONVNEXT_SMALL_IN22K,
    BACKBONE_SWIN_BASE_PATCH4_WINDOWS7_224
]

BATCH_SIZE = 8
NUM_EPOCHS = 1
LR = 1e-3
WEIGHT_DECAY = 0  # 1e-5
LOG_INTERVAL = 50
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 25

# ランダムサンプリングする際のhyperparameter
STEP = 25
NUM_PATCHES = 25

WHOLE_IMAGE = 'C:\\Users\s2231\PycharmProjects\FastFlow\Fast_flow\FastFlow\whole_label.png'