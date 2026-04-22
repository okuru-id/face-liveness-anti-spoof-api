from enum import Enum


class Verdict(str, Enum):
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    UNCERTAIN = "UNCERTAIN"
    NO_FACE = "NO_FACE"
    POOR_QUALITY = "POOR_QUALITY"


class SpoofType(str, Enum):
    PRINT_ATTACK = "PRINT_ATTACK"
    SCREEN_REPLAY = "SCREEN_REPLAY"
    DEEPFAKE = "DEEPFAKE"
    UNKNOWN = "UNKNOWN"
