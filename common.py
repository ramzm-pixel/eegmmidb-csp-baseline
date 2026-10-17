from pathlib import Path

DEFAULT_DATA_DIR = Path("data") / "eegmmidb"

# Subjects with documented data-quality problems in EEGMMIDB:
# S088, S092, S100 were recorded at 128 Hz instead of 160 Hz; S089 has corrupted annotations.
BAD_SUBJECTS = {"S088", "S089", "S092", "S100"}

CONDITIONS = ("motor_raw", "motor_ica", "hybrid_raw", "hybrid_ica")

TASKS = {
    "real_lr": ("left_fist_real", "right_fist_real"),
    "imagined_lr": ("left_fist_imagined", "right_fist_imagined"),
    "real_feet": ("both_fists_real", "both_feet_real"),
    "imagined_feet": ("both_fists_imagined", "both_feet_imagined"),
}

MOTOR_CHANNELS = [
    "C3", "C4", "CZ", "C1", "C5", "FC1", "FC3", "FC5",
    "CP1", "CP3", "CP5", "C2", "C6", "FC2", "FC4", "FC6",
    "CP2", "CP4", "CP6", "FCZ", "CPZ",
]

OCULAR_ARTIFACT_CHANNELS = [
    "FP1", "FP2", "FPZ", "AF3", "AF4", "AF7", "AF8", "AFZ",
    "FZ", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FT7", "FT8", "T7", "T8",
]
