from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "ptb-xl-dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure processed directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Parameters
BATCH_SIZE = 64
SAMPLE_RATE = 100
TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Architectural Paramaters
CHANNELS = 12
RECORDING_LENGTH_SEC = 10
SEQ_LEN = SAMPLE_RATE * RECORDING_LENGTH_SEC

# Signal Processing Hyperparameters
FILTER_LOWCUT = 0.5
FILTER_HIGHCUT = 40.0
FILTER_ORDER = 4

# Cross-Validation Folds
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLD = 9
TEST_FOLD = 10
