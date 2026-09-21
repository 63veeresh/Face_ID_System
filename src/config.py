"""
Central configuration for the Face Recognition & Identification System.
Provides paths, model hyperparameters, detection parameters, and recognition thresholds.
"""

from pathlib import Path
import torch

# Base project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ENROLLED_DIR = DATA_DIR / "enrolled"
DATABASE_FILE = DATA_DIR / "database.json"
EVAL_DIR = DATA_DIR / "eval_set"
TEST_SAMPLES_DIR = DATA_DIR / "test_samples"

# Ensure essential directories exist
ENROLLED_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Hardware acceleration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Face Detection (MTCNN) settings
FACE_IMAGE_SIZE = 160  # Required input size for InceptionResnetV1
DETECTION_MARGIN = 20  # Margin to add around detected face bounding box
DETECTION_MIN_CONFIDENCE = 0.85  # Minimum detection probability
MTCNN_MIN_FACE_SIZE = 40  # Minimum face size in pixels

# Face Embedding (InceptionResnetV1) settings
EMBEDDING_DIM = 512
PRETRAINED_WEIGHTS = "vggface2"  # VGGFace2 provides robust diverse facial features

# Recognition & Threshold settings
# For cosine similarity on L2-normalized 512-D VGGFace2 embeddings:
# Same person: typically >= 0.70 - 0.85
# Different person: typically <= 0.40 - 0.55
# Default threshold 0.65 strikes an optimal balance between False Accept Rate (FAR) and False Reject Rate (FRR)
DEFAULT_SIMILARITY_THRESHOLD = 0.65

# Unknown label
UNKNOWN_LABEL = "UNKNOWN"

# UI / Visualization styling
BOX_COLOR_MATCH = (40, 167, 69)     # Emerald Green in RGB
BOX_COLOR_UNKNOWN = (220, 53, 69)   # Crimson Red in RGB
TEXT_COLOR = (255, 255, 255)
