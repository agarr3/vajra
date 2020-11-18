from enum import Enum

class ExtractionLibraries(Enum):
    LIBROSA = "librosa"
    PYTORCH_AUDIO = "pyTorch-Audio"

DEFAULT_FEATURE_EXTRACTION_LIBARY = ExtractionLibraries.LIBROSA