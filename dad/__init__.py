from .logging import logger as logger
from .logging import configure_logger as configure_logger
import os
from .detectors import load_DaD as load_DaD
from .detectors import dedode_detector_S as dedode_detector_S
from .detectors import dedode_detector_B as dedode_detector_B
from .detectors import dedode_detector_L as dedode_detector_L
from .detectors import load_DaDDark as load_DaDDark
from .detectors import load_DaDLight as load_DaDLight
from .types import Detector as Detector
from .types import Matcher as Matcher
from .types import Benchmark as Benchmark

configure_logger()
DEBUG_MODE = bool(os.environ.get("DEBUG", False))
RANK = 0
GLOBAL_STEP = 0
