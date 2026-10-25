import threading
from pathlib import Path
import onnxruntime as rt

BASE_DIR = Path(__file__).resolve().parent
CLASSIFY_MODEL_PATH = Path("./app/Models/crane_models/crane_classifier.onnx")
ROI_MODEL_PATH = Path("./app/Models/crane_models/LoadRegressionAdam250.onnx")

class ModelCache:
    _classify_session = None
    _roi_session = None

    _classify_model_path = str(CLASSIFY_MODEL_PATH)
    _roi_model_path = str(ROI_MODEL_PATH)
    _lock = threading.Lock()

    @classmethod
    def get_classify_path(cls) -> str:
        return cls._classify_model_path

    @classmethod
    def set_classify_path(cls, path):
        if path:
            new_path = str(Path(path).resolve())
            if new_path != cls._classify_model_path:
                with cls._lock:
                    cls._classify_model_path = new_path
                    cls._session = None  # Invalidate session on path change
    @classmethod
    def get_roi_path(cls) -> str:
        return cls._classify_model_path

    @classmethod
    def set_roi_path(cls, path):
        if path:
            new_path = str(Path(path).resolve())
            if new_path != cls._roi_model_path:
                with cls._lock:
                    cls._roi_model_path = new_path
                    cls._session = None  # Invalidate session on path change
    @classmethod
    def get_classify_session(cls, path=None) -> rt.InferenceSession:
        if path:
            cls.set_classify_path(path)

        if cls._classify_session is None:
            with cls._lock:
                if cls._classify_session is None:
                    cls._classify_session = rt.InferenceSession(
                        cls._classify_model_path, 
                        providers=['CPUExecutionProvider']
                    )
        return cls._classify_session
    
    @classmethod
    def get_roi_session(cls, path=None) -> rt.InferenceSession:
        if path:
            cls.set_roi_path(path)

        if cls._roi_session is None:
            with cls._lock:
                if cls._roi_session is None:
                    cls._roi_session = rt.InferenceSession(
                        cls._roi_model_path, 
                        providers=['CPUExecutionProvider']
                    )
        return cls._roi_session