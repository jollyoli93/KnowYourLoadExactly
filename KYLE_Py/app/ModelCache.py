import threading
from pathlib import Path
import onnxruntime as rt

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path("./app/Models/crane_models/crane_classifier.onnx")

class ModelCache:
    _session = None
    _model_path = str(DEFAULT_MODEL_PATH)
    _lock = threading.Lock()

    @classmethod
    def get_path(cls) -> str:
        return cls._model_path

    @classmethod
    def set_path(cls, path):
        if path:
            new_path = str(Path(path).resolve())
            if new_path != cls._model_path:
                with cls._lock:
                    cls._model_path = new_path
                    cls._session = None  # Invalidate session on path change

    @classmethod
    def get_session(cls, path=None) -> rt.InferenceSession:
        if path:
            cls.set_path(path)

        if cls._session is None:
            with cls._lock:
                if cls._session is None:
                    cls._session = rt.InferenceSession(
                        cls._model_path, 
                        providers=['CPUExecutionProvider']
                    )
        return cls._session