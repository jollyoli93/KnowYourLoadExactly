from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = Path(__file__).resolve().parent

# Download entire repository
model_dir = snapshot_download(
    repo_id="jollyoli93/KYLE",
    local_dir=BASE_DIR/ "Models"
)