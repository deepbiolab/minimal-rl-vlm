from huggingface_hub import snapshot_download
import os

# Get the parent directory of the current script directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define model list
models = [
    # "Qwen2.5-VL-7B-Instruct",
    # "Qwen2-VL-7B-Instruct", 
    # "Qwen2.5-VL-3B-Instruct",
    "Qwen2-VL-2B-Instruct"
]

# Download each model and save to models folder
for model in models:
    snapshot_download(
        repo_id=f"Qwen/{model}",
        repo_type="model",
        local_dir=os.path.join(base_dir, "checkpoints", model),  # Use relative path
        local_dir_use_symlinks=False  # Use hard links instead of symbolic links
    )
