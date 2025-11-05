import os
from huggingface_hub import HfApi, create_repo

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/workspace/LLaDA-8B-Instruct",
    repo_id="LoveFlowers793/MouseMDM",
    repo_type="model",
)
    
