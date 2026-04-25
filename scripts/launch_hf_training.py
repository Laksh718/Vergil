import os
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

# Load token from .env
load_dotenv()

TOKEN = os.getenv("HF_TOKEN")
USERNAME = "thekrishdshah"
SPACE_NAME = "vergil-sota-trainer"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"

if not TOKEN:
    print("Error: HF_TOKEN not found in .env file.")
    exit(1)

api = HfApi()

def launch():
    print(f"Initializing Hugging Face Space: {REPO_ID}...")
    
    try:
        # 1. Create the Space (Docker SDK)
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            token=TOKEN,
            exist_ok=True
        )
        print("Space created/verified.")

        # 2. Add HF_TOKEN as a Secret for the Space
        api.add_space_secret(
            repo_id=REPO_ID,
            key="HF_TOKEN",
            value=TOKEN,
            token=TOKEN
        )
        print("HF_TOKEN Secret added.")

        # 3. Upload project (IMPORTANT: Ignore .env to avoid HF security block)
        print("Uploading project files (ignoring secret files)...")
        upload_folder(
            folder_path=".",
            repo_id=REPO_ID,
            repo_type="space",
            token=TOKEN,
            ignore_patterns=[
                ".git*", 
                "node_modules*", 
                "venv*", 
                "__pycache__*", 
                ".env*", 
                "scripts/launch_hf_training.py"
            ]
        )
        print("Project files uploaded.")

        # 4. Set Hardware to T4 (Cost-effective for 0.5B model)
        print("Requesting T4 GPU hardware...")
        api.request_space_hardware(
            repo_id=REPO_ID,
            hardware="t4-small",
            token=TOKEN
        )
        print(f"SUCCESS! Training launched at: https://huggingface.co/spaces/{REPO_ID}")
        print("Monitoring logs on Hugging Face is recommended.")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    launch()
