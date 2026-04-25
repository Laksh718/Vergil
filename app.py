
import os
import subprocess
print("🔥 Booting VERGIL Remote GRPO Training Cluster on Hugging Face Spaces...")
print("Note: If this is running on a CPU Space, it will fail. Ensure you have allocated a T4/A10G GPU in the Space Settings.")
# Inject the huggingface token for the upload step
HF_TOKEN = os.getenv("HF_TOKEN")

# Run the primary training script
try:
    subprocess.run(['python', 'scripts/train_grpo_colab.py'], check=True)
except Exception as e:
    print(f"Training crashed: {e}")
    
# Keep space open so logs can be read
while True:
    pass
