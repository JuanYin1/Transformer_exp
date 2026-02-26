import wandb
from huggingface_hub import whoami, model_info
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
import os
from dotenv import load_dotenv

# ==========================================
# INSERT YOUR KEYS HERE
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_KEY = os.getenv("WANDB_API_KEY")

print("==========================================")
print("1. TESTING W&B CONNECTION")
print("==========================================\n")

try:
    # Attempt to log in to W&B
    wandb.login(key=WANDB_KEY, verify=True)
    
    # Create a tiny, temporary project to ensure write-access
    run = wandb.init(project="connection-test", name="test-run")
    print("\n✅ W&B Success! Connected to your account and created a test run.")
    
    # Close it immediately so it doesn't hang
    run.finish()
except Exception as e:
    print(f"\n❌ W&B Failed! Error: {e}")


print("\n==========================================")
print("2. TESTING HUGGING FACE CONNECTION")
print("==========================================\n")

try:
    # Check if the token is valid and who it belongs to
    user_info = whoami(token=HF_TOKEN)
    print(f"✅ HF Token is valid! Logged in as: {user_info['name']}")
    
    # Check if this token has the power to see the Gated Llama 3.1 model
    print("Checking access to the gated Llama-3.1-8B model...")
    model = model_info("meta-llama/Meta-Llama-3.1-8B-Instruct", token=HF_TOKEN)
    
    print("✅ HF Llama Access Success! Your token has permission to download the model.")

except GatedRepoError:
    print("\n❌ HF Failed! Your token is valid, but you have not been granted access to Llama 3.1 yet. Did you sign the agreement on the model page?")
except RepositoryNotFoundError:
    print("\n❌ HF Failed! The token cannot find the model. Make sure you checked 'Read access to contents of all public gated repos you can access' when making the token.")
except Exception as e:
    print(f"\n❌ HF Failed! Invalid token or connection error: {e}")

print("\n==========================================")
print("TEST COMPLETE")
print("==========================================")