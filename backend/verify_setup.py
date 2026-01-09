import torch
import torchvision
import os
import sys

print("Checking environment...")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

try:
    from model import SkinCancerClassifier
    print("Successfully imported SkinCancerClassifier model class.")
except ImportError as e:
    print(f"Error importing model class: {e}")
    sys.exit(1)

model_path = r"d:/6th SEM/skin_lesion_project/Model/skin_cancer/best_skin_cancer_model.pth"
if os.path.exists(model_path):
    print(f"Model file found at {model_path}")
    try:
        model = SkinCancerClassifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
else:
    print(f"Model file NOT found at {model_path}")

print("\nSetup verification complete.")
