from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import sys
import os

# Add current directory to path so we can import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import SkinCancerClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = r"d:/6th SEM/skin_lesion_project/Model/skin_cancer/best_skin_cancer_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
print(f"Loading model from {MODEL_PATH}...")
try:
    model = SkinCancerClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists at the specified path.")

# Preprocessing transform (same as Validation transform in notebook)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    """Serve the frontend HTML page"""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Website', 'test.html')
    return send_file(html_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        tensor = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension
        
        # Inference
        with torch.no_grad():
            output = model(tensor)
            probability = torch.sigmoid(output).item()
            
        # Interpret result
        # Logic: Output > 0.5 (logit) or > 0.5 (prob) depends on training.
        # Notebook used: predictions = (torch.sigmoid(outputs) > 0.5).float()
        # Malignant = 1, Benign = 0
        
        is_malignant = probability > 0.5
        prediction = "Malignant" if is_malignant else "Benign"
        
        # Confidence:
        # If malignant (prob > 0.5), confidence is prob.
        # If benign (prob <= 0.5), confidence is 1 - prob.
        confidence = probability if is_malignant else 1 - probability
        
        result = {
            'prediction': prediction,
            'confidence': f"{confidence * 100:.2f}%",
            'probability': probability
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
