# Skin Lesion Classification Project

A deep learning-based web application for classifying skin lesions as **Benign** or **Malignant** using a ResNet-18 model.

## Overview

This project implements a binary classification system for skin lesion analysis. It features:
- A ResNet-18 based deep learning model trained on skin lesion images
- A Flask backend API for serving predictions
- A web-based frontend for easy image upload and result visualization

## Project Structure

```
skin_lesion_project/
├── backend/                # Flask API server
│   ├── app.py             # Main Flask application
│   ├── model.py           # Model architecture definition
│   └── verify_setup.py    # Environment verification script
├── Model/                  # Trained model files
│   └── skin_cancer/
│       └── best_skin_cancer_model.pth
├── Website/                # Frontend files
│   └── test.html          # Web interface
├── Sample/                 # Sample images for testing
└── README.md              # This file
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/adi-harsh007/Skin_lesion.git
   cd skin_lesion_project
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision flask flask-cors pillow
   ```

3. **Verify setup** (optional)
   ```bash
   cd backend
   python verify_setup.py
   ```

### Running the Application

1. **Start the Flask backend**
   ```bash
   cd backend
   python app.py
   ```
   The server will start on `http://localhost:5000`

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`

3. **Notes**
   - Use virtual environment for better management of dependencies

## Usage

1. **Upload an image**: Click the upload area or drag and drop a skin lesion image
2. **Get prediction**: The model will classify the lesion as Benign or Malignant
3. **View results**: See the prediction with confidence score

## Model Details

- **Architecture**: ResNet-18 (pretrained on ImageNet)
- **Task**: Binary classification (Benign vs. Malignant)
- **Input**: 224x224 RGB images
- **Output**: Probability score with sigmoid activation
- **Modifications**: 
  - Custom fully connected layer with dropout (0.5)
  - Fine-tuned for skin lesion classification

## API Endpoints

### `GET /`
Serves the frontend HTML interface

### `POST /predict`
Classifies uploaded skin lesion images

**Request**: 
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response**:
```json
{
  "prediction": "Benign" | "Malignant",
  "confidence": "95.34%",
  "probability": 0.9534
}
```

## Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision models and transforms
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **Pillow**: Image processing

## Important Notes

- This is an educational project and should **NOT** be used for actual medical diagnosis
- Always consult qualified healthcare professionals for medical concerns
- The model's predictions are probabilistic and may not be 100% accurate

## Academic Project

This project was created as part of the 6th semester coursework.

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

---

**Note**: Ensure the model file (`best_skin_cancer_model.pth`) is present in the `Model/skin_cancer/` directory before running the application.
