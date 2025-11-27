import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Force TensorFlow to use legacy Keras 2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import io

# Create FastAPI app
app = FastAPI(
    title="Animal Classifier API",
    description="MLOps deployment of animal classification model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define animal labels
ANIMALS = ['Cat', 'Dog', 'Panda']
model = None

def load_model():
    """Load the model from the local filesystem"""
    global model
    
    if model is not None:
        return model
    
    # Model will be copied here during Docker build
    model_path = './model/named-outputs/model/animal-cnn/model.keras'
    
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    return model

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("Starting up - loading model...")
    try:
        load_model()
        print("Startup complete!")
    except Exception as e:
        print(f"WARNING: Failed to load model on startup: {e}")
        print("Model will be loaded on first prediction request")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Animal Classifier API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #333; }
                .info { background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }
                code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>🐾 Animal Classifier API</h1>
            <div class="info">
                <h2>Welcome to the MLOps Animal Classifier!</h2>
                <p>This model was trained using Azure Machine Learning and deployed via GitHub Actions.</p>
                <p><strong>Available endpoints:</strong></p>
                <ul>
                    <li><code>GET /</code> - This page</li>
                    <li><code>GET /health</code> - Health check</li>
                    <li><code>POST /predict</code> - Upload image for classification</li>
                    <li><code>GET /docs</code> - Interactive API documentation</li>
                </ul>
                <p><a href="/docs">📖 View Interactive API Documentation</a></p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    global model
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": ANIMALS
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Upload an image file to classify it as Cat, Dog, or Panda
    """
    global model
    
    # Ensure model is loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            return {
                "success": False,
                "error": f"Model not loaded: {str(e)}"
            }
    
    try:
        # Read image
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents))
        
        # Resize to 64x64
        resized_image = original_image.resize((64, 64))
        
        # Convert to array and prepare for prediction
        image_array = np.array(resized_image)
        images_to_predict = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(images_to_predict, verbose=0)
        prediction_probabilities = predictions[0]
        classification = prediction_probabilities.argmax()
        
        return {
            "success": True,
            "filename": file.filename,
            "predicted_class": ANIMALS[classification],
            "confidence": float(prediction_probabilities[classification]),
            "all_probabilities": {
                animal: float(prob) 
                for animal, prob in zip(ANIMALS, prediction_probabilities)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/model-info")
async def model_info():
    return {
        "model_source": "Azure Machine Learning",
        "model_name": "animal-classification",
        "model_type": "CNN (Convolutional Neural Network)",
        "classes": ANIMALS,
        "input_size": "64x64 RGB images",
        "framework": "TensorFlow/Keras",
        "deployment": "MLOps Pipeline via GitHub Actions"
    }