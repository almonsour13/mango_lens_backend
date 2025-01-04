import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import base64
import cv2
from typing import List, Dict

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for request
class ImageRequest(BaseModel):
    image: str

# Pydantic model for response
class PredictionResponse(BaseModel):
    predictions: List[Dict]
    boundingBoxes: List[Dict]
    originalImage: str
    analyzedImage: str

# Rest of your model setup remains the same
img_height = 224
img_width = 224

model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.layers[0].trainable = False
model.compile(Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.load_weights('model/mango_leaf_classification_model_weights_omdena_resnet50.hdf5')

base_model = model.layers[0]
layer_name = 'conv5_block3_3_conv'
grad_model = tf.keras.models.Model([base_model.inputs], [base_model.get_layer(layer_name).output, base_model.output])

classes = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

# Your helper functions remain the same
def process_heatmaps_and_generate_boxes(img, heatmaps, predictions):
    # Your existing implementation
    pass

def predict_and_generate_heatmaps(img_array):
    # Your existing implementation
    pass

def decode_base64_image(base64_string):
    # Your existing implementation
    pass

def encode_base64_image(img):
    # Your existing implementation
    pass

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        imageData = decode_base64_image(request.image)

        img = cv2.resize(imageData, (img_height, img_width))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = np.expand_dims(imgRGB, axis=0)
        processed_img = preprocess_input(processed_img)
        
        heatmaps, top_predictions = predict_and_generate_heatmaps(processed_img)
        predictions = sorted(top_predictions, key=lambda x: x['likelihoodScore'], reverse=True)

        bounding_boxes, superimposed_merged_img = process_heatmaps_and_generate_boxes(imageData, heatmaps, predictions) 

        superimposed_merged_img_base64 = encode_base64_image(superimposed_merged_img)

        return {
            "predictions": predictions,
            "boundingBoxes": bounding_boxes,
            "originalImage": encode_base64_image(img),
            "analyzedImage": superimposed_merged_img_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# To run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
