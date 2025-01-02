import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import io
import base64
import cv2

# Initialize Flask app
app = Flask(__name__)

# Image dimensions for ResNet50 input
img_height = 224
img_width = 224

# Build the model
model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8))  # 8 output classes (based on your class labels)
model.layers[0].trainable = False
model.compile(Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Load the model weights
model.load_weights('/model/mango_leaf_classification_model_weights_omdena_resnet50.hdf5')

base_model = model.layers[0]
layer_name = 'conv5_block3_3_conv'
# layer_name = 'conv5_block3_out'
grad_model = tf.keras.models.Model([base_model.inputs], [base_model.get_layer(layer_name).output, base_model.output])

# Define the class labels (diseases)
classes = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

def process_heatmaps_and_generate_boxes(img, heatmaps, predictions):
    img = cv2.resize(img, (img_width, img_height))

    merged_heatmap = np.zeros_like(img, dtype=np.float32)
    bounding_boxes = []

    for idx, prediction in enumerate(predictions):
        heatmap = heatmaps[idx][0]
        heatmap = cv2.resize(heatmap, (img_width, img_height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        merged_heatmap += heatmap.astype(np.float32)
        
        # Find the disease area
        heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(heatmap_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            bounding_boxes.append({
                "diseaseName": prediction["diseaseName"],
                "x": int(x),
                "y": int(y),
                "h": int(w),
                "w": int(h),
            })
    
    merged_heatmap = np.clip(merged_heatmap, 0, 255).astype(np.uint8)
    superimposed_merged_img = cv2.addWeighted(img, 0.3, merged_heatmap, 0.7, 0)
    
    return bounding_boxes, superimposed_merged_img

def predict_and_generate_heatmaps(img_array):
    logits = model.predict(img_array)
    probabilities = tf.nn.softmax(logits).numpy()

    img_tensor = tf.convert_to_tensor(img_array)

    heatmaps = []
    top_predictions = []

    significant_indices = [i for i, prob in enumerate(probabilities[0]) if round(prob * 100, 2) > 0]

    for class_idx in significant_indices:
        with tf.GradientTape() as tape:
            conv_output, pred = grad_model(img_tensor)
            loss = pred[:, class_idx]
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
        conv_output_val = conv_output.numpy()
        pooled_grads_val = pooled_grads.numpy()
            
        for i in range(pooled_grads_val.shape[0]):
            if i < conv_output_val.shape[-1]:
                conv_output_val[0, :, :, i] *= pooled_grads_val[i]
        
        heatmap = np.mean(conv_output_val, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val != 0:
            heatmap /= max_val
            
        heatmaps.append(heatmap)
        top_predictions.append({"diseaseName": classes[class_idx], "likelihoodScore": round(probabilities[0][class_idx] * 100, 2)})   

    return heatmaps, top_predictions

def decode_base64_image(base64_string):   
    if ',' in base64_string:
        base64_string = base64_string.split(",")[1]  # Remove metadata if present
    img_data = base64.b64decode(base64_string)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    return img

def encode_base64_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        imageData = decode_base64_image(data['image'])

        img = cv2.resize(imageData, (img_height, img_width))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = np.expand_dims(imgRGB, axis=0)
        processed_img = preprocess_input(processed_img)
        
        heatmaps, top_predictions = predict_and_generate_heatmaps(processed_img)
        predictions = sorted(top_predictions, key=lambda x: x['likelihoodScore'], reverse=True)

        bounding_boxes, superimposed_merged_img = process_heatmaps_and_generate_boxes(imageData, heatmaps, predictions) 

        superimposed_merged_img_base64 = encode_base64_image(superimposed_merged_img)

        return jsonify({ 
            "predictions": predictions,
            "boundingBoxes": bounding_boxes,
            "originalImage": encode_base64_image(img),  # Now returns the RGB resized image
            "analyzedImage": superimposed_merged_img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
