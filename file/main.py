import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
import io

img_height = 224
img_width = 224

model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8)) 
model.layers[0].trainable = False
model.compile(Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.load_weights('mango_leaf_classification_model_weights_omdena_resnet50.hdf5')

classes = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

def process_heatmaps_and_generate_boxes(imgs, heatmaps, top_predictions):
    img = np.array(imgs)
    img = cv2.resize(img, (img_width, img_height))
    merged_heatmap = np.zeros_like(img, dtype=np.float32)
    bounding_boxes = []

    for idx, prediction in enumerate(top_predictions):
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

def predict_and_generate_heatmaps(model, img_array):
    logits = model.predict(img_array)
    probabilities = tf.nn.softmax(logits).numpy()

    img_tensor = tf.convert_to_tensor(img_array)

    # Get base model and create grad model
    base_model = model.layers[0]
    # Using an even earlier layer for more detailed features
    layer_name = 'conv3_block4_3_conv'  # Changed to earlier layer for finer details
    grad_model = tf.keras.models.Model([base_model.inputs], [base_model.get_layer(layer_name).output, base_model.output])

    heatmaps = []
    top_predictions = []

    # Lower threshold further to capture more subtle features
    significant_indices = [i for i, prob in enumerate(probabilities[0]) if round(prob * 100, 2) > 0]

    for class_idx in significant_indices:
        with tf.GradientTape() as tape:
            conv_output, pred = grad_model(img_tensor)
            loss = pred[:, class_idx]
        grads = tape.gradient(loss, conv_output)
        
        # Enhanced pooling method
        pooled_grads = tf.reduce_mean(tf.abs(grads), axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        heatmap = cam.numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Use smaller epsilon for better detail preservation
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
        
        # Apply smaller kernel for sharper details
        heatmap = cv2.GaussianBlur(heatmap, (3,3), 0.5)
        
        # Enhance contrast
        heatmap = np.power(heatmap, 0.7)  # Adjust gamma for better visualization
            
        heatmaps.append(heatmap)
        top_predictions.append({"diseaseName":classes[class_idx], "likelihoodScore":probabilities[0][class_idx]})   

    return heatmaps, top_predictions

def encode_image_to_base64(image_path):
    # Open the image file
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    
    # Encode image to base64
    base64_string = base64.b64encode(img_data).decode('utf-8')
    
    return base64_string
def decode_base64_image(base64_string):
    # Decode the base64 string to image data
    img_data = base64.b64decode(base64_string)
    
    # Open the image from the byte data
    img = Image.open(io.BytesIO(img_data))
    return img

upload = 'test images/images (3).jpg'
encoded_image = encode_image_to_base64(upload)
decoded_image = decode_base64_image(encoded_image)

img = decoded_image.resize((img_height, img_width)) 
img = img.convert("RGB") 
img_array = image.img_to_array(img) 
img_array = np.expand_dims(img_array, axis=0) 
img_array = preprocess_input(img_array) 
# Call the function
heatmaps, top_predictions = predict_and_generate_heatmaps(model, img_array)

print("Top predicted diseases:")
print(top_predictions)

bounding_boxes, superimposed_merged_img = process_heatmaps_and_generate_boxes(decoded_image, heatmaps, top_predictions)

with open('bounding_boxes.json', 'w') as f:
    json.dump(bounding_boxes, f)

print(bounding_boxes)



plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(superimposed_merged_img, cv2.COLOR_BGR2RGB))
plt.title("Merged Heatmap with Strong Highlights")
plt.axis('off')
plt.show()




