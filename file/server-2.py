from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(_name_)
model = tf.keras.models.load_model('mango_disease_model.h5')  # Load trained model

# Function for Grad-CAM
def generate_gradcam_heatmap(img_array, model, last_conv_layer_name='convnext_xlarge_stage4_block5_1'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@app.route('/predict', methods=['POST'])
def predict():
    # Receive the uploaded file
    file = request.files['image']
    img = tf.keras.utils.load_img(file, target_size=(224, 224))  # Resize
    img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)[0]
    class_names = ['Anthracnose', 'Sooty Mould', 'Healthy']  # Update as per dataset classes
    result = {class_name: float(predictions[i]) for i, class_name in enumerate(class_names)}

    # Generate Grad-CAM
    heatmap = generate_gradcam_heatmap(img_array, model)
    plt.imshow(img_array[0])
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    heatmap_path = 'static/heatmap.jpg'
    plt.axis('off')
    plt.savefig(heatmap_path)
    plt.close()

    return jsonify({'predictions': result, 'heatmap': heatmap_path})

if _name_ == '_main_':
    app.run(debug=True)