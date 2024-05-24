from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
import torch
from torchvision import transforms
import numpy as np
import json
from PIL import Image
import os

app = Flask(__name__)

# Loading the saved tokenizer and model
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

loaded_model = load_model("BiLSTM_INAPPRO_TEXT_CLASSIFIER.h5")



def classify_text(text):
    # Tokenize and pad sequences
    sequence = tokenizer.texts_to_sequences(text)
    padded_sequence = pad_sequences(sequence, maxlen=128)

    result = loaded_model.predict(padded_sequence)
    # Converting prediction to reflect the sentiment predicted.
    print(result, "----------------------------------")
    
    result = np.where(result>=0.5, 1, 0)

    return result

@app.route('/classify-text', methods=['POST', 'GET'])
def classify_text_view():
    if request.method == 'POST':
        try:
            text = request.form['text']
            print(text, "---------------------------")
            if text:
                result = classify_text([text])

                if result[0] == 0:
                    result = "Appropriate"
                else:
                    result = "Inappropriate"
            else:
                result = 'Please Provide valid input'
        except Exception as e:
            result = 'Something went wrong!'

        return jsonify({"response": result})
    else:
        return render_template("classify-text.html", data={})



# Use a pipeline as a high-level helper


model = pipeline("image-classification", model="Pratik-hf/Inappropriate-image-classification-using-ViT")

def classify_image(image):
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize the image to match the model's input size
    #     transforms.ToTensor(),           # Convert the image to a tensor
    #     transforms.Normalize(            # Normalize the image
    #         mean=[0.485, 0.456, 0.406],   # Mean and standard deviation values are taken from ImageNet normalization
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])
    # input_tensor = preprocess(image).unsqueeze(0)

    
    # Forward pass
    with torch.no_grad():
        outputs = model(image)

    # Get predicted class probabilities
    # Get the label with the highest probabilities
    prediction = max(outputs, key=lambda x: x['score'])

    if prediction['label'] == "LABEL_0":
        prediction = f"{round(prediction['score'], 2)} - Safe"
    else:
        prediction = f"{round(prediction['score'], 2)} - Unsafe"

    # Print predicted probabilities for each class
    print("Predicted probabilities:", prediction)

    return prediction


@app.route('/classify-image', methods=['POST', 'GET'])
def classify_image_view():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})

        image_file = request.files['image']

        image = Image.open(image_file)

        upload_dir = 'uploads'
        image_path = os.path.join(upload_dir, image_file.filename)
        image_file.save(image_path)

        # Classify the image
        prediction = classify_image(image)

        return jsonify({"response": prediction})

    else:
        return render_template("classify-image.html", data={})



@app.route('/')
def index_view():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
