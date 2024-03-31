from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
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



image_model = load_model('NSFW_Image_Classifier_inceptionv3.h5')

def classify_image(image):
    image = image.resize((299, 299))
    image = np.array(image) / 255.0 

    prediction = image_model.predict(np.expand_dims(image, axis=0))
    class_label = 'Appropriate' if (prediction[0][1] > prediction[0][0]) else 'Inappropriate'

    return class_label


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
