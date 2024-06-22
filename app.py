import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load each model and its labels
models = {
    'apple': (load_model(os.path.join(MODELS_DIR, 'Apple.h5')), {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy'}),
    'corn': (load_model(os.path.join(MODELS_DIR, 'Corn.h5')), {0: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 1: 'Corn___Common_rust', 2: 'Corn___healthy', 3: 'Corn___Northern_Leaf_Blight'}),
    'grapes': (load_model(os.path.join(MODELS_DIR, 'Grapes.h5')), {0: 'Grapes___Black_rot', 1: 'Grapes___Esca_(Black_Measles)', 2: 'Grapes___healthy', 3: 'Grapes___Leaf_blight_(Isariopsis_Leaf_Spot)'}),
    'blueberry_cherry': (load_model(os.path.join(MODELS_DIR, 'Blueberry_Cherry.h5')), {0: 'Blueberry___healthy', 1: 'Cherry_(including_sour)___Powdery_mildew'}),
    'peach': (load_model(os.path.join(MODELS_DIR, 'Peach.h5')), {0: 'Peach___Bacterial_spot', 1: 'Peach___healthy'}),
    'orange_raspberry': (load_model(os.path.join(MODELS_DIR, 'Orange_Raspberry.h5')), {0: 'Orange___Haunglongbing_(Citrus_greening)', 1: 'Raspberry___healthy'}),
    'pepper': (load_model(os.path.join(MODELS_DIR, 'Pepper.h5')), {0: 'Pepper_bell_Bacterial_spot', 1: 'Pepper_bell_healthy'}),
    'potato': (load_model(os.path.join(MODELS_DIR, 'Potato.h5')), {0: 'Potato___Early_blight', 1: 'Potato___healthy', 2: 'Potato___Late_blight'}),
    'soyabean_squash': (load_model(os.path.join(MODELS_DIR, 'Soyabean_Squash.h5')), {0: 'Soybean___healthy', 1: 'Squash___Powdery_mildew'}),
    'strawberry': (load_model(os.path.join(MODELS_DIR, 'Strawberry.h5')), {0: 'Strawberry___healthy', 1: 'Strawberry___Leaf_scorch'}),
    'tomato': (load_model(os.path.join(MODELS_DIR, 'Tomato.h5')), {0: 'Tomato_Bacterial_spot', 1: 'Tomato_Early_blight', 2: 'Tomato_Late_blight', 3: 'Tomato_Leaf_Mold', 4: 'Tomato_Septoria_leaf_spot', 5: 'Tomato_Spider_mites_Two-spotted_spider_mite', 6:'Tomato_Target_Spot', 7: 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 8: 'Tomato_Tomato_mosaic_virus', 9: 'Tomato_healthy'})
}

def get_result(image_path, model, labels):
    img = load_img(image_path, target_size=(224,224))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return labels[np.argmax(predictions)]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        selected_model = request.form['model']
        if selected_model not in models:
            return "Model not found"
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        model, labels = models[selected_model]
        result = get_result(file_path, model, labels)
        return f"{selected_model.capitalize()}: {result}"
        
    return "Invalid request"

if __name__ == '__main__':
    app.run(debug=True)
