from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('crop_yield_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = {
            'crop': request.form['crop'],
            'precipitation': request.form['precipitation'],
            'specific_humidity': request.form['specific_humidity'],
            'relative_humidity': request.form['relative_humidity'],
            'temperature': request.form['temperature']
        }
        
        crop = form_data['crop']
        precipitation = float(form_data['precipitation'])
        specific_humidity = float(form_data['specific_humidity'])
        relative_humidity = float(form_data['relative_humidity'])
        temperature = float(form_data['temperature'])
        
        # Convert crop to numeric
        crop_mapping = {'Cocoa, beans': 0, 'Oil palm fruit': 1, 'Rice, paddy': 2, 'Rubber, natural': 3}
        crop = crop_mapping[crop]

        # Prepare features array
        features = np.array([[crop, precipitation, specific_humidity, relative_humidity, temperature]])
        
        # Predict
        prediction = model.predict(features)
        
        return render_template('index.html', form_data=form_data, prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
