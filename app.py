from flask import Flask, render_template, request, redirect, flash
import numpy as np
import pickle
import os

# Create the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Load the Gradient Boosting model
model_filename = 'gradient_boosting_model.pkl'
if os.path.isfile(model_filename):
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

@app.route("/")
def main():
    return render_template('main.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    prediction_text = None  # Initialize prediction_text variable
    if request.method == 'POST':
        try:
            # Fetch data from form inputs
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            # Prepare the data for prediction
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

            # Make the prediction
            prediction = loaded_model.predict(features)
            prediction_text = 'No Heart Disease' if prediction[0] == 0 else 'Heart Disease Detected'
            
            # Flash the result
            flash(prediction_text, 'danger' if prediction[0] == 0 else 'success')

        except Exception as e:
            flash('Error in input data. Please check your values.', 'danger')

    return render_template('home.html', prediction_text=prediction_text)  # Pass the prediction_text

if __name__ == '__main__':
    app.run(debug=True)
