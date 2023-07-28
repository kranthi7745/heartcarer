from flask import Flask, flash, request, redirect, url_for, render_template
import pickle
import numpy as np

# Loading Models
heart_model = pickle.load(open('models/heart_model.pkl', "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')
           
@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        gender = request.form['gender']
        ca = request.form['nmv']
        cp = request.form['tcp']
        exang = request.form['eia']
        thal = request.form['thal']
        oldpeak = request.form['op']
        thalach = request.form['mhra']
        age = request.form['age']
        pred = heart_model.predict(np.array([ca, cp, exang, thal, oldpeak, thalach, age]).reshape(1, -1))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

if __name__ == '__main__':
    app.run(debug=True)
