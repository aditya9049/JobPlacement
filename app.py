from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)


with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  
    ssc_percentage = float(request.form['ssc_percentage'])
    hsc_percentage = float(request.form['hsc_percentage'])
    degree_percentage = float(request.form['degree_percentage'])
    mba_percent = float(request.form['mba_percent'])
    
   
    input_features = np.array([[ssc_percentage, hsc_percentage, degree_percentage, mba_percent]])
    prediction = model.predict(input_features)
    
    result = 'Placed' if prediction[0] == 1 else 'Not Placed'
    
    return render_template('index.html', prediction_text=f'The student will be: {result}')

if __name__ == '__main__':
    app.run(debug=True)
 