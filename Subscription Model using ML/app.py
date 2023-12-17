import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_customer_subscription', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('index.html')

@app.route('/about_page')
def about_page():
    return render_template('about.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = "User may not get the Subscription" if prediction[0] == 0 else "User may get the Subscription"

    return render_template('index.html', prediction_text=output)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)