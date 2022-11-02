import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='Template')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    return render_template('Index.html', prediction_text= 'Is this claim is a Fraud:{}'.format(output).replace('0','No').replace('1','Yes'))

@app.route('/View', methods=['POST', 'GET'])
def View():
    return render_template('view.html')


if __name__ =="__main__":
    app.run(debug=True)
