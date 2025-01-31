import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

#import the model and the scaler
model=pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('score_svc.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    flt_features = [float(x) for x in request.form.values()]
    final_features = [np.array(flt_features)]
    
    prediction = model.predict(final_features)
    
    print(prediction)
    
    output = prediction[0]
    
    return render_template('index.html', prediction_text=f'Estimated wine type: {output}')
    

@app.route('/results', methods=['POST'])
def results():
    features=request.get_json(force=True)
    final_features = [np.array(list(features.values()))]
    final_features = scaler.transform(final_features)
  
    prediction = model.predict(final_features)
    
    output =  round(prediction[0],2)
    
    return jsonify(output)
    

if __name__ == "__main__":
    app.run(debug=True)
