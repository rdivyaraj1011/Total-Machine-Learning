from flask import Flask, request, jsonify
import joblib
import numpy as np

with open('random_forest_model.joblib', 'rb') as model_file:
    model = joblib.load(model_file)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    
    # Extract the relevant features from the input
    tenure = data.get('tenure')
    internet_service = data.get('internet_service')
    contract = data.get('contract')
    monthly_charges = data.get('monthly_charges')
    total_charges = data.get('total_charges')

    # Map categorical features to numerical values 
    internet_service_mapping = {
        'No': 0,
        'DSL': 1,
        'Fiber optic': 2
    }
    
    contract_mapping = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }

    # Convert input data to the format expected by the model
    input_data = np.array([
        tenure,
        internet_service_mapping.get(internet_service, 0),
        contract_mapping.get(contract, 0),
        monthly_charges,
        total_charges
    ]).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(input_data)
    
    # Interpret the prediction
    if prediction[0] == 1:
        result = "This customer is likely to churn."
    else:
        result = "This customer is likely to stay."

    # Return the result as JSON
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
