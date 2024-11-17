import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained Decision Tree model from the .pkl file
model_filename = 'decision_tree_model_iris.pkl'

# Load the model when the application starts
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()
    app.logger.info('Data received: %s', data)

    # Validate the data
    if not data:
        app.logger.error('No data received')
        return jsonify({'error': 'No data received'}), 400

    # Ensure the required keys are in the incoming JSON data
    required_keys = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    if not all(key in data for key in required_keys):
        app.logger.error('Missing required keys in the request data')
        return jsonify({'error': 'Missing required keys in the request data'}), 400

    # Convert the incoming JSON data into a DataFrame for prediction
    new_data = pd.DataFrame([data])
    app.logger.info('New data for prediction: %s', new_data)
    # Make predictions
    prediction = model.predict(new_data)
    app.logger.info('Prediction: %s', prediction)
    # Return the prediction as a JSON response
    return jsonify({'Predicted output species': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
