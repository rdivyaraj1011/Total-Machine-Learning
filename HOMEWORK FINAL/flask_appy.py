from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)


details = [
    {'name':'Jimit','address':'India'}
]

@app.route('/person', methods=['GET'])
def details_route():
    app.logger.info('API CALLED---------')
    return jsonify({'message': 'Details added', 'details': details}), 200


@app.route('/add_person', methods=['POST'])
def add_person():
    # Get the JSON payload
    data = request.get_json()
    app.logger.info('Data received: %s', data)

    # Validate the incoming data
    if not data or 'name' not in data or 'address' not in data:
        app.logger.error('Invalid input, please provide name and address.')
        return jsonify({'error': 'Invalid input, please provide name and address.'}), 400

    # Append the new person to the details list
    new_person = {
        'output_name': data['name'].lower(),
        'output_address': data['address'].lower()
    }

    # Return the updated list of details
    return jsonify(new_person), 200


@app.route('/numbers')
def print_list():
    return jsonify(list(range(5)))


def calculate_string_length(input_string):
    """Calculate the length of the input string."""
    return len(input_string)

@app.route('/string_length', methods=['POST'])
def string_length():
    # Get the JSON payload
    data = request.get_json()
    app.logger.info('Data received: %s', data)

    # Validate the incoming data
    if not data or 'input_string' not in data:
        app.logger.error('Invalid input, please provide input_string.')
        return jsonify({'error': 'Invalid input, please provide input_string.'}), 400
    
    app.logger.info("Data validated successfully")

    # Call the function to calculate the length of the input string
    length = calculate_string_length(data['input_string'])
    app.logger.info('Length received: %s', length)

    # Return the length as a JSON response
    return jsonify({'length': length}), 200


def perform_statistics(numbers):
    app.logger.info("Fucntion called for statistics------------")
    """Function to calculate mean, median, mode, and standard deviation."""
    # Convert input to a NumPy array for easier calculations
    numbers_array = np.array(numbers)
    app.logger.info('Numbers array received: %s', numbers_array)

    # Calculate statistics
    mean = np.mean(numbers_array)
    median = np.median(numbers_array)
    std_dev = np.std(numbers_array)

    # Create a result dictionary
    result = {
        'mean': mean,
        'median': median,
        'standard_deviation': std_dev
    }

    return result

@app.route('/statistics', methods=['POST'])
def calculate_statistics():
    # Get the JSON payload
    data = request.get_json()
    app.logger.info('Data received for statistics api: %s', data)
    # Validate the incoming data
    if not data or 'numbers' not in data:
        app.logger.warning('Invalid input, please provide a list of numbers.')
        return jsonify({'error': 'Invalid input, please provide a list of numbers.'}), 400

    numbers = data['numbers']
    app.logger.info('Numbers received: %s', numbers)

    # Call the function to calculate statistics
    stats_result = perform_statistics(numbers)

    # Return the statistics as a JSON response
    return jsonify(stats_result), 200


app.run(debug=True)


# app.logger.info('This is an INFO message for name: %s', name)
# app.logger.warning('This is a WARNING message')
# app.logger.error('This is an ERROR message')
# app.logger.info('This is an INFO message')