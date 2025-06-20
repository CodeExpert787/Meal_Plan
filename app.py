from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sample import load_and_train_model, predict_pcos_type

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model when the application starts
# print("Loading model...")
# model, scaler, feature_names, column_types = load_and_train_model("PCOS profilling.xlsx")

# Define PCOS types mapping based on actual data
PCOS_TYPES = {
    0: 'PCOS Adrenal',
    1: 'PCOS Keradangan/Inflammation',
    2: 'PCOS Keradangan/Infllammation',
    3: 'PCOS Pil Perancang/Post Birth Control',
    4: 'PCOS Pos Pil Perancang/Post Birth Control',
    5: 'PCOS Rintangan Insulin/Insulin Resistance'
}
MEAL_TYPES = {
    0: 'Build muscle',
    1: 'Improve blood sugar regulation',
    2: 'Improve fertility',
    3: 'Lose weight'
}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/home')
def home():
    return render_template('./pcos/index.html', features=feature_names)

@app.route('/')
def upload_page():
    return render_template('./pcos/upload.html')

@app.route('/meal')
def meal_upload_page():
    return render_template('./meal/upload.html')

@app.route('/meal_home')
def meal_home():
    return render_template('./meal/index.html', features=feature_names)


@app.route('/', methods=['POST'])
def upload_file():
    print("Upload request received")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    print(f"Received file: {file.filename}")
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Attempting to save file to: {filepath}")
        try:
            file.save(filepath)
            print("File saved successfully")
            
            # Reload the model with the new data
            global model, scaler, feature_names, column_types
            # Reset globals to None (optional, for clarity)
            model = scaler = feature_names = column_types = None
            print("Loading and training model...")
            model, scaler, feature_names, column_types = load_and_train_model(filepath)
            
            # Format/validate globals
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            if model is None or scaler is None or not feature_names:
                raise ValueError("Model, scaler, or feature_names not properly loaded.")
            
            print("Model updated successfully")
            return jsonify({'message': 'File uploaded and model updated successfully'})
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    print("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None:
                return jsonify({'error': f'Missing value for {feature}'}), 400
            input_data[feature] = value
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        # Make prediction
        prediction, type_probabilities = predict_pcos_type(input_df, model, scaler)
        
        # Get the PCOS type name
        pcos_type = PCOS_TYPES.get(prediction, 'Unknown Type')
        print(f"prediction: {pcos_type}, type_probabilities:{type_probabilities }")
        return jsonify({
            'pcos_type': str(pcos_type),
            'type_probabilities': type_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    error = request.args.get('error')
    name = request.args.get('name')
    height = request.args.get('height')
    weight = request.args.get('weight')
    
    # Parse the probabilities from URL
    import json
    probabilities_str = request.args.get('probabilities')
    type_probabilities = {}
    if probabilities_str:
        try:
            type_probabilities = json.loads(probabilities_str)
        except:
            type_probabilities = {}
    
    return render_template('./pcos/result.html', 
                         prediction=prediction, 
                         error=error,
                         name=name,
                         height=height,
                         weight=weight,
                         type_probabilities=type_probabilities)


@app.route('/meal_predict', methods=['POST'])
def meal_predict():
    try:
        # Get input data from the form
        input_data = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None:
                return jsonify({'error': f'Missing value for {feature}'}), 400
            input_data[feature] = value
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        # Make prediction
        prediction, type_probabilities = predict_pcos_type(input_df, model, scaler)
        
        # Get the PCOS type name
        pcos_type = MEAL_TYPES.get(prediction, 'Unknown Type')
        print(f"prediction: {pcos_type}, type_probabilities:{type_probabilities }")
        return jsonify({
            'pcos_type': str(pcos_type),
            'type_probabilities': type_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/meal_result')
def meal_result():
    prediction = request.args.get('prediction')
    error = request.args.get('error')
    name = request.args.get('name')
    height = request.args.get('height')
    weight = request.args.get('weight')
    age = request.args.get('age')
    # Parse the probabilities from URL
    import json
    probabilities_str = request.args.get('probabilities')
    type_probabilities = {}
    if probabilities_str:
        try:
            type_probabilities = json.loads(probabilities_str)
        except:
            type_probabilities = {}
    
    return render_template('./meal/result.html', 
                         prediction=prediction, 
                         error=error,
                         name=name,
                         height=height,
                         age=age,
                         weight=weight,
                         type_probabilities=type_probabilities)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True) 