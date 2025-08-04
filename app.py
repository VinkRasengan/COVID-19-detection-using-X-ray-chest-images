import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'], supports_credentials=True)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the improved model with priority order
available_models = {}
current_model = None
current_model_info = {}

# Model priority order (best to worst)
model_candidates = [
    {
        'name': 'DenseNet201 Model',
        'path': 'models/densenet201_covid_model.h5', 
        'input_size': (224, 224),
        'description': 'DenseNet201 fine-tuned cho COVID detection'
    },
    {
        'name': 'Optimized Model',
        'path': 'models/optimized_covid_model.h5',
        'input_size': (224, 224),
        'description': 'Model tối ưu hóa'
    },
    {
        'name': 'Simple COVID Model',
        'path': 'models/simple_covid_model.h5',
        'input_size': (224, 224),
        'description': 'Simple COVID detection model'
    }
]

print("🩺 COVID-19 X-RAY DETECTION SYSTEM 🩺")
print("="*50)

# Load all available models
for candidate in model_candidates:
    try:
        if os.path.exists(candidate['path']):
            print(f"📥 Thử load {candidate['name']}...")
            model = tf.keras.models.load_model(candidate['path'])
            available_models[candidate['name']] = {
                'model': model,
                'info': candidate
            }
            print(f"✅ Đã load thành công: {candidate['name']}")
            
            # Set first available model as current
            if current_model is None:
                current_model = model
                current_model_info = candidate
                print(f"🎯 Model mặc định: {candidate['name']}")
        else:
            print(f"⚠️ {candidate['name']} không tồn tại")
    except Exception as e:
        print(f"❌ Lỗi load {candidate['name']}: {str(e)}")
        continue

if not available_models:
    print("❌ Không thể load bất kỳ model nào!")
    exit(1)

print(f"\n📊 Tổng cộng đã load {len(available_models)} models")
print(f"🔧 Models có sẵn: {list(available_models.keys())}")

# Define class labels
CLASS_LABELS = ['COVID-19', 'Normal', 'Pneumonia']

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define the folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image with adaptive input size
def preprocess_image(image_path, model_info):
    """
    Tiền xử lý ảnh với kích thước input phù hợp với model
    """
    try:
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        # Chuyển đổi BGR sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize theo input size của model hiện tại
        input_size = model_info.get('input_size', (224, 224))
        image = cv2.resize(image, input_size)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        app.logger.error(f"Lỗi preprocessing ảnh: {str(e)}")
        return None

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for multi-model interface
@app.route('/multi')
def multi_model():
    return render_template('multi_model.html')

# Route to get available models
@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models_list = []
    for model_name, model_data in available_models.items():
        models_list.append({
            'name': model_name,
            'description': model_data['info']['description'],
            'input_size': model_data['info']['input_size'],
            'is_current': model_name == current_model_info.get('name')
        })
    
    return jsonify({
        'success': True,
        'models': models_list,
        'current_model': current_model_info.get('name', 'Unknown'),
        'total_models': len(available_models)
    })

# Route to switch model
@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    global current_model, current_model_info
    
    data = request.get_json()
    if not data or 'model_name' not in data:
        return jsonify({'error': 'Model name required'}), 400
    
    model_name = data['model_name']
    
    if model_name not in available_models:
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    # Switch model
    current_model = available_models[model_name]['model']
    current_model_info = available_models[model_name]['info']
    
    app.logger.info(f"Switched to model: {model_name}")
    
    return jsonify({
        'success': True,
        'message': f'Switched to {model_name}',
        'current_model': {
            'name': current_model_info['name'],
            'description': current_model_info['description'],
            'input_size': current_model_info['input_size']
        }
    })

# Route to predict with specific model
@app.route('/predict_with_model', methods=['POST'])
def predict_with_model():
    """Predict using a specific model"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    model_name = request.form.get('model_name', current_model_info.get('name'))
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    if model_name not in available_models:
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get model and info
        selected_model = available_models[model_name]['model']
        selected_model_info = available_models[model_name]['info']
        
        # Preprocess image
        image = preprocess_image(filepath, selected_model_info)
        if image is None:
            return jsonify({'error': 'Lỗi xử lý ảnh'}), 500
        
        # Predict
        prediction = selected_model.predict(image, verbose=0)
        class_idx = np.argmax(prediction)
        predicted_label = CLASS_LABELS[class_idx]
        confidence = float(prediction[0][class_idx])
        
        # Get probabilities
        prediction_probabilities = {
            CLASS_LABELS[i]: round(float(prediction[0][i]), 4) 
            for i in range(len(CLASS_LABELS))
        }
        
        # Confidence level
        confidence_level = "Cao" if confidence > 0.7 else "Trung bình" if confidence > 0.5 else "Thấp"
        
        return jsonify({
            'success': True,
            'label': predicted_label,
            'confidence': round(confidence, 4),
            'confidence_level': confidence_level,
            'probabilities': prediction_probabilities,
            'filename': filename,
            'model_info': {
                'name': selected_model_info['name'],
                'description': selected_model_info['description'],
                'input_size': selected_model_info['input_size']
            },
            'recommendation': {
                'reliable': confidence > 0.7,
                'message': 'Kết quả tin cậy' if confidence > 0.7 else 'Nên kiểm tra thêm' if confidence > 0.5 else 'Kết quả không chắc chắn'
            }
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# Route to compare all models
@app.route('/compare_models', methods=['POST'])
def compare_models():
    """Compare predictions from all available models"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    results = {}
    
    # Test with all available models
    for model_name, model_data in available_models.items():
        try:
            model = model_data['model']
            model_info = model_data['info']
            
            # Preprocess image
            image = preprocess_image(filepath, model_info)
            if image is None:
                continue
            
            # Predict
            prediction = model.predict(image, verbose=0)
            class_idx = np.argmax(prediction)
            predicted_label = CLASS_LABELS[class_idx]
            confidence = float(prediction[0][class_idx])
            
            # Get probabilities
            prediction_probabilities = {
                CLASS_LABELS[i]: round(float(prediction[0][i]), 4) 
                for i in range(len(CLASS_LABELS))
            }
            
            # Confidence level
            confidence_level = "Cao" if confidence > 0.7 else "Trung bình" if confidence > 0.5 else "Thấp"
            
            results[model_name] = {
                'label': predicted_label,
                'confidence': round(confidence, 4),
                'confidence_level': confidence_level,
                'probabilities': prediction_probabilities,
                'model_info': {
                    'name': model_info['name'],
                    'description': model_info['description'],
                    'input_size': model_info['input_size']
                }
            }
            
        except Exception as e:
            app.logger.error(f"Error with model {model_name}: {str(e)}")
            results[model_name] = {
                'error': str(e)
            }
    
    return jsonify({
        'success': True,
        'filename': filename,
        'results': results,
        'total_models': len(results)
    })

# Route to handle image uploads and predictions (run all models)
@app.route('/predict', methods=['POST'])
def predict():
    app.logger.debug(f"Received request: {request.method}")
    app.logger.debug(f"Request files: {request.files}")
    
    if 'file' not in request.files:
        app.logger.error("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    app.logger.debug(f"File name: {file.filename}")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.debug(f"File saved to: {filepath}")

        try:
            results = {}
            
            # Run prediction with all available models
            for model_name, model_data in available_models.items():
                try:
                    model = model_data['model']
                    model_info = model_data['info']
                    
                    # Preprocess image
                    image = preprocess_image(filepath, model_info)
                    if image is None:
                        continue
                    
                    # Predict
                    prediction = model.predict(image, verbose=0)
                    class_idx = np.argmax(prediction)
                    predicted_label = CLASS_LABELS[class_idx]
                    confidence = float(prediction[0][class_idx])
                    
                    # Get probabilities
                    prediction_probabilities = {
                        CLASS_LABELS[i]: round(float(prediction[0][i]), 4) 
                        for i in range(len(CLASS_LABELS))
                    }
                    
                    # Confidence level
                    confidence_level = "Cao" if confidence > 0.7 else "Trung bình" if confidence > 0.5 else "Thấp"
                    
                    results[model_name] = {
                        'label': predicted_label,
                        'confidence': round(confidence, 4),
                        'confidence_level': confidence_level,
                        'probabilities': prediction_probabilities,
                        'model_info': {
                            'name': model_info['name'],
                            'description': model_info['description'],
                            'input_size': model_info['input_size']
                        },
                        'recommendation': {
                            'reliable': confidence > 0.7,
                            'message': 'Kết quả tin cậy' if confidence > 0.7 else 'Nên kiểm tra thêm' if confidence > 0.5 else 'Kết quả không chắc chắn'
                        }
                    }
                    
                    app.logger.debug(f"Prediction successful for {model_name}: {predicted_label} ({confidence:.3f})")
                    
                except Exception as e:
                    app.logger.error(f"Error with model {model_name}: {str(e)}")
                    results[model_name] = {
                        'error': str(e)
                    }
            
            # Get the best model result (highest confidence)
            best_model = None
            best_confidence = 0
            for model_name, result in results.items():
                if 'error' not in result and result['confidence'] > best_confidence:
                    best_confidence = result['confidence']
                    best_model = model_name
            
            return jsonify({
                'success': True,
                'filename': filename,
                'results': results,
                'best_model': best_model,
                'total_models': len(results),
                'summary': {
                    'best_result': results[best_model] if best_model else None,
                    'models_used': list(results.keys())
                }
            })
            
        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500

    app.logger.error(f"Invalid file type: {file.filename}")
    return jsonify({'error': 'Invalid file type'}), 400

# Route to display the uploaded image
@app.route('/uploads/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
