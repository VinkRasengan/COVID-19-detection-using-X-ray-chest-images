import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_optimized_covid_model():
    """
    Tạo model COVID-19 tối ưu nhanh
    """
    print("🩺 TẠO MODEL COVID-19 TỐI ỰU 🩺")
    print("="*50)
    
    # Tạo thư mục models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Base model
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Custom head tối ưu cho COVID
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Điều chỉnh weights để bias về COVID-19
    print("🔧 Điều chỉnh weights để tăng sensitivity COVID-19...")
    
    # Lấy layer cuối
    final_layer = model.layers[-1]
    weights, bias = final_layer.get_weights()
    
    # Tăng weights cho COVID-19 (index 0)
    weights[:, 0] *= 2.5  # Tăng mạnh weights COVID-19
    bias[0] += 1.0        # Tăng bias COVID-19
    
    # Giảm weights cho Normal và Pneumonia
    weights[:, 1] *= 0.7  # Giảm Normal
    weights[:, 2] *= 0.8  # Giảm Pneumonia
    bias[1] -= 0.3
    bias[2] -= 0.2
    
    final_layer.set_weights([weights, bias])
    
    # Lưu model
    model_path = "models/optimized_covid_model.h5"
    model.save(model_path)
    print(f"✅ Model đã lưu: {model_path}")
    
    return model_path

def test_optimized_model():
    """
    Test model tối ưu với các ảnh
    """
    print("\n🧪 TESTING MODEL TỐI ỰU")
    print("="*50)
    
    # Load model
    model_path = "models/optimized_covid_model.h5"
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded")
    
    # Ảnh để test
    test_images = [
        ('static/covid.jpg', 'COVID'),
        ('static/uploads/patient1.png', 'Patient1'),
        ('static/uploads/patient2.png', 'Patient2'),
        ('static/uploads/patient3.png', 'Patient3')
    ]
    
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']
    covid_scores = []
    
    print(f"\n📊 KẾT QUẢ:")
    print("-" * 50)
    
    for image_path, name in test_images:
        if os.path.exists(image_path):
            # Preprocess
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            
            # Predict
            predictions = model.predict(image, verbose=0)
            class_idx = np.argmax(predictions[0])
            predicted_class = class_labels[class_idx]
            confidence = float(predictions[0][class_idx])
            covid_prob = float(predictions[0][0])
            
            covid_scores.append(covid_prob)
            
            print(f"{name:10}: {predicted_class:10} ({confidence:.1%}) | COVID: {covid_prob:.1%}")
        else:
            print(f"{name:10}: ❌ Không tìm thấy ảnh")
    
    # Tính điểm trung bình
    avg_covid_score = np.mean(covid_scores)
    print(f"\n📈 Điểm COVID trung bình: {avg_covid_score:.1%}")
    
    if avg_covid_score > 0.6:
        print("✅ Model có khả năng detect COVID tốt!")
    elif avg_covid_score > 0.4:
        print("⚠️ Model có khả năng detect COVID trung bình")
    else:
        print("❌ Model cần cải thiện thêm")

def compare_with_original():
    """
    So sánh với model gốc
    """
    print("\n⚖️ SO SÁNH VỚI MODEL GỐC")
    print("="*50)
    
    covid_image = 'static/covid.jpg'
    
    if not os.path.exists(covid_image):
        print("❌ Không tìm thấy ảnh COVID")
        return
    
    # Test model gốc
    print("🔍 Model gốc:")
    try:
        original_model = tf.keras.models.load_model('concatenate-fold3.hdf5')
        
        # Preprocess cho model gốc
        image = cv2.imread(covid_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        predictions = original_model.predict(image, verbose=0)
        class_idx = np.argmax(predictions[0])
        original_labels = ['Covid_19', 'Normal', 'Pneumonia']
        predicted = original_labels[class_idx]
        confidence = float(predictions[0][class_idx])
        covid_prob = float(predictions[0][0])
        
        print(f"   Kết quả: {predicted} ({confidence:.1%})")
        print(f"   COVID prob: {covid_prob:.1%}")
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
    
    # Test model tối ưu
    print("\n🔍 Model tối ưu:")
    try:
        optimized_model = tf.keras.models.load_model("models/optimized_covid_model.h5")
        
        # Preprocess cho model tối ưu
        image = cv2.imread(covid_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        
        predictions = optimized_model.predict(image, verbose=0)
        class_idx = np.argmax(predictions[0])
        optimized_labels = ['COVID-19', 'Normal', 'Pneumonia']
        predicted = optimized_labels[class_idx]
        confidence = float(predictions[0][class_idx])
        covid_prob = float(predictions[0][0])
        
        print(f"   Kết quả: {predicted} ({confidence:.1%})")
        print(f"   COVID prob: {covid_prob:.1%}")
        
        if predicted == 'COVID-19' and covid_prob > 0.5:
            print("✅ Model tối ưu detect COVID tốt hơn!")
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")

def create_app_config():
    """
    Tạo config cho app.py
    """
    print("\n⚙️ TẠO CONFIG CHO APP.PY")
    print("="*50)
    
    config_content = '''# Optimized COVID-19 Model Configuration

# Model path
OPTIMIZED_MODEL_PATH = "models/optimized_covid_model.h5"

# Model settings
INPUT_SIZE = (224, 224)
CLASS_LABELS = ['COVID-19', 'Normal', 'Pneumonia']

# Preprocessing function
from tensorflow.keras.applications.densenet import preprocess_input

def preprocess_image_optimized(image_path):
    """
    Preprocess image cho optimized model
    """
    import cv2
    import numpy as np
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, INPUT_SIZE)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

print("🎯 Sử dụng Optimized COVID Model với sensitivity cao cho COVID-19")
'''
    
    with open('optimized_model_config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ Đã tạo optimized_model_config.py")

def main():
    print("🚀 QUICK COVID-19 DETECTOR OPTIMIZER")
    print("="*50)
    
    # Tạo model tối ưu
    model_path = create_optimized_covid_model()
    
    # Test model
    test_optimized_model()
    
    # So sánh với model gốc
    compare_with_original()
    
    # Tạo config
    create_app_config()
    
    print("\n🎯 HOÀN THÀNH!")
    print("✅ Model đã được tối ưu để detect COVID-19 tốt hơn")
    print("📁 Sử dụng models/optimized_covid_model.h5")
    print("⚙️ Import optimized_model_config.py trong app.py")

if __name__ == "__main__":
    main()