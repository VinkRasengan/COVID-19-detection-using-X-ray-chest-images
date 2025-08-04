# 🩺 COVID-19 X-Ray Detection - Multi-Model System

> **Hệ thống phát hiện COVID-19 từ ảnh X-ray ngực sử dụng nhiều model AI với giao diện web hiện đại**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.1+-purple.svg)](https://getbootstrap.com)

## 🎯 Tổng quan

Dự án này cung cấp một hệ thống AI tiên tiến để phát hiện COVID-19 từ ảnh X-ray ngực, sử dụng **nhiều model deep learning** với khả năng:

- ✅ **Phát hiện chính xác** COVID-19, Normal, Pneumonia
- ✅ **So sánh nhiều model AI** cùng lúc  
- ✅ **Giao diện web hiện đại** với Bootstrap 5
- ✅ **Chuyển đổi model real-time** không cần reload
- ✅ **API REST** đầy đủ cho integration
- ✅ **Drag & drop upload** thân thiện

## 🧠 Models Available

| Model | Architecture | Accuracy | Speed | Recommended |
|-------|-------------|----------|-------|-------------|
| **DenseNet201** ⭐ | DenseNet201 fine-tuned | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Production |
| **Optimized** | DenseNet201 simplified | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Quick scan |
| **Simple COVID** | ResNet50V2 backbone | ⭐⭐⭐ | ⭐⭐⭐⭐ | Basic use |
| **Original** | ResNet+Xception concat | ⭐⭐ | ⭐⭐ | Baseline |

### Model Details:
- **Input size**: 224x224 (optimized) hoặc 300x300 (original)
- **Classes**: COVID-19, Normal, Pneumonia  
- **Confidence scoring**: Cao (>70%), Trung bình (50-70%), Thấp (<50%)
- **Fine-tuning**: Specialized cho COVID detection

## 🚀 Quick Start

### 1. Cài đặt Dependencies
```bash
# Clone repository
git clone <your-repo-url>
cd covid19-xray-detection

# Install requirements
pip install tensorflow opencv-python flask flask-cors pillow numpy matplotlib seaborn scikit-learn requests tqdm
```

### 2. Download Models

**⚠️ Quan trọng**: Các model AI không được include trong repository để giảm kích thước. Bạn cần download chúng trước khi sử dụng.

#### Option A: Download từ Google Drive (Recommended)
```bash
# Tạo thư mục models
mkdir models

# Download models từ Google Drive
# Link: https://drive.google.com/drive/folders/1FKdIL4aq6Uy0J8baT9D_zlkTeGjSyHrq?usp=sharing
# Hoặc sử dụng script download tự động:
python download_models.py
```

#### Option B: Tạo Models từ đầu
```bash
# Tạo models cải tiến (mất thời gian)
python simple_model_creator.py

# Test models
python quick_test.py
```

#### Models cần thiết:
- `models/densenet201_covid_model.h5` (76.2MB) - Model chính xác nhất
- `models/optimized_covid_model.h5` (73.9MB) - Model tối ưu
- `models/simple_covid_model.h5` (76.2MB) - Model đơn giản

#### Troubleshooting:
- **Lỗi download**: Kiểm tra kết nối internet và thử lại
- **File không đầy đủ**: Xóa file và download lại
- **Không có models**: Chạy `python download_models.py` hoặc tạo từ đầu

### 3. Chạy Application

#### Backend (Flask API)
```bash
# Kích hoạt virtual environment (nếu có)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Chạy Flask backend
python app.py
```

#### Frontend (React App)
```bash
# Chuyển vào thư mục frontend
cd frontend

# Cài đặt dependencies (chỉ lần đầu)
npm install

# Chạy React development server
npm start
```

#### Truy cập ứng dụng:
- **Frontend (React)**: http://localhost:3000 ⭐
- **Backend API**: http://localhost:5000
- **Multi-Model Interface**: http://localhost:5000/multi
- **Basic Interface**: http://localhost:5000

## 🌐 Web Interfaces

### 🎯 React Frontend (`http://localhost:3000`) - **RECOMMENDED**

**Giao diện chính với full features:**
- 🧠 **Model Selector**: Chọn và switch giữa các AI models
- 📊 **Model Comparison**: So sánh kết quả từ tất cả models
- 🎨 **Modern UI**: React với Material-UI components
- 📱 **Mobile-friendly**: Tương thích mọi thiết bị
- 🖱️ **Drag & Drop**: Upload ảnh dễ dàng
- 📈 **Visual Analytics**: Charts, progress bars, confidence indicators
- ⚡ **Real-time**: Live updates và smooth interactions

**Cách sử dụng:**
1. **Chọn Model**: Click vào model button để switch
2. **Upload Image**: Drag & drop hoặc click để chọn file X-ray
3. **Analyze**: 
   - "Phân tích với Model hiện tại" - Single prediction
   - "So sánh tất cả Models" - Multi-model comparison
4. **View Results**: Detailed analysis với confidence scores

### 📝 Flask Templates (`http://localhost:5000`)
- **Multi-Model Interface** (`/multi`): Bootstrap 5 interface
- **Basic Interface** (`/`): Simple upload form
- **API Testing**: JSON response format

## 📡 API Endpoints

### Model Management
```bash
# Get available models
GET /models
Response: {
  "success": true,
  "models": [...],
  "current_model": "DenseNet201 Model",
  "total_models": 4
}

# Switch current model
POST /switch_model
Body: {"model_name": "DenseNet201 Model"}
Response: {"success": true, "message": "Switched to..."}
```

### Predictions
```bash
# Predict with current model
POST /predict
Body: FormData with 'file'

# Predict with specific model
POST /predict_with_model
Body: FormData with 'file' + 'model_name'

# Compare all models
POST /compare_models
Body: FormData with 'file'
```

### Response Format
```json
{
  "success": true,
  "label": "COVID-19",
  "confidence": 0.8542,
  "confidence_level": "Cao",
  "probabilities": {
    "COVID-19": 0.8542,
    "Normal": 0.1234,
    "Pneumonia": 0.0224
  },
  "model_info": {
    "name": "DenseNet201 Model",
    "description": "DenseNet201 fine-tuned cho COVID detection",
    "input_size": [224, 224]
  },
  "recommendation": {
    "reliable": true,
    "message": "Kết quả tin cậy"
  }
}
```

## 🧪 Testing & Validation

### Automated Testing
```bash
# Test all API endpoints
python test_multi_model_api.py

# Test model performance comparison
python model_comparison.py

# Quick functionality test
python quick_test.py

# Final system test
python final_test.py
```

### Manual Testing
1. **Upload test images** từ thư mục `uploads/`
2. **Switch models** và compare performance
3. **Check confidence levels** cho reliability
4. **Use comparison mode** cho comprehensive analysis

## 📊 Performance Metrics

### Confidence Levels:
- 🟢 **Cao (>70%)**: Kết quả tin cậy, có thể sử dụng
- 🟡 **Trung bình (50-70%)**: Nên kiểm tra thêm
- 🔴 **Thấp (<50%)**: Kết quả không chắc chắn

### Model Comparison Results:
- **DenseNet201**: Highest accuracy, stable confidence
- **Optimized**: Good balance of speed vs accuracy  
- **Simple COVID**: Decent performance, fast inference
- **Original**: Baseline comparison, needs improvement

## 🔧 Development

### Project Structure
```
covid19-xray-detection/
├── app.py                          # Main Flask application
├── models/                         # AI models directory
│   ├── densenet201_covid_model.h5
│   ├── optimized_covid_model.h5
│   └── simple_covid_model.h5
├── templates/                      # Web templates
│   ├── index.html                  # Basic interface
│   └── multi_model.html           # Multi-model interface
├── static/                         # Static files
├── uploads/                        # Image uploads
├── requirements.txt               # Dependencies
├── simple_model_creator.py       # Create models
├── start_app.bat                 # Windows startup script
├── test_multi_model_api.py       # API testing
└── README.md                     # This file
```

### Key Features Implementation:
- **Multi-model loading**: Load all models at startup
- **Real-time switching**: Change models without restart
- **Adaptive preprocessing**: Different input sizes per model
- **Error handling**: Robust error management
- **Logging**: Comprehensive debug information

## 🎯 Best Practices

### For Medical Use:
1. **Always compare multiple models** for consensus
2. **Check confidence levels** before making decisions
3. **Use DenseNet201 model** for highest accuracy
4. **Validate with medical expertise** - AI is supportive tool

### For Development:
1. **Test all models** before deployment
2. **Monitor system resources** with multiple models
3. **Implement proper error handling**
4. **Use logging** for debugging
5. **Regular model validation** with new data

## 📋 Requirements

### System Requirements:
- **Python**: 3.8+ (recommended 3.10+)
- **RAM**: 8GB+ (for multiple models)
- **Storage**: 2GB+ for models
- **GPU**: Optional but recommended for faster inference

### Dependencies:
```bash
tensorflow>=2.16.1
opencv-python>=4.8.1
flask>=3.0.3
flask-cors
pillow
numpy>=1.26.4
matplotlib
seaborn
scikit-learn
```

## 🔮 Roadmap

### Upcoming Features:
- [ ] **Ensemble predictions** combining all models
- [ ] **Model performance dashboard**
- [ ] **Batch processing** for multiple images
- [ ] **Export results** to PDF/Excel
- [ ] **User authentication** and result history
- [ ] **Real-time model monitoring**

### Model Improvements:
- [ ] **COVID-Net** integration
- [ ] **Vision Transformer** models
- [ ] **Custom ensemble** architectures
- [ ] **Federated learning** capabilities

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original research**: [COVID-Net paper](https://www.sciencedirect.com/science/article/pii/S2352914820302537)
- **Dataset**: COVID-Net dataset
- **Architecture inspiration**: DenseNet, ResNet, Xception models
- **UI Components**: Bootstrap 5, Font Awesome

## 📞 Support

- 📧 **Email**: [your-email@domain.com]
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/covid19-xray-detection/issues)
- 📖 **Documentation**: [Wiki](https://github.com/your-username/covid19-xray-detection/wiki)

---

## 🎉 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models (if needed)
python download_models.py

# 3. Start backend
python app.py

# 4. Start frontend (in new terminal)
cd frontend && npm install && npm start

# 5. Open browser
# Frontend: http://localhost:3000
# Backend: http://localhost:5000
```

**🩺 Ready to detect COVID-19 with AI! Try the React frontend for best experience.**
