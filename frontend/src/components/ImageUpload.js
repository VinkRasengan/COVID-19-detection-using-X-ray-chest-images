import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { FaCloudUploadAlt, FaImage, FaSpinner } from 'react-icons/fa';
import axios from 'axios';

const ImageUpload = ({ onPrediction, onError, onLoading, loading }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false
  });

  const handleSubmit = async () => {
    if (!selectedFile) {
      onError('Vui lòng chọn một ảnh X-ray');
      return;
    }

    onLoading();
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    console.log('Sending request to backend with file:', selectedFile.name);
    console.log('FormData contents:', formData.get('file'));

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });

      console.log('Response received:', response.data);

      if (response.data.success) {
        // Check if we have multiple model results
        if (response.data.results && Object.keys(response.data.results).length > 1) {
          // New format with multiple models
          onPrediction({
            results: response.data.results,
            best_model: response.data.best_model,
            total_models: response.data.total_models,
            summary: response.data.summary,
            imageUrl: preview,
            filename: selectedFile.name
          });
        } else {
          // Legacy format with single model
          onPrediction({
            label: response.data.label,
            probabilities: response.data.probabilities,
            imageUrl: preview,
            filename: selectedFile.name
          });
        }
      } else {
        console.error('Server returned error:', response.data.error);
        onError(response.data.error || 'Lỗi không xác định từ server');
      }
    } catch (error) {
      console.error('Full error object:', error);
      console.error('Error response:', error.response);
      console.error('Error message:', error.message);
      
      if (error.response && error.response.data && error.response.data.error) {
        onError(error.response.data.error);
      } else if (error.code === 'ECONNABORTED') {
        onError('Timeout: Server mất quá nhiều thời gian để phản hồi');
      } else if (error.code === 'ERR_NETWORK') {
        onError('Lỗi kết nối: Không thể kết nối đến server');
      } else {
        onError('Lỗi khi gửi ảnh lên server: ' + (error.message || 'Unknown error'));
      }
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 style={{ marginBottom: '2rem', color: '#333' }}>
        🩺 Phát hiện COVID-19 từ ảnh X-Ray
      </h1>
      
      <p style={{ marginBottom: '2rem', color: '#666', fontSize: '1.1rem' }}>
        Upload ảnh X-ray phổi để phân tích và dự đoán tình trạng bệnh
      </p>

      <div
        {...getRootProps()}
        className={`upload-area ${isDragActive ? 'dragover' : ''}`}
      >
        <input {...getInputProps()} />
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          id="file-input"
        />
        
        {!preview ? (
          <>
            <FaCloudUploadAlt className="upload-icon" />
            <div className="upload-text">
              {isDragActive ? 'Thả ảnh vào đây' : 'Kéo thả ảnh hoặc click để chọn'}
            </div>
            <div className="upload-hint">
              Hỗ trợ: PNG, JPG, JPEG
            </div>
          </>
        ) : (
          <div>
            <FaImage className="upload-icon" style={{ fontSize: '2rem' }} />
            <div className="upload-text">Ảnh đã được chọn</div>
            <img src={preview} alt="Preview" className="image-preview" />
          </div>
        )}
      </div>

      {preview && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          style={{ marginTop: '2rem' }}
        >
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <>
                <FaSpinner className="spinner" />
                Đang phân tích...
              </>
            ) : (
              'Phân tích ảnh'
            )}
          </button>
          
          <button
            className="btn btn-secondary"
            onClick={() => {
              setSelectedFile(null);
              setPreview(null);
            }}
            disabled={loading}
          >
            Chọn ảnh khác
          </button>
        </motion.div>
      )}

      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="loading"
          style={{ marginTop: '2rem' }}
        >
          <div className="spinner"></div>
          <p>Đang phân tích ảnh X-ray...</p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default ImageUpload; 