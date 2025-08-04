import React from 'react';
import { motion } from 'framer-motion';
import { FaVirus, FaLungs, FaSmile, FaRedo, FaTrophy, FaBrain } from 'react-icons/fa';

const ResultDisplay = ({ prediction, onReset }) => {
  // Check if we have results from multiple models
  const hasMultipleModels = prediction.results && Object.keys(prediction.results).length > 1;
  
  const getLabelIcon = (label) => {
    switch (label) {
      case 'COVID-19':
        return <FaVirus />;
      case 'Normal':
        return <FaSmile />;
      case 'Pneumonia':
        return <FaLungs />;
      default:
        return <FaVirus />;
    }
  };

  const getLabelClass = (label) => {
    switch (label) {
      case 'COVID-19':
        return 'prediction-covid';
      case 'Normal':
        return 'prediction-normal';
      case 'Pneumonia':
        return 'prediction-pneumonia';
      default:
        return 'prediction-covid';
    }
  };

  const getProbabilityClass = (className) => {
    switch (className) {
      case 'COVID-19':
        return 'probability-fill-covid';
      case 'Normal':
        return 'probability-fill-normal';
      case 'Pneumonia':
        return 'probability-fill-pneumonia';
      default:
        return 'probability-fill-covid';
    }
  };

  const getLabelText = (label) => {
    switch (label) {
      case 'COVID-19':
        return 'COVID-19';
      case 'Normal':
        return 'BÌNH THƯỜNG';
      case 'Pneumonia':
        return 'VIÊM PHỔI';
      default:
        return label;
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.7) return '#28a745';
    if (confidence > 0.5) return '#ffc107';
    return '#dc3545';
  };

  const renderSingleModelResult = (result, modelName) => {
    const { label, probabilities, confidence, confidence_level } = result;
    const sortedProbabilities = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    return (
      <motion.div
        key={modelName}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="model-result-card"
        style={{
          background: '#fff',
          borderRadius: '15px',
          padding: '1.5rem',
          marginBottom: '1.5rem',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          border: prediction.best_model === modelName ? '3px solid #28a745' : '1px solid #e9ecef'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
          <FaBrain style={{ marginRight: '0.5rem', color: '#6c757d' }} />
          <h3 style={{ margin: 0, color: '#333', fontSize: '1.2rem' }}>
            {modelName}
            {prediction.best_model === modelName && (
              <FaTrophy style={{ marginLeft: '0.5rem', color: '#ffc107' }} title="Model tốt nhất" />
            )}
          </h3>
        </div>

        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          marginBottom: '1rem',
          padding: '0.5rem',
          background: '#f8f9fa',
          borderRadius: '8px'
        }}>
          <div className={`prediction-label ${getLabelClass(label)}`} style={{ marginRight: '1rem' }}>
            {getLabelIcon(label)} {getLabelText(label)}
          </div>
          <div style={{ 
            color: getConfidenceColor(confidence),
            fontWeight: 'bold',
            fontSize: '0.9rem'
          }}>
            {confidence_level} ({confidence * 100}%)
          </div>
        </div>

        <div className="probability-container">
          <h4 style={{ marginBottom: '0.5rem', color: '#666', fontSize: '0.9rem' }}>
            📊 Chi tiết xác suất
          </h4>
          
          {sortedProbabilities.map(([className, probability], index) => (
            <motion.div
              key={className}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 + index * 0.1 }}
              className="probability-item"
              style={{ marginBottom: '0.5rem' }}
            >
              <div className="probability-label" style={{ fontSize: '0.8rem' }}>
                {getLabelIcon(className)} {getLabelText(className)}
              </div>
              <div className="probability-bar">
                <motion.div
                  className={`probability-fill ${getProbabilityClass(className)}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${probability * 100}%` }}
                  transition={{ delay: 0.6 + index * 0.1, duration: 0.8 }}
                />
              </div>
              <div className="probability-value" style={{ fontSize: '0.8rem' }}>
                {(probability * 100).toFixed(1)}%
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    );
  };

  const renderLegacyResult = () => {
    // Fallback for old single model format
    const { label, probabilities, imageUrl, filename } = prediction;
    const sortedProbabilities = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    return (
      <>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <img src={imageUrl} alt="X-Ray" className="result-image" />
          <p style={{ color: '#666', marginBottom: '1rem' }}>
            <strong>File:</strong> {filename}
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className={`prediction-label ${getLabelClass(label)}`}>
            {getLabelIcon(label)} {getLabelText(label)}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="probability-container"
        >
          <h3 style={{ marginBottom: '1rem', color: '#333' }}>
            📈 Chi tiết xác suất
          </h3>
          
          {sortedProbabilities.map(([className, probability], index) => (
            <motion.div
              key={className}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.8 + index * 0.1 }}
              className="probability-item"
            >
              <div className="probability-label">
                {getLabelIcon(className)} {getLabelText(className)}
              </div>
              <div className="probability-bar">
                <motion.div
                  className={`probability-fill ${getProbabilityClass(className)}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${probability * 100}%` }}
                  transition={{ delay: 1 + index * 0.1, duration: 0.8 }}
                />
              </div>
              <div className="probability-value">
                {(probability * 100).toFixed(1)}%
              </div>
            </motion.div>
          ))}
        </motion.div>
      </>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="result-container"
    >
      <h1 style={{ marginBottom: '2rem', color: '#333' }}>
        📊 Kết quả phân tích {hasMultipleModels && `(${Object.keys(prediction.results).length} models)`}
      </h1>

      {hasMultipleModels ? (
        <>
          {/* Summary section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            style={{
              background: '#e8f5e8',
              padding: '1rem',
              borderRadius: '10px',
              marginBottom: '2rem',
              border: '1px solid #28a745'
            }}
          >
            <h3 style={{ margin: '0 0 0.5rem 0', color: '#155724' }}>
              🏆 Kết quả tổng hợp
            </h3>
            <p style={{ margin: 0, color: '#155724' }}>
              <strong>Model tốt nhất:</strong> {prediction.best_model} 
              {prediction.summary.best_result && (
                <span> - {getLabelText(prediction.summary.best_result.label)} 
                ({(prediction.summary.best_result.confidence * 100).toFixed(1)}%)</span>
              )}
            </p>
          </motion.div>

          {/* Individual model results */}
          <div className="models-grid">
            {Object.entries(prediction.results).map(([modelName, result]) => 
              renderSingleModelResult(result, modelName)
            )}
          </div>
        </>
      ) : (
        renderLegacyResult()
      )}

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.2 }}
        style={{ marginTop: '2rem' }}
      >
        <button className="btn btn-primary" onClick={onReset}>
          <FaRedo /> Phân tích ảnh khác
        </button>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.4 }}
        style={{ 
          marginTop: '2rem', 
          padding: '1rem', 
          background: '#f8f9fa', 
          borderRadius: '10px',
          fontSize: '0.9rem',
          color: '#666'
        }}
      >
        <p><strong>⚠️ Lưu ý:</strong> Đây chỉ là kết quả dự đoán từ AI, không thay thế chẩn đoán của bác sĩ.</p>
        <p>Vui lòng tham khảo ý kiến chuyên môn y tế để có kết luận chính xác.</p>
        {hasMultipleModels && (
          <p><strong>💡 Gợi ý:</strong> Model có độ tin cậy cao nhất được đánh dấu bằng biểu tượng 🏆.</p>
        )}
      </motion.div>
    </motion.div>
  );
};

export default ResultDisplay; 