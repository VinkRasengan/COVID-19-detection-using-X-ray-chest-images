import React, { useState } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import Header from './components/Header';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePrediction = (result) => {
    setPrediction(result);
    setLoading(false);
    setError(null);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setLoading(false);
  };

  const handleLoading = () => {
    setLoading(true);
    setError(null);
  };

  const resetPrediction = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="App">
      <Header />
      <main className="main-content">
        <div className="container">
          {!prediction ? (
            <ImageUpload 
              onPrediction={handlePrediction}
              onError={handleError}
              onLoading={handleLoading}
              loading={loading}
            />
          ) : (
            <ResultDisplay 
              prediction={prediction}
              onReset={resetPrediction}
            />
          )}
          
          {error && (
            <div className="error-message">
              <p>❌ {error}</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
