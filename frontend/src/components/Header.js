import React from 'react';
import { FaLungs, FaVirus } from 'react-icons/fa';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <FaLungs className="logo-icon" />
          <span>COVID-19 X-Ray Detector</span>
        </div>
        <div className="header-info">
          <FaVirus style={{ marginRight: '0.5rem' }} />
          <span>AI-Powered Diagnosis</span>
        </div>
      </div>
    </header>
  );
};

export default Header; 