import React from 'react';

const LoadingIndicator: React.FC = () => {
  return (
    <div className="loading-indicator">
      <div className="loading-dot"></div>
      <div className="loading-dot"></div>
      <div className="loading-dot"></div>
    </div>
  );
};

export default LoadingIndicator;