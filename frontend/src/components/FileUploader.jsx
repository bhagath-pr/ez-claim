import React, { useRef } from 'react';
import { UploadCloud, FileText, X } from 'lucide-react';

export default function FileUploader({ selectedFile, onFileSelect, onFileRemove, disabled }) {
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        onFileSelect(file);
      } else {
        alert('Please upload a valid PDF file.');
      }
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div>
      <div className="field-label">
        <FileText size={18} color="#8B0000" />
        Select Medical Bill (PDF)
      </div>

      {!selectedFile ? (
        <div
          className="dropzone"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleInputChange}
            accept=".pdf,application/pdf"
            className="file-input"
            disabled={disabled}
          />
          <UploadCloud className="dropzone-icon" />
          <div className="dropzone-text">
            Drag & drop PDF hospital invoice here, or click to browse
          </div>
          <div className="dropzone-subtext">PDF files only (max 10MB)</div>
        </div>
      ) : (
        <div className="selected-file-badge">
          <div className="file-info">
            <FileText size={20} />
            <div>
              <div>{selectedFile.name}</div>
              <div style={{ fontSize: '0.75rem', color: '#666', fontWeight: 400 }}>
                {(selectedFile.size / 1024).toFixed(1)} KB
              </div>
            </div>
          </div>
          {!disabled && (
            <button
              type="button"
              className="remove-file-btn"
              onClick={onFileRemove}
              title="Remove file"
            >
              <X size={18} />
            </button>
          )}
        </div>
      )}
    </div>
  );
}
