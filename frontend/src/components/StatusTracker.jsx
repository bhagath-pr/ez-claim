import React from 'react';
import { Loader2, CheckCircle2, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

export default function StatusTracker({ steps, isProcessing, isComplete, error, isExpanded, onToggleExpand }) {
  if (!isProcessing && steps.length === 0 && !error) return null;

  return (
    <div className="status-card">
      <div className="status-header">
        <div className="status-title">
          {isProcessing ? (
            <>
              <Loader2 className="spinner" size={18} />
              <span>Processing Claim...</span>
            </>
          ) : error ? (
            <>
              <AlertCircle className="error-icon" size={18} />
              <span style={{ color: '#8B0000' }}>Processing Error</span>
            </>
          ) : (
            <>
              <CheckCircle2 className="check-icon" size={18} />
              <span>Claim Processing Complete</span>
            </>
          )}
        </div>
        <button
          onClick={onToggleExpand}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#666', display: 'flex', alignItems: 'center' }}
        >
          {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>

      {isExpanded && (
        <div className="status-list">
          {steps.map((step, idx) => (
            <div
              key={idx}
              className={`status-item ${
                step.status === 'running'
                  ? 'active'
                  : step.status === 'completed'
                  ? 'completed'
                  : step.status === 'error'
                  ? 'error'
                  : ''
              }`}
            >
              {step.status === 'running' && <Loader2 className="spinner" size={16} />}
              {step.status === 'completed' && <CheckCircle2 className="check-icon" size={16} />}
              {step.status === 'error' && <AlertCircle className="error-icon" size={16} />}
              <span>{step.message}</span>
            </div>
          ))}
          {error && (
            <div className="status-item error">
              <AlertCircle className="error-icon" size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
