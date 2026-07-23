import React from 'react';
import { ShieldCheck, Clock, AlertTriangle, IndianRupee } from 'lucide-react';

// Formats basic markdown bold, lists, and paragraphs cleanly
function renderFormattedText(text) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let listItems = [];

  lines.forEach((line, idx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      if (listItems.length > 0) {
        elements.push(<ul key={`ul-${idx}`}>{listItems}</ul>);
        listItems = [];
      }
      return;
    }

    // Bullet item
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      const content = trimmed.substring(2);
      listItems.push(<li key={`li-${idx}`}>{parseBold(content)}</li>);
    } else {
      if (listItems.length > 0) {
        elements.push(<ul key={`ul-${idx}`}>{listItems}</ul>);
        listItems = [];
      }

      if (trimmed.startsWith('### ')) {
        elements.push(<h4 key={idx} style={{ marginTop: '0.85rem', marginBottom: '0.4rem', color: '#8B0000' }}>{parseBold(trimmed.substring(4))}</h4>);
      } else if (trimmed.startsWith('## ')) {
        elements.push(<h3 key={idx} style={{ marginTop: '1rem', marginBottom: '0.5rem', color: '#8B0000' }}>{parseBold(trimmed.substring(3))}</h3>);
      } else {
        elements.push(<p key={idx}>{parseBold(trimmed)}</p>);
      }
    }
  });

  if (listItems.length > 0) {
    elements.push(<ul key="ul-final">{listItems}</ul>);
  }

  return elements;
}

function parseBold(text) {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

export default function VerdictDisplay({ result }) {
  if (!result) return null;

  const { final_status, required_deposit, reasoner_analysis, extracted_data } = result;

  const colorMap = {
    APPROVED: '#2E8B57',
    PENDING_DEPOSIT: '#D2691E',
    REJECTED_OR_ESCALATED: '#8B0000',
    REJECTED: '#8B0000',
  };

  const statusColor = colorMap[final_status] || '#8B0000';

  const statusIcon = {
    APPROVED: <ShieldCheck size={26} color="#2E8B57" />,
    PENDING_DEPOSIT: <Clock size={26} color="#D2691E" />,
    REJECTED_OR_ESCALATED: <AlertTriangle size={26} color="#8B0000" />,
  }[final_status] || <AlertTriangle size={26} color="#8B0000" />;

  return (
    <div className="verdict-card">
      <div className="verdict-header">
        <div className={`verdict-title ${final_status}`}>
          {statusIcon}
          <span>Verdict: {final_status}</span>
        </div>

        {required_deposit !== undefined && (
          <div className="deposit-badge">
            Deposit Required: ₹{required_deposit.toLocaleString()}
          </div>
        )}
      </div>

      <div className="divider" />

      {reasoner_analysis ? (
        <div className="reasoning-body">
          {renderFormattedText(reasoner_analysis)}
        </div>
      ) : (
        <div style={{ color: '#8B0000', fontStyle: 'italic' }}>
          No reasoning analysis was produced.
        </div>
      )}

      {extracted_data && (
        <div style={{ marginTop: '0.5rem' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#666', marginBottom: '0.4rem' }}>
            EXTRACTED CLAIM METRICS
          </div>
          <div className="extracted-grid">
            {extracted_data.patient_name && (
              <div className="extracted-item">
                <div className="extracted-label">Patient Name</div>
                <div className="extracted-value">{extracted_data.patient_name}</div>
              </div>
            )}
            {extracted_data.claim_amount && (
              <div className="extracted-item">
                <div className="extracted-label">Claim Amount</div>
                <div className="extracted-value">₹{extracted_data.claim_amount.toLocaleString()}</div>
              </div>
            )}
            {extracted_data.diagnosis_code && (
              <div className="extracted-item">
                <div className="extracted-label">ICD-10 Code</div>
                <div className="extracted-value">{extracted_data.diagnosis_code}</div>
              </div>
            )}
            {extracted_data.treatment_category && (
              <div className="extracted-item">
                <div className="extracted-label">Category</div>
                <div className="extracted-value">{extracted_data.treatment_category}</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
