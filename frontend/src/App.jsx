import React, { useState } from 'react';
import FileUploader from './components/FileUploader';
import StatusTracker from './components/StatusTracker';
import VerdictDisplay from './components/VerdictDisplay';
import { Activity, Play } from 'lucide-react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [steps, setSteps] = useState([]);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [isStatusExpanded, setIsStatusExpanded] = useState(true);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    setSteps([]);
  };

  const handleFileRemove = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    setSteps([]);
  };

  const handleProcessClaim = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);
    setResult(null);
    setIsStatusExpanded(true);
    setSteps([
      { step: 'upload', status: 'completed', message: 'File uploaded successfully.' },
      { step: 'extract_pdf', status: 'running', message: 'Extracting content from PDF...' }
    ]);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Attempt SSE Streaming approach first for real-time progress
      try {
        const streamResponse = await fetch('/api/process-claim/stream', {
          method: 'POST',
          body: formData,
        });

        if (streamResponse.ok) {
          const reader = streamResponse.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let buffer = '';

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const eventData = JSON.parse(line.replace('data: ', ''));

                setSteps((prev) => {
                  const existingIdx = prev.findIndex((s) => s.step === eventData.step);
                  if (existingIdx >= 0) {
                    const next = [...prev];
                    next[existingIdx] = {
                      step: eventData.step,
                      status: eventData.status,
                      message: eventData.message
                    };
                    return next;
                  } else {
                    return [...prev, {
                      step: eventData.step,
                      status: eventData.status,
                      message: eventData.message
                    }];
                  }
                });

                if (eventData.step === 'reasoning_graph' && eventData.status === 'completed') {
                  setResult(eventData.payload);
                  setIsStatusExpanded(false);
                }
              }
            }
          }
          setIsProcessing(false);
          return;
        }
      } catch (streamErr) {
        console.warn('Streaming endpoint unavailable, falling back to standard API', streamErr);
      }

      // Fallback: Standard API Endpoint
      const response = await fetch('/api/process-claim', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({ detail: 'Server Error' }));
        throw new Error(errJson.detail || 'Claim processing failed.');
      }

      const data = await response.json();

      setSteps([
        { step: 'upload', status: 'completed', message: 'File uploaded successfully.' },
        { step: 'extract_pdf', status: 'completed', message: 'PDF text & tables extracted.' },
        { step: 'ai_analysis', status: 'completed', message: 'Document analyzed with AI.' },
        { step: 'ai_extraction', status: 'completed', message: 'AI Extraction Complete' },
        { step: 'reasoning_graph', status: 'completed', message: 'Reasoning graph executed.' }
      ]);

      setResult({
        final_status: data.final_status,
        required_deposit: data.required_deposit,
        reasoner_analysis: data.reasoner_analysis,
        extracted_data: data.extracted_data
      });

      setIsStatusExpanded(false);
    } catch (err) {
      setError(err.message || 'An error occurred during claim processing.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header matching st.title */}
      <div className="header-card">
        <h1 className="header-title">
          <Activity size={32} color="#8B0000" />
          EZ Claim Processing
        </h1>
        <p className="header-subtitle">
          Upload a hospital invoice (PDF) to automatically extract, process, and evaluate the claim using AI.
        </p>
      </div>

      {/* Main Form Card matching Streamlit inputs */}
      <div className="main-card">
        <FileUploader
          selectedFile={selectedFile}
          onFileSelect={handleFileSelect}
          onFileRemove={handleFileRemove}
          disabled={isProcessing}
        />

        <button
          type="button"
          className="btn-primary"
          onClick={handleProcessClaim}
          disabled={!selectedFile || isProcessing}
        >
          <Play size={18} />
          {isProcessing ? 'Processing Claim...' : 'Process Claim'}
        </button>
      </div>

      {/* Status Container matching st.status */}
      <StatusTracker
        steps={steps}
        isProcessing={isProcessing}
        isComplete={!!result}
        error={error}
        isExpanded={isStatusExpanded}
        onToggleExpand={() => setIsStatusExpanded(!isStatusExpanded)}
      />

      {/* Verdict & Reasoner Output Section */}
      {result && <VerdictDisplay result={result} />}
    </div>
  );
}
