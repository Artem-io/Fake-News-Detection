import React, { useEffect, useState } from 'react';
import type { AnalysisResult } from './Api';

const Results: React.FC = () => {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [extractedText, setExtractedText] = useState('');

  useEffect(() => {
    chrome.storage.local.get(['analysisResult', 'extractedText'], (data: { [key: string]: unknown }) => {
      if (data.analysisResult) {
        setResult(data.analysisResult as AnalysisResult);
        setExtractedText((data.extractedText as string) || '');
      }
    });
  }, []);

  if (!result) {
    return (
      <div style={{ maxWidth: '900px', margin: '40px auto', padding: '20px', textAlign: 'center' }}>
        <p>No analysis results found.</p>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '900px', margin: '40px auto', padding: '20px' }}>
      <h1>Analysis Results</h1>

      <h2>Backend Response</h2>
      <pre style={{
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontSize: '13px',
        lineHeight: '1.5',
        padding: '16px',
        border: '1px solid #ccc',
        borderRadius: '8px',
        marginBottom: '24px',
      }}>
        {JSON.stringify(result, null, 2)}
      </pre>

      <h2>Extracted Text (sent to API)</h2>
      <pre style={{
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontSize: '13px',
        lineHeight: '1.5',
        padding: '16px',
        border: '1px solid #ccc',
        borderRadius: '8px',
      }}>
        {extractedText}
      </pre>
    </div>
  );
};

export default Results;
