import React, { useState } from 'react';
import { analyzeText } from './Api';

const App: React.FC = () => {
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setError('');
    setLoading(true);

    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true,
      });

      if (!tab.id) {
        setError('No active tab found');
        setLoading(false);
        return;
      }

      const url = tab.url || '';
      const nonArticleDomains = [
        'youtube.com', 'youtu.be', 'twitter.com', 'x.com', 'facebook.com',
        'instagram.com', 'reddit.com', 'tiktok.com', 'linkedin.com',
        'google.com', 'gmail.com', 'wikipedia.org', 'amazon.com',
        'netflix.com', 'spotify.com', 'twitch.tv', 'discord.com',
        'github.com', 'stackoverflow.com',
      ];
      const isNonArticle = nonArticleDomains.some(d => url.includes(d));
      if (isNonArticle || !url.startsWith('http')) {
        setError('This page does not appear to be a news article.');
        setLoading(false);
        return;
      }

      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['content.js'],
      });

      const response = await chrome.tabs.sendMessage(tab.id, {
        type: 'EXTRACT_ARTICLE',
      });

      if (!response.success || !response.textContent || response.textContent.length < 200) {
        setError('No article content found. Please open a specific news article.');
        setLoading(false);
        return;
      }

      const extractedTitle = response.title || '';
      const extractedText = response.textContent || '';
      const extractedUrl = tab.url || '';

      const data = await analyzeText(extractedText, extractedUrl, extractedTitle);

      // Store results and open full results page
      await chrome.storage.local.set({
        analysisResult: data,
        articleTitle: extractedTitle,
        extractedText: extractedText,
      });

      chrome.tabs.create({ url: chrome.runtime.getURL('results.html') });
    } catch (err) {
      setError('Failed to analyze. Make sure you are on a webpage and the API is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: '350px', padding: '16px' }}>
      <h2>Fake News Detector</h2>
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
};

export default App;
