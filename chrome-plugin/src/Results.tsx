import React, { useEffect, useState } from 'react';
import type { AnalysisResult, ModuleDecision, LinguisticSignal, LinguisticFlag, MatchedSource } from './Api';
import './Results.css';

// ── helpers ──────────────────────────────────────────────────────────────────

function verdict(result: AnalysisResult): 'fake' | 'real' | 'uncertain' {
  const p = result.prediction?.toLowerCase() ?? '';
  if (p.includes('fake')) return 'fake';
  if (p.includes('real')) return 'real';
  return 'uncertain';
}

function pct(v: number) {
  return `${Math.round(v * 100)}%`;
}

function lingClass(score: number): 'good' | 'medium' | 'bad' {
  if (score >= 0.6) return 'good';
  if (score >= 0.35) return 'medium';
  return 'bad';
}

function sourceClass(status: string): 'reliable' | 'unreliable' | 'unknown' {
  const s = status?.toLowerCase() ?? '';
  if (s.includes('reliable') && !s.includes('un')) return 'reliable';
  if (s.includes('unreliable') || s.includes('fake') || s.includes('false')) return 'unreliable';
  return 'unknown';
}

function sourceIcon(cls: 'reliable' | 'unreliable' | 'unknown') {
  if (cls === 'reliable') return '✅';
  if (cls === 'unreliable') return '❌';
  return '❓';
}

function moduleClass(realProb: number): 'real' | 'fake' | 'uncertain' {
  if (realProb >= 0.6) return 'real';
  if (realProb <= 0.4) return 'fake';
  return 'uncertain';
}

function similarityColor(sim: number) {
  if (sim >= 0.7) return '#16a34a';
  if (sim >= 0.4) return '#d97706';
  return '#94a3b8';
}

// Circumference for r=50 circle
const CIRC = 2 * Math.PI * 50;

// ── sub-components ────────────────────────────────────────────────────────────

function Gauge({ value, cls }: { value: number; cls: 'fake' | 'real' | 'uncertain' }) {
  const offset = CIRC * (1 - value);
  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 120 120">
        <circle className="gauge-track" cx="60" cy="60" r="50" />
        <circle
          className={`gauge-fill ${cls}`}
          cx="60" cy="60" r="50"
          strokeDasharray={CIRC}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="gauge-label">
        <span className={`gauge-pct ${cls}`}>{Math.round(value * 100)}</span>
        <span className="gauge-sub">%</span>
      </div>
    </div>
  );
}

function ModuleRow({ d }: { d: ModuleDecision }) {
  const cls = moduleClass(d.real_probability);
  const barWidth = d.has_data ? `${d.real_probability * 100}%` : '0%';
  const label = d.module.replace(/_/g, ' ');
  return (
    <div className="module-row">
      <span className="module-name">{label}</span>
      {d.has_data ? (
        <>
          <div className="module-bar-bg">
            <div className={`module-bar-fill ${cls}`} style={{ width: barWidth }} />
          </div>
          <span className="module-weight">w {d.effective_weight.toFixed(2)}</span>
          <span className={`module-prob ${cls}`}>{pct(d.real_probability)} real</span>
        </>
      ) : (
        <span className="module-no-data">No data available</span>
      )}
      {d.reasoning && (
        <span className="module-reasoning">{d.reasoning}</span>
      )}
    </div>
  );
}

function SourceRow({ s }: { s: MatchedSource }) {
  const simPct = Math.round(s.similarity * 100);
  const color = similarityColor(s.similarity);
  const date = s.published_at ? new Date(s.published_at).toLocaleDateString() : null;
  return (
    <div className="source-row">
      <div className="source-row-main">
        <span className="source-row-name">{s.source}</span>
        <span className="source-row-sim" style={{ color }}>{simPct}% match</span>
      </div>
      <div className="source-row-title">
        {s.url
          ? <a href={s.url} target="_blank" rel="noreferrer" className="source-row-link">{s.title} ↗</a>
          : s.title
        }
      </div>
      {date && <div className="source-row-date">{date}</div>}
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

const Results: React.FC = () => {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    chrome.storage.local.get(['analysisResult'], (data: { [key: string]: unknown }) => {
      if (data.analysisResult) {
        setResult(data.analysisResult as AnalysisResult);
      }
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="loading-state">
        <div className="spinner" />
        <span>Loading analysis…</span>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="no-data-state">
        <span style={{ fontSize: 48 }}>🔍</span>
        <p>No analysis results found.</p>
        <p style={{ fontSize: 13 }}>Open the extension and analyze a page first.</p>
      </div>
    );
  }

  const vrd = verdict(result);
  const fakePct = result.fake_probability ?? 0;
  const realPct = result.real_probability ?? 0;
  const mainValue = vrd === 'real' ? realPct : fakePct;
  const srcCls = sourceClass(result.source_status ?? '');
  const ling = result.linguistic;
  const lingCls = lingClass(ling?.score ?? 0);
  const cs = result.cross_source;

  const verdictLabel = {
    fake: 'Likely Fake',
    real: 'Likely Real',
    uncertain: 'Uncertain',
  }[vrd];

  const dotColor = {
    reliable: '#22c55e',
    unreliable: '#ef4444',
    unknown: '#64748b',
  }[srcCls];

  return (
    <div className="results-page">

      {/* Header */}
      {result.source_domain && (
        <div className="header">
          <div className="domain-badge">
            <span className="dot" style={{ background: dotColor }} />
            {result.source_domain}
          </div>
        </div>
      )}

      {/* Verdict card */}
      <div className={`verdict-card ${vrd}`}>
        <Gauge value={mainValue} cls={vrd} />
        <div className="verdict-info">
          <div className={`verdict-headline ${vrd}`}>{verdictLabel}</div>
          <div className="prob-bars">
            <div className="prob-row">
              <span>Fake</span>
              <div className="prob-bar-bg">
                <div className="prob-bar-fill fake" style={{ width: pct(fakePct) }} />
              </div>
              <span>{pct(fakePct)}</span>
            </div>
            <div className="prob-row">
              <span>Real</span>
              <div className="prob-bar-bg">
                <div className="prob-bar-fill real" style={{ width: pct(realPct) }} />
              </div>
              <span>{pct(realPct)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Source + Linguistic */}
      <div className="grid-2">

        {/* Source */}
        <div className="card">
          <div className="card-title">Source Verification</div>
          <div className="source-status-row">
            <span className="status-icon">{sourceIcon(srcCls)}</span>
            <span className={`status-label ${srcCls}`}>
              {result.source_status || 'Unknown'}
            </span>
          </div>
          {result.source_domain && (
            <div className="source-domain-text">{result.source_domain}</div>
          )}
        </div>

        {/* Linguistic */}
        <div className="card">
          <div className="card-title">Linguistic Analysis</div>
          <div className="ling-score-row">
            <span className={`ling-score-num ${lingCls}`}>
              {Math.round((ling?.score ?? 0) * 100)}
            </span>
            <span className="ling-score-denom">/100</span>
          </div>
          <div className="ling-bar-bg">
            <div
              className={`ling-bar-fill ${lingCls}`}
              style={{ width: pct(ling?.score ?? 0) }}
            />
          </div>
          {ling?.flags?.length > 0 && (
            <div className="ling-flags">
              {ling.flags.map((f: LinguisticFlag, i: number) => (
                <span
                  key={i}
                  className={`ling-flag ${f.positive ? 'flag-positive' : 'flag-warning'}`}
                  title={f.description}
                >
                  {f.code}
                </span>
              ))}
            </div>
          )}
          {ling?.explanation && (
            <div className="ling-signals">
              {ling.explanation.verdict && (
                <p className="ling-verdict">{ling.explanation.verdict}</p>
              )}
              {ling.explanation.signals?.map((s: LinguisticSignal, i: number) => (
                <div key={i} className="signal-row">
                  <span className={`signal-dot ${s.positive === true ? 'pos' : s.positive === false ? 'neg' : 'neu'}`} />
                  <span className="signal-label">{s.label}</span>
                  <span className="signal-value">{s.value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Module Breakdown */}
      {result.decisions?.length > 0 && (
        <div className="card modules-card">
          <div className="card-title">Module Breakdown</div>
          {result.decisions.map((d, i) => (
            <ModuleRow key={i} d={d} />
          ))}
        </div>
      )}

      {/* Cross-Source */}
      {cs && (
        <div className="card">
          <div className="card-title">Cross-Source Comparison</div>
          <div className="cs-header">
            <div className="fact-stat">
              <span className="fact-stat-num">{cs.sources_found ?? 0}</span>
              <span className="fact-stat-label">Sources found</span>
            </div>
            {cs.verdict && (
              <span className="cs-verdict">{cs.verdict}</span>
            )}
          </div>
          {cs.matched_sources?.length > 0 ? (
            <div className="source-list">
              {cs.matched_sources.map((s, i) => (
                <SourceRow key={i} s={s} />
              ))}
            </div>
          ) : (
            <div className="empty-state">No matching sources found</div>
          )}
        </div>
      )}

      {/* Raw Response */}
      <div className="card raw-response-card">
        <div className="card-title">Raw Backend Response</div>
        <pre className="raw-response-pre">{JSON.stringify(result, null, 2)}</pre>
      </div>

    </div>
  );
};

export default Results;
