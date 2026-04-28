// src/components/CodeDetector.jsx
import React, { useState, useContext } from 'react';
import { detectCode } from '../services/api';
import { HistoryContext } from '../App';
import {
  C, Card, SectionLabel, ScoreMeter, VerdictBadge,
  SignalRow, Spinner, ErrorBox, RawScores, DatasetBadge
} from './ui';

const LANG_OPTIONS = ['auto', 'python', 'javascript', 'typescript', 'java', 'cpp', 'rust', 'go', 'sql'];

export default function CodeDetector() {
  const { onResult } = useContext(HistoryContext);
  const [code, setCode] = useState('');
  const [lang, setLang] = useState('auto');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = async () => {
    setError(null); setResult(null); setLoading(true);
    try {
      const data = await detectCode(code, lang === 'auto' ? null : lang);
      setResult(data);
      onResult('code', data.verdict, data.confidence);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const canRun = code.trim().length >= 10 && !loading;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <Card>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
          <SectionLabel>Language (optional)</SectionLabel>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
            {LANG_OPTIONS.map(l => (
              <button key={l} onClick={() => setLang(l)} style={{
                padding: '4px 10px', borderRadius: 12, cursor: 'pointer',
                fontSize: 11, fontFamily: 'monospace', transition: 'all .15s',
                background: lang === l ? `${C.neutral}20` : '#06060e',
                border: `1px solid ${lang === l ? C.neutral : C.border}`,
                color: lang === l ? C.neutral : C.muted,
              }}>{l}</button>
            ))}
          </div>
        </div>

        <SectionLabel>Code to Analyze (min 10 chars)</SectionLabel>
        <textarea value={code} onChange={e => setCode(e.target.value)}
          placeholder="// Paste any code here — Python, JavaScript, Java, C++, SQL..."
          style={{ width: '100%', minHeight: 240, background: '#05050d',
            border: `1px solid ${C.border}`, borderRadius: 8, color: '#7ee787',
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, lineHeight: 1.8,
            padding: '14px 16px', resize: 'vertical', outline: 'none',
            transition: 'border-color .2s' }}
          onFocus={e => e.target.style.borderColor = C.neutral}
          onBlur={e => e.target.style.borderColor = C.border} />
        <div style={{ textAlign: 'right', marginTop: 5, marginBottom: 12,
          fontSize: 11, fontFamily: 'monospace', color: C.muted }}>
          {code.split('\n').length} lines · {code.length} chars
        </div>

        <button onClick={run} disabled={!canRun} style={{
          width: '100%', padding: '14px',
          background: canRun ? `linear-gradient(135deg, ${C.neutral}, ${C.human})` : '#1a1a2e',
          border: 'none', borderRadius: 7, color: canRun ? '#fff' : C.muted,
          fontFamily: 'monospace', fontSize: 12, letterSpacing: 3,
          textTransform: 'uppercase', cursor: canRun ? 'pointer' : 'not-allowed',
          transition: 'all .2s' }}>
          {loading ? 'Analyzing Code...' : '▶  Analyze Code'}
        </button>
        {error && <ErrorBox message={error} />}
      </Card>

      {loading && <Card><Spinner /></Card>}

      {result && !loading && (
        <>
          <Card>
            <VerdictBadge verdict={result.verdict} confidence={result.confidence} />
            <ScoreMeter aiScore={result.breakdown?.ai_score} humanScore={result.breakdown?.human_score} />
            {result.detected_language && (
              <div style={{ textAlign: 'center', marginBottom: 10 }}>
                <span style={{ padding: '3px 12px', borderRadius: 12, fontSize: 11,
                  background: `${C.neutral}15`, border: `1px solid ${C.neutral}30`,
                  color: C.neutral, fontFamily: 'monospace' }}>
                  Detected: {result.detected_language}
                </span>
              </div>
            )}
            {result.summary && (
              <p style={{ color: '#999', fontSize: 13, lineHeight: 1.85, padding: '12px 14px',
                background: '#05050d', borderRadius: 7, borderLeft: `3px solid ${C.human}` }}>
                {result.summary}
              </p>
            )}
          </Card>

          {/* Code features */}
          {result.code_features && (
            <Card>
              <SectionLabel>📊 Code Feature Analysis</SectionLabel>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(130px,1fr))', gap: 8 }}>
                {[
                  ['Lines', result.code_features.line_count, C.neutral],
                  ['Comment Ratio', `${result.code_features.comment_ratio_pct}%`,
                    result.code_features.comment_ratio_pct > 40 ? C.ai : result.code_features.comment_ratio_pct < 5 ? C.human : C.neutral],
                  ['Avg Identifier', `${result.code_features.avg_identifier_length} chars`,
                    result.code_features.avg_identifier_length > 14 ? C.ai : C.human],
                  ['Error Handling', `${result.code_features.error_handling_ratio_pct}%`,
                    result.code_features.error_handling_ratio_pct > 60 ? C.ai : C.neutral],
                  ['Docstrings', result.code_features.has_docstrings ? 'YES ⚠' : 'No ✓',
                    result.code_features.has_docstrings ? C.ai : C.human],
                  ['Human Markers', result.code_features.human_code_markers,
                    result.code_features.human_code_markers > 0 ? C.human : C.muted],
                ].map(([label, val, color]) => (
                  <div key={label} style={{ padding: '10px 12px', background: '#06060e',
                    borderRadius: 6, border: `1px solid ${C.border}`, textAlign: 'center' }}>
                    <div style={{ fontSize: 15, fontWeight: 700, color, fontFamily: 'monospace' }}>{val}</div>
                    <div style={{ fontSize: 10, color: C.muted, marginTop: 3,
                      fontFamily: 'monospace', letterSpacing: 1, textTransform: 'uppercase' }}>{label}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {result.signals?.length > 0 && (
            <Card>
              <SectionLabel>🔍 Detection Signals</SectionLabel>
              {result.signals.map((s, i) => (
                <SignalRow key={i} signal={s} last={i === result.signals.length - 1} />
              ))}
            </Card>
          )}

          <Card>
            <SectionLabel>🤗 Raw Model Output — {result.model_used}</SectionLabel>
            <RawScores scores={result.breakdown?.raw_scores} />
          </Card>

          <DatasetBadge dataset={result.dataset_info} />

          <button onClick={() => { setResult(null); setCode(''); }}
            style={{ alignSelf: 'flex-start', background: 'none',
              border: `1px solid ${C.border}`, color: C.muted, padding: '8px 20px',
              borderRadius: 5, cursor: 'pointer', fontFamily: 'monospace',
              fontSize: 11, letterSpacing: 2, textTransform: 'uppercase' }}>
            ↺ Reset
          </button>
        </>
      )}
    </div>
  );
}
