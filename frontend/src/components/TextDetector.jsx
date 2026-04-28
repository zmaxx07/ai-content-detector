// src/components/TextDetector.jsx
import React, { useState, useContext } from 'react';
import { detectText } from '../services/api';
import { HistoryContext } from '../App';
import {
  C, Card, SectionLabel, ScoreMeter, VerdictBadge,
  SignalRow, Spinner, ErrorBox, RawScores, DatasetBadge
} from './ui';

const TOPIC_PRESETS = ['technology', 'science', 'health', 'history', 'AI', 'finance'];

export default function TextDetector() {
  const { onResult } = useContext(HistoryContext);
  const [text, setText] = useState('');
  const [topic, setTopic] = useState('technology');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = async () => {
    setError(null); setResult(null); setLoading(true);
    try {
      const data = await detectText(text, topic, true);
      setResult(data);
      onResult('text', data.verdict, data.confidence);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const canRun = text.trim().length >= 40 && !loading;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Input */}
      <Card>
        <SectionLabel>Topic for Live Human References</SectionLabel>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
          <input value={topic} onChange={e => setTopic(e.target.value)}
            placeholder="e.g. technology"
            style={{ flex: 1, minWidth: 140, background: '#06060e', border: `1px solid ${C.border}`,
              color: '#ddd', padding: '8px 12px', borderRadius: 6, fontFamily: 'monospace', fontSize: 12,
              outline: 'none' }} />
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {TOPIC_PRESETS.map(t => (
              <button key={t} onClick={() => setTopic(t)} style={{
                padding: '5px 10px', borderRadius: 12, cursor: 'pointer', fontSize: 11,
                fontFamily: 'monospace', transition: 'all .15s',
                background: topic === t ? `${C.neutral}20` : '#06060e',
                border: `1px solid ${topic === t ? C.neutral : C.border}`,
                color: topic === t ? C.neutral : C.muted,
              }}>{t}</button>
            ))}
          </div>
        </div>

        <SectionLabel>Text to Analyze (min 40 chars)</SectionLabel>
        <textarea value={text} onChange={e => setText(e.target.value)}
          placeholder="Paste any text — essay, article, email, social post, assignment..."
          style={{ width: '100%', minHeight: 200, background: '#030308',
            border: `1px solid ${C.border}`, borderRadius: 8, color: '#d4d4e8',
            fontFamily: "'Outfit', sans-serif", fontSize: 14, lineHeight: 1.9,
            padding: '14px 16px', resize: 'vertical', outline: 'none',
            transition: 'border-color .2s' }}
          onFocus={e => e.target.style.borderColor = C.neutral}
          onBlur={e => e.target.style.borderColor = C.border} />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, marginBottom: 14 }}>
          <span style={{ fontSize: 11, fontFamily: 'monospace',
            color: text.length >= 40 ? C.human : C.ai }}>{text.length} chars · {wordCount} words</span>
          <span style={{ fontSize: 11, fontFamily: 'monospace', color: C.muted }}>
            Sources: Wikipedia · DEV.to · Quotable · NewsAPI
          </span>
        </div>

        <button onClick={run} disabled={!canRun} style={{
          width: '100%', padding: '14px', background: canRun
            ? `linear-gradient(135deg, ${C.neutral}, ${C.ai})`
            : '#1a1a2e',
          border: 'none', borderRadius: 7, color: canRun ? '#fff' : C.muted,
          fontFamily: 'monospace', fontSize: 12, letterSpacing: 3,
          textTransform: 'uppercase', cursor: canRun ? 'pointer' : 'not-allowed',
          boxShadow: canRun ? `0 4px 20px rgba(99,102,241,.3)` : 'none',
          transition: 'all .2s' }}>
          {loading ? 'Running Pipeline...' : '▶  Run Full Detection'}
        </button>

        {error && <ErrorBox message={error} />}
      </Card>

      {loading && <Card><Spinner /></Card>}

      {/* Results */}
      {result && !loading && (
        <>
          <Card>
            <VerdictBadge verdict={result.verdict} confidence={result.confidence} />
            <ScoreMeter aiScore={result.breakdown?.ai_score} humanScore={result.breakdown?.human_score} />
            {result.summary && (
              <p style={{ color: '#999', fontSize: 13, lineHeight: 1.85, padding: '12px 14px',
                background: '#05050d', borderRadius: 7, borderLeft: `3px solid ${C.neutral}` }}>
                {result.summary}
              </p>
            )}
            {result.key_evidence && (
              <div style={{ marginTop: 12, padding: '10px 14px',
                background: `${C.ai}08`, border: `1px solid ${C.ai}20`, borderRadius: 6 }}>
                <SectionLabel>Key Evidence</SectionLabel>
                <div style={{ fontSize: 13, color: '#ccc' }}>{result.key_evidence}</div>
              </div>
            )}
          </Card>

          {/* Linguistic features */}
          {result.linguistic_features && (
            <Card>
              <SectionLabel>📊 Linguistic Feature Analysis</SectionLabel>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(110px,1fr))', gap: 8, marginBottom: 14 }}>
                {[
                  ['Words', result.linguistic_features.word_count, C.neutral],
                  ['Unique', result.linguistic_features.unique_words, C.neutral],
                  ['Lex Diversity', `${result.linguistic_features.lexical_diversity}%`,
                    result.linguistic_features.lexical_diversity > 60 ? C.human : C.ai],
                  ['Avg Sent Len', result.linguistic_features.avg_sentence_length,
                    result.linguistic_features.avg_sentence_length > 22 ? C.ai : C.human],
                  ['Sent Variance', result.linguistic_features.sentence_length_variance,
                    result.linguistic_features.sentence_length_variance > 15 ? C.human : C.ai],
                  ['Human Signals', result.linguistic_features.human_informal_signals,
                    result.linguistic_features.human_informal_signals > 3 ? C.human : C.muted],
                ].map(([label, val, color]) => (
                  <div key={label} style={{ padding: '10px 12px', background: '#06060e',
                    borderRadius: 6, border: `1px solid ${C.border}`, textAlign: 'center' }}>
                    <div style={{ fontSize: 17, fontWeight: 700, color, fontFamily: 'monospace' }}>{val}</div>
                    <div style={{ fontSize: 10, color: C.muted, marginTop: 3,
                      fontFamily: 'monospace', letterSpacing: 1, textTransform: 'uppercase' }}>{label}</div>
                  </div>
                ))}
              </div>
              {result.linguistic_features.ai_phrases_found?.length > 0 && (
                <div>
                  <div style={{ fontSize: 11, fontFamily: 'monospace', color: C.ai,
                    marginBottom: 6, letterSpacing: 1, textTransform: 'uppercase' }}>
                    ⚠ AI Phrase Markers Found
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {result.linguistic_features.ai_phrases_found.map(p => (
                      <span key={p} style={{ padding: '2px 9px', borderRadius: 12,
                        background: `${C.ai}15`, border: `1px solid ${C.ai}30`,
                        color: C.ai, fontFamily: 'monospace', fontSize: 11 }}>{p}</span>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          )}

          {/* Signals */}
          {result.signals?.length > 0 && (
            <Card>
              <SectionLabel>🔍 Detection Signals</SectionLabel>
              {result.signals.map((s, i) => (
                <SignalRow key={i} signal={s} last={i === result.signals.length - 1} />
              ))}
            </Card>
          )}

          {/* Raw model scores */}
          {result.breakdown?.raw_scores?.length > 0 && (
            <Card>
              <SectionLabel>🤗 RoBERTa Raw Model Output</SectionLabel>
              <RawScores scores={result.breakdown.raw_scores} />
            </Card>
          )}

          {/* Human references */}
          {result.human_references?.length > 0 && (
            <Card>
              <SectionLabel>🌐 Live Human Reference Samples</SectionLabel>
              {result.comparison_note && (
                <div style={{ padding: '10px 14px', background: `${C.human}08`,
                  border: `1px solid ${C.human}15`, borderRadius: 6, marginBottom: 12,
                  fontSize: 13, color: '#aaa', lineHeight: 1.7 }}>
                  <span style={{ color: C.human, fontFamily: 'monospace', fontSize: 10,
                    letterSpacing: 1, textTransform: 'uppercase' }}>Comparison: </span>
                  {result.comparison_note}
                </div>
              )}
              {result.human_references.slice(0, 4).map((s, i) => (
                <div key={i} style={{ padding: '12px 14px', background: '#06060e',
                  borderRadius: 7, border: `1px solid ${C.border}`, marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                    <span style={{ color: C.human, fontFamily: 'monospace', fontSize: 12 }}>
                      {s.icon} {s.source}
                    </span>
                    {s.similarity !== undefined && (
                      <span style={{ fontSize: 11, fontFamily: 'monospace',
                        color: s.similarity > 30 ? C.human : C.muted }}>
                        {s.similarity}% vocab match
                      </span>
                    )}
                  </div>
                  {s.title && <div style={{ fontSize: 12, color: '#aaa', fontStyle: 'italic', marginBottom: 4 }}>{s.title}</div>}
                  <div style={{ fontSize: 12, color: '#666', lineHeight: 1.7 }}>
                    "{s.text.slice(0, 180)}{s.text.length > 180 ? '...' : ''}"
                  </div>
                  {s.url && s.url !== '#' && (
                    <a href={s.url} target="_blank" rel="noreferrer"
                      style={{ fontSize: 10, color: C.muted, fontFamily: 'monospace', marginTop: 4, display: 'block' }}>
                      ↗ view source
                    </a>
                  )}
                </div>
              ))}
            </Card>
          )}

          <DatasetBadge dataset={result.dataset_info} />

          <button onClick={() => { setResult(null); setText(''); }}
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
