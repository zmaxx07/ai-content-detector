// src/components/ui.jsx  — reusable design-system components

import React from 'react';

const C = {
  ai: '#f43f5e', human: '#10b981', neutral: '#6366f1',
  bg: '#040408', card: 'rgba(255,255,255,0.03)',
  border: '#0f0f1e', muted: '#4a4a6a', text: '#e2e2f0',
};
export { C };

export function SectionLabel({ children }) {
  return (
    <div style={{ fontFamily: 'monospace', fontSize: 10, letterSpacing: 3,
      color: C.muted, textTransform: 'uppercase', marginBottom: 10 }}>
      {children}
    </div>
  );
}

export function Pill({ label, color }) {
  return (
    <span style={{ padding: '2px 9px', borderRadius: 12, fontSize: 11,
      background: `${color}18`, border: `1px solid ${color}30`,
      color, fontFamily: 'monospace', letterSpacing: 0.5 }}>
      {label}
    </span>
  );
}

export function Card({ children, style = {} }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: '20px 22px', ...style }}>
      {children}
    </div>
  );
}

export function ScoreMeter({ aiScore, humanScore }) {
  return (
    <div style={{ margin: '16px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between',
        fontSize: 11, fontFamily: 'monospace', marginBottom: 6 }}>
        <span style={{ color: C.human }}>HUMAN {humanScore}%</span>
        <span style={{ color: C.ai }}>AI {aiScore}%</span>
      </div>
      <div style={{ height: 10, background: '#0a0a14', borderRadius: 5,
        overflow: 'hidden', border: `1px solid ${C.border}` }}>
        <div style={{ height: '100%', display: 'flex' }}>
          <div style={{ width: `${humanScore}%`, background: `linear-gradient(90deg,${C.human},#34d399)`,
            transition: 'width 1.2s ease' }} />
          <div style={{ width: `${aiScore}%`, background: `linear-gradient(90deg,#fb923c,${C.ai})`,
            transition: 'width 1.2s ease' }} />
        </div>
      </div>
    </div>
  );
}

export function VerdictBadge({ verdict, confidence }) {
  const isAI = verdict?.toLowerCase().includes('ai');
  const color = isAI ? C.ai : C.human;
  return (
    <div style={{ textAlign: 'center', padding: '20px 0 10px' }}>
      <div style={{ display: 'inline-block', padding: '13px 36px', borderRadius: 4,
        background: `${color}10`, border: `1.5px solid ${color}35`,
        color, fontFamily: 'monospace', fontSize: 13, letterSpacing: 4,
        textTransform: 'uppercase', boxShadow: `0 0 30px ${color}18` }}>
        {verdict}
      </div>
      <div style={{ marginTop: 10, fontFamily: 'monospace', fontSize: 12, color: C.muted }}>
        Confidence: <span style={{ color }}>{confidence}%</span>
      </div>
    </div>
  );
}

export function SignalRow({ signal, last }) {
  const col = signal.type === 'ai' ? C.ai : signal.type === 'human' ? C.human : C.muted;
  return (
    <div style={{ display: 'flex', gap: 10, padding: '9px 0',
      borderBottom: last ? 'none' : `1px solid ${C.border}` }}>
      <div style={{ width: 7, height: 7, borderRadius: '50%', background: col,
        marginTop: 5, flexShrink: 0, boxShadow: `0 0 6px ${col}` }} />
      <div>
        <div style={{ fontSize: 10, fontFamily: 'monospace', color: C.muted,
          letterSpacing: 1, textTransform: 'uppercase' }}>{signal.label}</div>
        <div style={{ fontSize: 13, color: '#bbb', marginTop: 2 }}>{signal.value}</div>
      </div>
    </div>
  );
}

export function Spinner() {
  return (
    <div style={{ textAlign: 'center', padding: '40px 0' }}>
      <div style={{ width: 56, height: 56, margin: '0 auto 16px', position: 'relative' }}>
        <div style={{ position: 'absolute', inset: 0, borderRadius: '50%',
          border: `2px solid ${C.ai}`, animation: 'spin 1s linear infinite' }} />
        <div style={{ position: 'absolute', inset: 8, borderRadius: '50%',
          border: `2px solid ${C.neutral}`, animation: 'spin 1.4s linear infinite reverse' }} />
      </div>
      <div style={{ color: C.ai, fontFamily: 'monospace', fontSize: 11,
        letterSpacing: 3, animation: 'pulse 1s infinite' }}>ANALYZING...</div>
    </div>
  );
}

export function ErrorBox({ message }) {
  return (
    <div style={{ marginTop: 14, padding: '12px 16px',
      background: 'rgba(244,63,94,.08)', border: '1px solid rgba(244,63,94,.25)',
      borderRadius: 6, color: C.ai, fontSize: 12, fontFamily: 'monospace' }}>
      ⚠ {message}
    </div>
  );
}

export function RawScores({ scores }) {
  if (!scores?.length) return null;
  return (
    <div style={{ marginTop: 14 }}>
      <SectionLabel>Raw Model Output</SectionLabel>
      {scores.map((s, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 7 }}>
          <span style={{ fontSize: 11, fontFamily: 'monospace', color: C.muted, width: 70 }}>{s.label}</span>
          <div style={{ flex: 1, height: 6, background: '#08080f', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{ width: `${Math.round(s.score * 100)}%`, height: '100%',
              background: C.ai, transition: 'width 1s' }} />
          </div>
          <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#aaa', width: 34, textAlign: 'right' }}>
            {Math.round(s.score * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export function DatasetBadge({ dataset }) {
  if (!dataset) return null;
  return (
    <div style={{ marginTop: 16, padding: '12px 14px', background: '#06060e',
      borderRadius: 8, border: `1px solid ${C.border}` }}>
      <SectionLabel>Training Dataset</SectionLabel>
      <div style={{ fontSize: 13, color: '#ccc', fontWeight: 600, marginBottom: 5 }}>{dataset.name}</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 6 }}>
        <Pill label={`📦 ${dataset.samples}`} color={C.neutral} />
        <Pill label={`📁 ${dataset.source}`} color='#f59e0b' />
      </div>
      <div style={{ fontSize: 11, color: C.muted, fontFamily: 'monospace' }}>
        {dataset.ai_models_covered?.join(' · ')}
      </div>
    </div>
  );
}
