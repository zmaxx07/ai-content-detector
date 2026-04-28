// src/components/History.jsx
import React from 'react';
import { C, SectionLabel } from './ui';

export default function History({ items }) {
  if (!items.length) return null;
  return (
    <div style={{ background: 'rgba(255,255,255,0.015)', border: `1px solid ${C.border}`,
      borderRadius: 10, padding: '16px 20px' }}>
      <SectionLabel>📋 Detection History</SectionLabel>
      {items.map((h, i) => {
        const isAI = h.verdict?.toLowerCase().includes('ai');
        const color = isAI ? C.ai : C.human;
        return (
          <div key={i} style={{ display: 'flex', justifyContent: 'space-between',
            alignItems: 'center', padding: '7px 0',
            borderBottom: i < items.length - 1 ? `1px solid ${C.border}` : 'none' }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: 10, fontFamily: 'monospace', color: C.muted, width: 36,
                textTransform: 'uppercase' }}>{h.type}</span>
              <span style={{ padding: '2px 10px', borderRadius: 10, fontSize: 11,
                background: `${color}10`, border: `1px solid ${color}25`, color,
                fontFamily: 'monospace' }}>{h.verdict}</span>
            </div>
            <span style={{ fontSize: 11, fontFamily: 'monospace', color: C.muted }}>
              {h.confidence}% · {h.time}
            </span>
          </div>
        );
      })}
    </div>
  );
}
