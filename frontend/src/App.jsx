// src/App.jsx  — root component
import React, { useState, useEffect, useCallback } from 'react';
import TextDetector  from './components/TextDetector';
import ImageDetector from './components/ImageDetector';
import CodeDetector  from './components/CodeDetector';
import History       from './components/History';
import { checkHealth } from './services/api';
import { C } from './components/ui';

const TABS = [
  { id: 'text',  label: 'Text',  icon: '📝', desc: 'Essays, articles, emails, assignments' },
  { id: 'image', label: 'Image', icon: '🖼', desc: 'Photos, artwork, generated images' },
  { id: 'code',  label: 'Code',  icon: '💻', desc: 'Python, JS, Java, any language' },
];

export default function App() {
  const [tab, setTab] = useState('text');
  const [history, setHistory] = useState([]);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'ok' | 'error' | 'checking'
  const [healthInfo, setHealthInfo] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then(data => { setBackendStatus('ok'); setHealthInfo(data); })
      .catch(() => setBackendStatus('error'));
  }, []);

  const addHistory = useCallback((type, verdict, confidence) => {
    setHistory(h => [{
      type, verdict, confidence,
      time: new Date().toLocaleTimeString()
    }, ...h.slice(0, 11)]);
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: C.bg,
      backgroundImage: `radial-gradient(ellipse 80% 45% at 50% -10%,rgba(99,102,241,.1) 0%,transparent 65%),
                        radial-gradient(ellipse 40% 30% at 95% 90%,rgba(16,185,129,.05) 0%,transparent 60%)` }}>

      {/* ── HEADER ── */}
      <header style={{ textAlign: 'center', padding: '44px 20px 28px' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, marginBottom: 18,
          padding: '4px 14px', borderRadius: 20,
          border: '1px solid rgba(99,102,241,.3)', background: 'rgba(99,102,241,.07)' }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: C.neutral,
            display: 'inline-block', animation: 'pulse 2s infinite' }} />
          <span style={{ fontFamily: 'monospace', fontSize: 10, color: C.neutral,
            letterSpacing: 3, textTransform: 'uppercase' }}>
             AI Detection System 
          </span>
        </div>

        <h1 style={{ fontSize: 'clamp(1.8rem,4.5vw,3.6rem)', fontWeight: 800,
          letterSpacing: -1.5, lineHeight: 1.05, marginBottom: 10 }}>
          AI Content&nbsp;
          <span style={{ color: C.ai }}>Detector</span>
        </h1>

        <p style={{ color: C.muted, maxWidth: 540, margin: '0 auto 20px',
          lineHeight: 1.8, fontSize: 15 }}>
          Detects AI-generated <strong style={{ color: '#a5b4fc' }}>text</strong>,{' '}
          <strong style={{ color: '#6ee7b7' }}>images</strong>, and{' '}
          <strong style={{ color: '#fda4af' }}>code</strong> using real ML models
          + live human reference data from Wikipedia, DEV.to &amp; NewsAPI.
        </p>

        {/* Tech badges */}
        <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: 7 }}>
          {[
            ['🤗', 'RoBERTa ML Model', '#f43f5e'],
            ['📖', 'Wikipedia API', '#6366f1'],
            ['💻', 'DEV.to API', '#10b981'],
            ['📰', 'NewsAPI', '#f59e0b'],
            ['⚡', 'FastAPI Backend', '#8b5cf6'],
            ['⚛️', 'React Frontend', '#38bdf8'],
          ].map(([icon, label, color]) => (
            <span key={label} style={{ padding: '4px 11px', borderRadius: 20,
              background: `${color}10`, border: `1px solid ${color}25`,
              fontSize: 11, color, fontFamily: 'monospace' }}>
              {icon} {label}
            </span>
          ))}
        </div>

        {/* Backend status */}
        <div style={{ marginTop: 16 }}>
          {backendStatus === 'checking' && (
            <span style={{ fontSize: 11, fontFamily: 'monospace', color: C.muted }}>
              ○ Connecting to backend...
            </span>
          )}
          {backendStatus === 'ok' && (
            <span style={{ fontSize: 11, fontFamily: 'monospace', color: C.human }}>
              ● Backend online · {healthInfo?.inference_mode?.toUpperCase()} mode · {healthInfo?.models_loaded?.text_model}
            </span>
          )}
          {backendStatus === 'error' && (
            <span style={{ fontSize: 11, fontFamily: 'monospace', color: C.ai }}>
              ✗ Backend offline — start it with: <code style={{ background: '#111', padding: '1px 6px', borderRadius: 3 }}>python run.py</code>
            </span>
          )}
        </div>
      </header>

      {/* ── MAIN ── */}
      <main style={{ maxWidth: 860, margin: '0 auto', padding: '0 16px 80px',
        display: 'flex', flexDirection: 'column', gap: 14 }}>

        {/* Tab bar */}
        <div style={{ background: 'rgba(0,0,0,.4)', border: `1px solid ${C.border}`,
          borderRadius: 12, overflow: 'hidden', display: 'flex' }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              flex: 1, padding: '16px 8px', border: 'none', cursor: 'pointer',
              background: tab === t.id ? 'rgba(99,102,241,.08)' : 'transparent',
              borderBottom: tab === t.id ? `2px solid ${C.neutral}` : '2px solid transparent',
              color: tab === t.id ? '#fff' : C.muted,
              fontFamily: 'monospace', fontSize: 11, letterSpacing: 2,
              textTransform: 'uppercase', transition: 'all .18s',
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
              <span style={{ fontSize: 18 }}>{t.icon}</span>
              <span>{t.label}</span>
              <span style={{ fontSize: 9, color: tab === t.id ? '#888' : '#333',
                letterSpacing: .5, textTransform: 'none', fontFamily: 'sans-serif' }}>
                {t.desc}
              </span>
            </button>
          ))}
        </div>

        {/* Tab content — wrapped to capture results for history */}
        <ResultCapture tab={tab} onResult={addHistory}>
          {tab === 'text'  && <TextDetector  />}
          {tab === 'image' && <ImageDetector />}
          {tab === 'code'  && <CodeDetector  />}
        </ResultCapture>

        {/* History */}
        <History items={history} />

        {/* Architecture panel */}
        <div style={{ background: 'rgba(255,255,255,0.015)', border: `1px solid ${C.border}`,
          borderRadius: 10, padding: '20px 22px' }}>
          <div style={{ fontFamily: 'monospace', fontSize: 10, letterSpacing: 3,
            color: C.muted, textTransform: 'uppercase', marginBottom: 14 }}>⚙ System Architecture</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 10 }}>
            {[
              { n: '01', t: 'Live Data Fetch', d: 'Wikipedia, DEV.to, Quotable, NewsAPI pulled in real-time as human ground-truth', c: '#6366f1' },
              { n: '02', t: 'ML Inference', d: 'RoBERTa fine-tuned on 500k docs via HuggingFace. CIFAKE-trained image classifier', c: '#10b981' },
              { n: '03', t: 'Linguistic Analysis', d: '15+ features: AI phrases, sentence variance, lexical diversity, human markers', c: '#f59e0b' },
              { n: '04', t: 'Score Blending', d: 'ML score ± linguistic adjustment (−20 to +20%). Jaccard vocab similarity', c: '#8b5cf6' },
              { n: '05', t: 'FastAPI Backend', d: 'Python backend with async endpoints, Pydantic validation, full REST API', c: '#f43f5e' },
              { n: '06', t: 'React Frontend', d: 'Calls backend API, renders results, history log, live reference display', c: '#38bdf8' },
            ].map(({ n, t, d, c }) => (
              <div key={n} style={{ padding: '13px', background: '#05050d',
                borderRadius: 8, border: `1px solid ${c}15` }}>
                <div style={{ fontFamily: 'monospace', fontSize: 20, color: `${c}40`,
                  fontWeight: 700, marginBottom: 5 }}>{n}</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#ccc', marginBottom: 4 }}>{t}</div>
                <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.65 }}>{d}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Datasets panel */}
        <div style={{ background: 'rgba(255,255,255,0.015)', border: `1px solid ${C.border}`,
          borderRadius: 10, padding: '20px 22px' }}>
          <div style={{ fontFamily: 'monospace', fontSize: 10, letterSpacing: 3,
            color: C.muted, textTransform: 'uppercase', marginBottom: 14 }}>📚 Training Datasets</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 10 }}>
            {[
              { name: 'GPT-2 Output Dataset', size: '500,000 docs', src: 'OpenAI', models: 'GPT-2, GPT-3, ChatGPT' },
              { name: 'AI Text Detection Pile', size: '130,000 docs', src: 'Hugging Face', models: 'ChatGPT, GPT-J, GPT-NeoX' },
              { name: 'CIFAKE + GenImage', size: '1.2M images', src: 'Kaggle/Research', models: 'SD, DALL-E, Midjourney, FLUX' },
              { name: 'CodeSearchNet + AI-CodeBench', size: '2M snippets', src: 'GitHub', models: 'Copilot, CodeLlama, StarCoder' },
            ].map(ds => (
              <div key={ds.name} style={{ padding: '12px 14px', background: '#08080f',
                borderRadius: 7, border: `1px solid ${C.border}` }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#bbb', marginBottom: 3 }}>{ds.name}</div>
                <div style={{ fontSize: 11, color: '#444', fontFamily: 'monospace', marginBottom: 5 }}>{ds.size} · {ds.src}</div>
                <div style={{ fontSize: 11, color: '#333', fontFamily: 'monospace' }}>{ds.models}</div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

// Wraps a detector and intercepts results for history logging
function ResultCapture({ children, tab, onResult }) {
  // Clone children and inject an onResult prop — in practice we use
  // context, but for simplicity each detector calls onResult directly.
  // Instead, we simply render children; history is added inside each detector.
  // The history state lives in App so we pass a callback down via context.
  return <HistoryContext.Provider value={{ tab, onResult }}>{children}</HistoryContext.Provider>;
}

export const HistoryContext = React.createContext({ tab: 'text', onResult: () => {} });
