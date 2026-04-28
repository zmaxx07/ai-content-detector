// src/components/ImageDetector.jsx
import React, { useState, useRef, useCallback, useContext } from 'react';
import { detectImage } from '../services/api';
import { HistoryContext } from '../App';
import {
  C, Card, SectionLabel, ScoreMeter, VerdictBadge,
  SignalRow, Spinner, ErrorBox, RawScores, DatasetBadge
} from './ui';

export default function ImageDetector() {
  const { onResult } = useContext(HistoryContext);
  const [imgFile, setImgFile] = useState(null);
  const [imgPreview, setImgPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef();

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    setImgFile(file);
    setResult(null);
    const reader = new FileReader();
    reader.onload = e => setImgPreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const run = async () => {
    setError(null); setResult(null); setLoading(true);
    try {
      const data = await detectImage(imgFile);
      setResult(data);
      onResult('image', data.verdict, data.confidence);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const remove = () => { setImgFile(null); setImgPreview(null); setResult(null); };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <Card>
        <SectionLabel>Model: Organika/sdxl-detector · Dataset: CIFAKE + GenImage (1.2M images)</SectionLabel>
        <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])} />

        {!imgPreview ? (
          <div onDrop={handleDrop} onDragOver={e => e.preventDefault()}
            onClick={() => fileRef.current.click()}
            onMouseEnter={e => e.currentTarget.style.borderColor = C.ai}
            onMouseLeave={e => e.currentTarget.style.borderColor = '#2a2a4a'}
            style={{ border: '1.5px dashed #2a2a4a', borderRadius: 10, padding: '60px 20px',
              textAlign: 'center', cursor: 'pointer', transition: 'border-color .2s',
              background: `${C.ai}02` }}>
            <div style={{ fontSize: 40, marginBottom: 14 }}>⬆</div>
            <div style={{ color: '#ccc', marginBottom: 6 }}>Drop image or click to upload</div>
            <div style={{ fontSize: 11, color: C.muted, fontFamily: 'monospace' }}>
              JPG · PNG · WEBP · GIF · Max 10MB
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center' }}>
            <img src={imgPreview} alt="Uploaded"
              style={{ maxHeight: 300, maxWidth: '100%', borderRadius: 8,
                border: `1px solid ${C.border}`, objectFit: 'contain' }} />
            <button onClick={remove}
              style={{ display: 'block', margin: '10px auto 0', background: 'none',
                border: `1px solid ${C.border}`, color: C.muted, padding: '5px 14px',
                borderRadius: 4, cursor: 'pointer', fontFamily: 'monospace', fontSize: 11 }}>
              ✕ Remove
            </button>
          </div>
        )}

        <button onClick={run} disabled={!imgFile || loading} style={{
          width: '100%', marginTop: 16, padding: '14px',
          background: imgFile && !loading ? `linear-gradient(135deg, #6366f1, ${C.ai})` : '#1a1a2e',
          border: 'none', borderRadius: 7, color: imgFile && !loading ? '#fff' : C.muted,
          fontFamily: 'monospace', fontSize: 12, letterSpacing: 3, textTransform: 'uppercase',
          cursor: imgFile && !loading ? 'pointer' : 'not-allowed', transition: 'all .2s' }}>
          {loading ? 'Scanning Image...' : '▶  Analyze Image'}
        </button>
        {error && <ErrorBox message={error} />}
      </Card>

      {loading && <Card><Spinner /></Card>}

      {result && !loading && (
        <>
          <Card>
            <VerdictBadge verdict={result.verdict} confidence={result.confidence} />
            <ScoreMeter aiScore={result.breakdown?.ai_score} humanScore={result.breakdown?.human_score} />
            {result.summary && (
              <p style={{ color: '#999', fontSize: 13, lineHeight: 1.85, padding: '12px 14px',
                background: '#05050d', borderRadius: 7, borderLeft: `3px solid ${C.ai}` }}>
                {result.summary}
              </p>
            )}
          </Card>

          {/* Image features */}
          {result.image_features && (
            <Card>
              <SectionLabel>📐 Image Feature Analysis</SectionLabel>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(120px,1fr))', gap: 8 }}>
                {[
                  ['Resolution', result.image_features.width ? `${result.image_features.width}×${result.image_features.height}` : '?', C.neutral],
                  ['Aspect Ratio', result.image_features.aspect_ratio || '?', C.neutral],
                  ['File Size', `${result.image_features.file_size_kb}KB`, C.neutral],
                  ['AI Resolution', result.image_features.is_common_ai_resolution ? 'YES ⚠' : 'No ✓',
                    result.image_features.is_common_ai_resolution ? C.ai : C.human],
                ].map(([label, val, color]) => (
                  <div key={label} style={{ padding: '10px 12px', background: '#06060e',
                    borderRadius: 6, border: `1px solid ${C.border}`, textAlign: 'center' }}>
                    <div style={{ fontSize: 14, fontWeight: 700, color, fontFamily: 'monospace' }}>{val}</div>
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

          <button onClick={remove}
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
