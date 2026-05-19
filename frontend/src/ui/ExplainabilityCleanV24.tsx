
import React from 'react'
import Plot from 'react-plotly.js'

function n(v: any): number {
  const x = Number(v)
  return Number.isFinite(x) ? x : NaN
}

function nearestIndex(xs: number[], x: number): number {
  if (!xs.length) return -1
  let best = 0
  for (let i = 1; i < xs.length; i += 1) {
    if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i
  }
  return best
}

async function getJSON(url: string) {
  const res = await fetch(url)
  const text = await res.text()
  if (!res.ok) throw new Error(`${res.status}: ${text.slice(0, 220)}`)
  return JSON.parse(text)
}

function unwrapPrecomputed(raw: any) {
  const item = raw?.payload ?? raw?.item ?? raw?.data ?? raw?.precomputed ?? raw
  return {
    item,
    cycles: (item?.query?.cycle ?? item?.cycles ?? []).map(n),
    pred: (item?.pred?.mean ?? item?.predRUL ?? item?.rul_pred ?? []).map(n),
    truth: (item?.query?.true_rul ?? item?.trueRUL ?? item?.rul_true ?? []).map(n),
    std: (item?.pred?.std ?? item?.std ?? item?.rul_std ?? []).map(n),
    metrics: item?.metrics ?? { rmse: item?.rmse, mae: item?.mae },
    qPos: Number.isFinite(Number(item?.q_pos)) ? Number(item.q_pos) : null,
  }
}

function metric(a: number[], b: number[], kind: 'rmse' | 'mae') {
  const pairs = a.map((x, i) => [x, b[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
  if (!pairs.length) return NaN
  if (kind === 'mae') return pairs.reduce((s, [x, y]) => s + Math.abs(x - y), 0) / pairs.length
  return Math.sqrt(pairs.reduce((s, [x, y]) => s + Math.pow(x - y, 2), 0) / pairs.length)
}

export function ExplainabilityCleanV24({ battery, rRatio, cycle }: { battery: string; rRatio: string | number; cycle: number }) {
  const [pre, setPre] = React.useState<any>(null)
  const [shap, setShap] = React.useState<any>(null)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    let alive = true
    async function run() {
      setError(null)
      try {
        const [pRaw, s] = await Promise.all([
          getJSON(`/api/battery/${battery}/precomputed?r_ratio=${encodeURIComponent(String(rRatio))}`),
          getJSON(`/api/fixed4/shap-v24`).catch(() => null),
        ])
        if (!alive) return
        setPre(unwrapPrecomputed(pRaw))
        setShap(s)
      } catch (e: any) {
        if (alive) setError(String(e?.message || e))
      }
    }
    if (battery) run()
    return () => { alive = false }
  }, [battery, rRatio])

  const idx = pre?.cycles ? nearestIndex(pre.cycles, Number(cycle)) : -1
  const pred = idx >= 0 ? n(pre.pred?.[idx]) : NaN
  const truth = idx >= 0 ? n(pre.truth?.[idx]) : NaN
  const sigma = idx >= 0 ? n(pre.std?.[idx]) : NaN
  const absErr = Number.isFinite(pred) && Number.isFinite(truth) ? Math.abs(pred - truth) : NaN
  const signedErr = Number.isFinite(pred) && Number.isFinite(truth) ? pred - truth : NaN
  const u2 = Number.isFinite(sigma) ? 2 * sigma : NaN
  const confidence = Number.isFinite(u2) && Math.abs(pred) > 1
    ? Math.max(0, Math.min(100, 100 * (1 - Math.min(1, u2 / Math.abs(pred)))))
    : NaN

  const qPos = pre?.qPos ?? idx
  const segTruth = pre?.truth?.slice(0, qPos + 1) ?? []
  const segPred = pre?.pred?.slice(0, qPos + 1) ?? []
  const rmseRaw = n(pre?.metrics?.rmse)
  const maeRaw = n(pre?.metrics?.mae)
  const rmse = Number.isFinite(rmseRaw) ? rmseRaw : metric(pre?.truth ?? [], pre?.pred ?? [], 'rmse')
  const mae = Number.isFinite(maeRaw) ? maeRaw : metric(pre?.truth ?? [], pre?.pred ?? [], 'mae')
  const segRmse = metric(segTruth, segPred, 'rmse')
  const segMae = metric(segTruth, segPred, 'mae')

  const shapItems = (shap?.items || []).slice(0, 15).slice().reverse()
  const shapPlot = shapItems.length ? [{
    type: 'bar',
    orientation: 'h',
    x: shapItems.map((x: any) => x.importance),
    y: shapItems.map((x: any) => x.feature),
    name: 'Global importance',
  }] : []

  const card = (title: string, value: string, sub = '') => (
    <div style={{ padding: 12, background: 'white', border: '1px solid #e5e7eb', borderRadius: 8 }}>
      <div style={{ fontSize: 11, color: '#666', fontWeight: 800, marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 22, fontWeight: 900 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#777', marginTop: 4 }}>{sub}</div>}
    </div>
  )

  return (
    <div style={{ overflowY: 'auto', paddingRight: 4 }}>
      <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 15, fontWeight: 800 }}>🧠 Explainability</h3>
      <p style={{ fontSize: 12, color: '#666', lineHeight: 1.55, marginTop: 0 }}>
        Prediction confidence, architecture, and global model-level SHAP importance. No re-inference action is triggered from this tab.
      </p>

      {error && (
        <div style={{ padding: 10, border: '1px solid #fbbf24', background: '#fffbeb', borderRadius: 8, marginBottom: 12, color: '#92400e' }}>
          Explainability data warning: {error}
        </div>
      )}

      <div style={{ marginBottom: 16, padding: 12, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Prediction Confidence</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8 }}>
          {card('Predicted RUL', Number.isFinite(pred) ? pred.toFixed(1) : '0%', 'cycles')}
          {card('True RUL', Number.isFinite(truth) ? truth.toFixed(1) : '평가 전', 'cycles')}
          {card('Uncertainty', Number.isFinite(u2) ? `±${u2.toFixed(1)}` : '0%', '2σ')}
          {card('Confidence', Number.isFinite(confidence) ? `${confidence.toFixed(0)}%` : '0%')}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8, marginTop: 8 }}>
          {card('Current Abs Error', Number.isFinite(absErr) ? absErr.toFixed(2) : '평가 전', 'cycles')}
          {card('Overall RMSE', Number.isFinite(rmse) ? rmse.toFixed(2) : '평가 전')}
          {card('Overall MAE', Number.isFinite(mae) ? mae.toFixed(2) : '평가 전')}
          {card('Segment RMSE / MAE', Number.isFinite(segRmse) && Number.isFinite(segMae) ? `${segRmse.toFixed(2)} / ${segMae.toFixed(2)}` : '평가 전')}
        </div>
        <div style={{ marginTop: 10, padding: 10, background: 'white', border: '1px solid #e5e7eb', borderRadius: 6, fontSize: 12, lineHeight: 1.55 }}>
          <strong>Interpretation:</strong>{' '}
          {Number.isFinite(signedErr)
            ? signedErr > 0
              ? `Over-estimation risk: prediction is +${signedErr.toFixed(2)} cycles above observed RUL.`
              : `Conservative under-estimation: prediction is ${signedErr.toFixed(2)} cycles below observed RUL.`
            : 'Prediction/observed RUL pair is not available at this cycle.'}
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Global Model Feature Importance</h4>
        <p style={{ fontSize: 12, color: '#555', lineHeight: 1.55 }}>
          This is global feature importance for the BMAML sequence model, not a cycle-local anomaly driver.
        </p>
        {shapItems.length ? (
          <Plot
            data={shapPlot}
            layout={{ height: 380, margin: { l: 130, r: 20, t: 20, b: 35 }, xaxis: { title: 'Importance' }, yaxis: { title: 'Feature' } }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: '100%' }}
          />
        ) : (
          <div style={{ padding: 12, color: '#777', background: '#f8fafc', borderRadius: 6 }}>SHAP global importance file is not available.</div>
        )}
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Model Architecture</h4>
        <div style={{ fontSize: 12, lineHeight: 1.65, color: '#444' }}>
          <p style={{ marginTop: 0 }}><strong>Backbone:</strong> CEEMDAN–Transformer–DNN decomposes noisy capacity trajectories and models local regeneration components plus global residual degradation trend.</p>
          <p><strong>Meta-learning:</strong> BMAML-SVGD adapts the RUL model with uncertainty-aware particles for few-shot battery conditions.</p>
          <p style={{ marginBottom: 0 }}><strong>Uncertainty:</strong> Predictive intervals come from model standard deviation; RMSE/MAE summarize observed prediction error.</p>
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Methods & References</h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 11, lineHeight: 1.5 }}>
          <div style={{ padding: 10, background: 'white', borderRadius: 6, border: '1px solid #eee' }}><strong>CEEMDAN–Transformer–DNN</strong><br />Decomposition-based sequence modeling for early Li-ion RUL prediction.</div>
          <div style={{ padding: 10, background: 'white', borderRadius: 6, border: '1px solid #eee' }}><strong>BMAML-SVGD</strong><br />Bayesian meta-learning with particle uncertainty for few-shot adaptation.</div>
        </div>
      </div>
    </div>
  )
}
