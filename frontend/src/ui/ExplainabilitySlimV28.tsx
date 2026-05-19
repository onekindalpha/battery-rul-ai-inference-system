
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
  if (!res.ok) throw new Error(`${res.status}: ${text.slice(0, 240)}`)
  return JSON.parse(text)
}

function unwrap(raw: any) {
  const item = raw?.payload ?? raw?.item ?? raw?.data ?? raw?.precomputed ?? raw
  return {
    cycles: (item?.query?.cycle ?? item?.cycles ?? []).map(n),
    pred: (item?.pred?.mean ?? item?.predRUL ?? item?.rul_pred ?? []).map(n),
    std: (item?.pred?.std ?? item?.std ?? item?.rul_std ?? []).map(n),
  }
}

export function ExplainabilitySlimV28({
  battery,
  rRatio,
  cycle,
}: {
  battery: string
  rRatio: string | number
  cycle: number
}) {
  const [pack, setPack] = React.useState<any>(null)
  const [shap, setShap] = React.useState<any>(null)
  const [warning, setWarning] = React.useState<string | null>(null)

  React.useEffect(() => {
    let alive = true
    async function run() {
      setWarning(null)
      try {
        const [preRaw, shapRaw] = await Promise.all([
          getJSON(`/api/battery/${battery}/precomputed?r_ratio=${encodeURIComponent(String(rRatio))}`),
          getJSON(`/api/fixed4/shap-current`).catch(() => ({ ok: false, items: [] })),
        ])
        if (!alive) return
        setPack(unwrap(preRaw))
        setShap(shapRaw)
      } catch (e: any) {
        if (alive) setWarning(String(e?.message || e))
      }
    }
    if (battery) run()
    return () => { alive = false }
  }, [battery, rRatio])

  const idx = pack?.cycles ? nearestIndex(pack.cycles, Number(cycle)) : -1
  const pred = idx >= 0 ? n(pack.pred?.[idx]) : NaN
  const sigma = idx >= 0 ? n(pack.std?.[idx]) : NaN
  const uncertainty2Sigma = Number.isFinite(sigma) ? 2 * sigma : NaN
  const confidence = Number.isFinite(uncertainty2Sigma) && Math.abs(pred) > 1
    ? Math.max(0, Math.min(100, 100 * (1 - Math.min(1, uncertainty2Sigma / Math.abs(pred)))))
    : NaN

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
      <div style={{ fontSize: 24, fontWeight: 900 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#777', marginTop: 4 }}>{sub}</div>}
    </div>
  )

  return (
    <div style={{ overflowY: 'auto', paddingRight: 4 }}>
      <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 15, fontWeight: 800 }}>🧠 Explainability</h3>
      <p style={{ fontSize: 12, color: '#666', lineHeight: 1.55, marginTop: 0 }}>
        This tab explains uncertainty and global model-level feature importance. Forecast/error metrics stay in Overview.
      </p>

      {warning && (
        <div style={{ padding: 10, border: '1px solid #fbbf24', background: '#fffbeb', borderRadius: 8, marginBottom: 12, color: '#92400e' }}>
          Explainability data warning: {warning}
        </div>
      )}

      <div style={{ marginBottom: 16, padding: 12, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Prediction Uncertainty</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
          {card('Uncertainty', Number.isFinite(uncertainty2Sigma) ? `±${uncertainty2Sigma.toFixed(1)}` : '예측 전', '2σ interval, cycles')}
          {card('Confidence', Number.isFinite(confidence) ? `${confidence.toFixed(0)}%` : '0%', 'derived from uncertainty / predicted RUL')}
        </div>
        <div style={{ marginTop: 10, padding: 10, background: 'white', border: '1px solid #e5e7eb', borderRadius: 6, fontSize: 12, lineHeight: 1.55 }}>
          <strong>Note:</strong> Predicted RUL, True RUL, RMSE, MAE, and current absolute error are shown in Overview and are not duplicated here.
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
          <div style={{ padding: 12, color: '#777', background: '#f8fafc', borderRadius: 6 }}>
            SHAP global importance is not available from the backend endpoint.
          </div>
        )}
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Model Architecture</h4>
        <div style={{ fontSize: 12, lineHeight: 1.65, color: '#444' }}>
          <p style={{ marginTop: 0 }}><strong>Backbone:</strong> CEEMDAN–Transformer–DNN decomposes noisy capacity trajectories and models local regeneration components plus global residual degradation trend.</p>
          <p style={{ marginBottom: 0 }}><strong>Meta-learning:</strong> BMAML-SVGD adapts the RUL model with uncertainty-aware particles for few-shot battery conditions.</p>
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Anomaly/health report placement</h4>
        <p style={{ fontSize: 12, color: '#555', lineHeight: 1.55, margin: 0 }}>
          Early-warning and driver ranking should be summarized from the Degradation anomaly report, not duplicated as SHAP. SHAP here remains global model-level feature importance.
        </p>
      </div>
    </div>
  )
}
