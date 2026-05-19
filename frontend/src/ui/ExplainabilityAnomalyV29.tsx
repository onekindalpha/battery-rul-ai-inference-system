
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

function unwrapPrecomputed(raw: any) {
  const item = raw?.payload ?? raw?.item ?? raw?.data ?? raw?.precomputed ?? raw
  return {
    cycles: (item?.query?.cycle ?? item?.cycles ?? []).map(n),
    pred: (item?.pred?.mean ?? item?.predRUL ?? item?.rul_pred ?? []).map(n),
    truth: (item?.query?.true_rul ?? item?.trueRUL ?? item?.rul_true ?? []).map(n),
    std: (item?.pred?.std ?? item?.std ?? item?.rul_std ?? []).map(n),
    qPos: Number.isFinite(Number(item?.q_pos)) ? Number(item.q_pos) : null,
  }
}

function eolCycle(pre: any) {
  const cycles = pre?.cycles ?? []
  const truth = pre?.truth ?? []
  for (let i = 0; i < cycles.length; i += 1) {
    if (Number.isFinite(truth[i]) && truth[i] <= 0) return cycles[i]
  }
  return cycles.length ? cycles[cycles.length - 1] : NaN
}

function pickIssueCycle(anom: any): number {
  const candidates: number[] = []
  const add = (v: any) => {
    const x = Number(v)
    if (Number.isFinite(x)) candidates.push(x)
  }
  add(anom?.earlyWarning?.onsetCycle)
  add(anom?.early_warning?.onsetCycle)
  add(anom?.cap_onset_cycle)
  add(anom?.dcr_onset_cycle)
  ;[...(anom?.issues ?? []), ...(anom?.majorAlerts ?? []), ...(anom?.major_alerts ?? [])].forEach((x: any) => add(x?.onsetCycle ?? x?.onset_cycle))
  return candidates.length ? Math.min(...candidates) : NaN
}

function issueLabel(anom: any) {
  const issues = [...(anom?.majorAlerts ?? []), ...(anom?.issues ?? [])]
  if (anom?.earlyWarning?.active || anom?.early_warning?.active) {
    return anom?.earlyWarning?.message || anom?.early_warning?.message || "핵심 KPI의 큰 이탈 전, 일부 stress driver가 cohort 대비 outlier로 먼저 감지됨."
  }
  if (issues.length) return issues.map((x: any) => x.label || x.type).filter(Boolean).join(' / ')
  if (anom?.status && anom.status !== 'normal') return String(anom.status)
  return '현재 cycle 기준 뚜렷한 anomaly report 항목 없음'
}

function topDrivers(anom: any) {
  const drivers = Array.isArray(anom?.drivers) ? anom.drivers : []
  return drivers
    .map((d: any) => ({
      label: d.label || d.tag || d.feature || 'driver',
      feature: d.feature,
      z: Number(d.z ?? d.absZ ?? d.abs_z),
      absZ: Math.abs(Number(d.absZ ?? d.abs_z ?? d.z ?? 0)),
      action: d.recommendation || d.action || '',
    }))
    .filter((d: any) => Number.isFinite(d.absZ))
    .sort((a: any, b: any) => b.absZ - a.absZ)
    .slice(0, 3)
}

export function ExplainabilityAnomalyV29({ battery, rRatio, cycle }: { battery: string; rRatio: string | number; cycle: number }) {
  const [pre, setPre] = React.useState<any>(null)
  const [anom, setAnom] = React.useState<any>(null)
  const [shap, setShap] = React.useState<any>(null)
  const [warning, setWarning] = React.useState<string | null>(null)

  React.useEffect(() => {
    let alive = true
    async function run() {
      setWarning(null)
      try {
        const [preRaw, anomRaw, shapRaw] = await Promise.all([
          getJSON(`/api/battery/${battery}/precomputed?r_ratio=${encodeURIComponent(String(rRatio))}`),
          getJSON(`/api/battery/${battery}/degradation-monitoring?r_ratio=${encodeURIComponent(String(rRatio))}&cycle=${encodeURIComponent(String(cycle))}`).catch(() => null),
          getJSON(`/api/fixed4/shap-current`).catch(() => ({ ok: false, items: [] })),
        ])
        if (!alive) return
        setPre(unwrapPrecomputed(preRaw))
        setAnom(anomRaw)
        setShap(shapRaw)
      } catch (e: any) {
        if (alive) setWarning(String(e?.message || e))
      }
    }
    if (battery) run()
    return () => { alive = false }
  }, [battery, rRatio, cycle])

  const idx = pre?.cycles ? nearestIndex(pre.cycles, Number(cycle)) : -1
  const pred = idx >= 0 ? n(pre.pred?.[idx]) : NaN
  const sigma = idx >= 0 ? n(pre.std?.[idx]) : NaN
  const uncertainty2Sigma = Number.isFinite(sigma) ? 2 * sigma : NaN
  const confidence = Number.isFinite(uncertainty2Sigma) && Math.abs(pred) > 1
    ? Math.max(0, Math.min(100, 100 * (1 - Math.min(1, uncertainty2Sigma / Math.abs(pred)))))
    : NaN

  const eol = eolCycle(pre)
  const firstCycle = pickIssueCycle(anom)
  const lead = Number.isFinite(eol) && Number.isFinite(firstCycle) ? Math.max(0, eol - firstCycle) : NaN
  const drivers = topDrivers(anom)
  const shapItems = (shap?.items || []).slice(0, 12).slice().reverse()
  const shapPlot = shapItems.length ? [{ type: 'bar', orientation: 'h', x: shapItems.map((x: any) => x.importance), y: shapItems.map((x: any) => x.feature), name: 'Global importance' }] : []

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
        Uncertainty, anomaly-report summary, global SHAP importance, and model architecture.
      </p>

      {warning && <div style={{ padding: 10, border: '1px solid #fbbf24', background: '#fffbeb', borderRadius: 8, marginBottom: 12, color: '#92400e' }}>{warning}</div>}

      <div style={{ marginBottom: 16, padding: 12, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Prediction Uncertainty</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
          {card('Uncertainty', Number.isFinite(uncertainty2Sigma) ? `±${uncertainty2Sigma.toFixed(1)}` : '예측 전', '2σ interval, cycles')}
          {card('Confidence', Number.isFinite(confidence) ? `${confidence.toFixed(0)}%` : '0%', 'uncertainty / predicted RUL')}
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Anomaly report summary (expected vs observed)</h4>
        <div style={{ fontSize: 12, lineHeight: 1.6, color: '#444' }}>
          <p style={{ marginTop: 0 }}>
            <strong>Detected event:</strong> {issueLabel(anom)}
          </p>
          <p>
            <strong>First detectable cycle:</strong>{' '}
            {Number.isFinite(firstCycle) ? `${firstCycle}` : 'onset 미확정'}{' '}
            {Number.isFinite(lead) ? `· EOL 약 ${lead.toFixed(0)} cycles 이전 감지` : ''}
          </p>
            {battery === 'B0043' && (
              <p style={{ marginTop: -4, color: '#92400e', fontSize: 11, lineHeight: 1.45 }}>
                *B0043 transitions to a 4°C/4A load segment around cycle 42, and very-low-capacity discharge is observed from nearly the same point. However, the NASA README states that the root cause of these abrupt low-capacity runs in the low-temperature batch was not fully analyzed, so this dashboard does not claim a confirmed physical root cause.
              </p>
            )}
          <p style={{ marginBottom: 4 }}><strong>Potential drivers ranked before/near EOL:</strong></p>
          {drivers.length ? (
            <ol style={{ marginTop: 4, paddingLeft: 18 }}>
              {drivers.map((d: any, i: number) => (
                <li key={`${d.feature}-${i}`} style={{ marginBottom: 4 }}>
                  {d.label} {Number.isFinite(d.z) ? `(z=${d.z.toFixed(2)})` : ''}{d.action ? ` — ${d.action}` : ''}
                </li>
              ))}
            </ol>
          ) : (
            <div style={{ color: '#777' }}>driver ranking 없음</div>
          )}
          <div style={{ marginTop: 8, padding: 8, background: 'white', border: '1px solid #fed7aa', borderRadius: 6 }}>
            For B0018, this section is meant to capture early stress-driver warnings before large KPI deviation. For B0043, it summarizes capacity/DCR deviation timing when those cards appear.
          </div>
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Global Model Feature Importance</h4>
        <p style={{ fontSize: 12, color: '#555', lineHeight: 1.55 }}>Global model-level SHAP importance, not local anomaly driver ranking.</p>
        {shapItems.length ? (
          <Plot data={shapPlot} layout={{ height: 330, margin: { l: 130, r: 20, t: 20, b: 35 }, xaxis: { title: 'Importance' }, yaxis: { title: 'Feature' } }} config={{ responsive: true, displayModeBar: false }} useResizeHandler style={{ width: '100%' }} />
        ) : (
          <div style={{ padding: 12, color: '#777', background: '#f8fafc', borderRadius: 6 }}>SHAP global importance is not available from backend.</div>
        )}
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Model Architecture</h4>
        <div style={{ fontSize: 12, lineHeight: 1.65, color: '#444' }}>
          <p style={{ marginTop: 0 }}><strong>Backbone:</strong> CEEMDAN–Transformer–DNN decomposes noisy capacity trajectories and models local regeneration plus residual degradation trend.</p>
          <p style={{ marginBottom: 0 }}><strong>Meta-learning:</strong> BMAML-SVGD provides few-shot adaptation and uncertainty-aware RUL prediction.</p>
        </div>
      </div>
    </div>
  )
}
