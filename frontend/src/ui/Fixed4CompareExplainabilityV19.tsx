
import React from 'react'
import Plot from 'react-plotly.js'

type AnyObj = Record<string, any>

const FALLBACK_BATTERIES = ['B0018', 'B0033', 'B0042', 'B0043', 'B0055']

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
  const txt = await res.text()
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${txt.slice(0, 240)}`)
  try { return JSON.parse(txt) } catch { throw new Error(`Invalid JSON from ${url}: ${txt.slice(0, 240)}`) }
}

function unwrapPrecomputed(raw: any) {
  const item = raw?.payload ?? raw?.item ?? raw?.data ?? raw?.precomputed ?? raw
  const cycles = (item?.query?.cycle ?? item?.cycles ?? []).map(n)
  const pred = (item?.pred?.mean ?? item?.rul_pred ?? item?.predRUL ?? []).map(n)
  const truth = (item?.query?.true_rul ?? item?.rul_true ?? item?.trueRUL ?? []).map(n)
  const std = (item?.pred?.std ?? item?.rul_std ?? item?.std ?? []).map(n)
  return { item, cycles, pred, truth, std, metrics: item?.metrics ?? { rmse: item?.rmse, mae: item?.mae } }
}

const METRICS = [
  ['soh', 'SoH (%)'],
  ['capacity_pct', 'Capacity (% of initial)'],
  ['impedance_sum', 'Impedance sum (Ω)'],
  ['dcr', 'DCR (Ω)'],
  ['thermal_stress', 'Thermal stress'],
  ['temperature_mean', 'Temp mean (°C)'],
  ['lli', 'LLI'],
  ['lam', 'LAM'],
] as const

export function Fixed4CompareTabV19({
  battery,
  cycle,
  batteries,
}: {
  battery: string
  cycle: number
  batteries?: string[]
}) {
  const options = React.useMemo(
    () => Array.from(new Set([battery, ...(batteries || []), ...FALLBACK_BATTERIES].filter(Boolean))),
    [battery, batteries]
  )

  const [selected, setSelected] = React.useState<string[]>(() => Array.from(new Set([battery, 'B0043'].filter(Boolean))).slice(0, 4))
  const [metric, setMetric] = React.useState<string>('capacity_pct')
  const [showBand, setShowBand] = React.useState(true)
  const [cohort, setCohort] = React.useState('all')
  const [data, setData] = React.useState<any>(null)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (!battery) return
    setSelected((prev) => prev.includes(battery) ? prev : [battery, ...prev].slice(0, 4))
  }, [battery])

  React.useEffect(() => {
    let alive = true
    async function run() {
      if (!selected.length) return
      setLoading(true)
      setError(null)
      try {
        const url = `/api/fixed4/compare?batteries=${encodeURIComponent(selected.join(','))}&metric=${encodeURIComponent(metric)}&cohort=${encodeURIComponent(cohort)}&cycle=${encodeURIComponent(String(cycle || ''))}`
        const d = await getJSON(url)
        if (alive) setData(d)
      } catch (e: any) {
        if (alive) setError(String(e?.message || e))
      } finally {
        if (alive) setLoading(false)
      }
    }
    run()
    return () => { alive = false }
  }, [selected.join(','), metric, cohort, cycle])

  const traces: any[] = []
  if (showBand && data?.band?.x?.length) {
    traces.push({
      x: [...data.band.x, ...data.band.x.slice().reverse()],
      y: [...data.band.q75, ...data.band.q25.slice().reverse()],
      type: 'scatter',
      mode: 'lines',
      fill: 'toself',
      fillcolor: 'rgba(148, 163, 184, 0.25)',
      line: { width: 0 },
      name: 'IQR (cohort)',
      hoverinfo: 'skip',
    })
    traces.push({
      x: data.band.x,
      y: data.band.median,
      type: 'scatter',
      mode: 'lines',
      line: { dash: 'dash', width: 2, color: '#64748b' },
      name: 'Median (cohort)',
    })
  }

  ;(data?.series || []).forEach((s: any) => {
    traces.push({
      x: s.x || [],
      y: s.y || [],
      type: 'scatter',
      mode: 'lines',
      name: s.battery,
      line: { width: String(s.battery) === String(battery) ? 3 : 2 },
      hovertemplate: `${s.battery}<br>Cycle: %{x}<br>${data?.metric_label || metric}: %{y:.4f}<extra></extra>`,
    })
  })

  const summaryRows = (data?.series || []).map((s: any) => {
    const xs = (s.x || []).map(n)
    const ys = (s.y || []).map(n)
    const idx = nearestIndex(xs, Number(cycle))
    const val = idx >= 0 ? ys[idx] : NaN
    return { battery: s.battery, value: val, cycle: idx >= 0 ? xs[idx] : NaN }
  })

  return (
    <div style={{ overflowY: 'auto', paddingRight: 4 }}>
      <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 15, fontWeight: 800 }}>⚖️ Compare / Fleet view</h3>
      <p style={{ fontSize: 12, color: '#666', margin: '0 0 12px 0', lineHeight: 1.55 }}>
        Fixed4-style comparison using cycle-level degradation features: SoH, Capacity%, DCR, Impedance, Thermal, Temp, LLI, and LAM.
      </p>

      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12, padding: 10, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <strong style={{ fontSize: 12 }}>Batteries</strong>
        {options.map((b) => (
          <label key={b} style={{ display: 'flex', gap: 5, alignItems: 'center', padding: '6px 9px', borderRadius: 999, border: selected.includes(b) ? '1px solid #1976d2' : '1px solid #ddd', background: selected.includes(b) ? '#1976d2' : 'white', color: selected.includes(b) ? 'white' : '#333', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={selected.includes(b)}
              onChange={(e) => e.target.checked ? setSelected((p) => Array.from(new Set([...p, b])).slice(0, 4)) : setSelected((p) => p.filter((x) => x !== b))}
            />
            {b}
          </label>
        ))}

        <select value={metric} onChange={(e) => setMetric(e.target.value)} style={{ padding: '7px 10px', borderRadius: 6, border: '1px solid #ccc', fontSize: 12 }}>
          {METRICS.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
        </select>

        <select value={cohort} onChange={(e) => setCohort(e.target.value)} style={{ padding: '7px 10px', borderRadius: 6, border: '1px solid #ccc', fontSize: 12 }}>
          <option value="all">Reference cohort: all</option>
          <option value="temp_le_10">Temp ≤ 10°C</option>
          <option value="temp_10_25">10°C &lt; Temp &lt; 25°C</option>
          <option value="temp_ge_25">Temp ≥ 25°C</option>
        </select>

        <label style={{ fontSize: 12, display: 'flex', gap: 5, alignItems: 'center' }}>
          <input type="checkbox" checked={showBand} onChange={(e) => setShowBand(e.target.checked)} />
          Expected band
        </label>
      </div>

      {loading && <div style={{ padding: 10, color: '#666' }}>Loading fixed4 compare data...</div>}
      {error && <div style={{ padding: 10, color: '#d32f2f', background: '#fff5f5', border: '1px solid #ffcaca', borderRadius: 6 }}>Compare error: {error}</div>}

      <div style={{ overflowX: 'auto', marginBottom: 12 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', background: 'white', border: '1px solid #ddd', fontSize: 11 }}>
          <thead>
            <tr style={{ background: '#f5f5f5' }}>
              <th style={{ padding: 8, textAlign: 'left' }}>Battery</th>
              <th style={{ padding: 8, textAlign: 'center' }}>Nearest cycle</th>
              <th style={{ padding: 8, textAlign: 'center' }}>{data?.metric_label || metric}</th>
              <th style={{ padding: 8, textAlign: 'center' }}>Line</th>
            </tr>
          </thead>
          <tbody>
            {summaryRows.map((r: any) => (
              <tr key={r.battery} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: 8, fontWeight: 800, color: String(r.battery) === String(battery) ? '#1976d2' : '#333' }}>{r.battery}</td>
                <td style={{ padding: 8, textAlign: 'center' }}>{Number.isFinite(r.cycle) ? r.cycle : 'N/A'}</td>
                <td style={{ padding: 8, textAlign: 'center', fontWeight: 700 }}>{Number.isFinite(r.value) ? r.value.toFixed(4) : 'N/A'}</td>
                <td style={{ padding: 8, textAlign: 'center' }}>{String(r.battery) === String(battery) ? 'selected · width 3' : 'width 2'}</td>
              </tr>
            ))}
            {!summaryRows.length && <tr><td colSpan={4} style={{ padding: 14, textAlign: 'center', color: '#999' }}>No compare data.</td></tr>}
          </tbody>
        </table>
      </div>

      <div style={{ padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        {traces.length ? (
          <Plot
            data={traces}
            layout={{
              height: 420,
              margin: { l: 55, r: 20, t: 20, b: 45 },
              xaxis: { title: 'Cycle' },
              yaxis: { title: data?.metric_label || metric },
              hovermode: 'x unified',
              legend: { orientation: 'h', y: 1.14, x: 0 },
              shapes: Number.isFinite(Number(cycle)) ? [{
                type: 'line',
                xref: 'x',
                yref: 'paper',
                x0: Number(cycle),
                x1: Number(cycle),
                y0: 0,
                y1: 1,
                line: { color: '#e74c3c', dash: 'dash', width: 2 },
              }] : [],
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: '100%' }}
          />
        ) : (
          <div style={{ padding: 24, textAlign: 'center', color: '#999' }}>No plot data available.</div>
        )}
      </div>
    </div>
  )
}

export function Fixed4ExplainabilityTabV19({
  battery,
  rRatio,
  cycle,
}: {
  battery: string
  rRatio: string | number
  cycle: number
}) {
  const [pre, setPre] = React.useState<any>(null)
  const [deg, setDeg] = React.useState<any>(null)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [inferring, setInferring] = React.useState(false)
  const [inferLog, setInferLog] = React.useState<any>(null)

  async function load() {
    if (!battery) return
    setLoading(true)
    setError(null)
    try {
      const [pRaw, d] = await Promise.all([
        getJSON(`/api/battery/${battery}/precomputed?r_ratio=${encodeURIComponent(String(rRatio))}`),
        getJSON(`/api/battery/${battery}/degradation-monitoring?r_ratio=${encodeURIComponent(String(rRatio))}`),
      ])
      setPre(unwrapPrecomputed(pRaw))
      setDeg(d)
    } catch (e: any) {
      setError(String(e?.message || e))
    } finally {
      setLoading(false)
    }
  }

  React.useEffect(() => { load() }, [battery, rRatio])

  const idx = pre?.cycles ? nearestIndex(pre.cycles, Number(cycle)) : -1
  const pred = idx >= 0 ? n(pre.pred?.[idx]) : NaN
  const truth = idx >= 0 ? n(pre.truth?.[idx]) : NaN
  const sigma = idx >= 0 ? n(pre.std?.[idx]) : NaN
  const residual = Number.isFinite(pred) && Number.isFinite(truth) ? pred - truth : NaN
  const u2 = Number.isFinite(sigma) ? 2 * sigma : NaN
  const conf = Number.isFinite(u2) && Math.abs(pred) > 1 ? Math.max(0, Math.min(100, 100 * (1 - Math.min(1, u2 / Math.abs(pred))))) : NaN
  const confLabel = !Number.isFinite(conf) ? 'N/A' : conf >= 80 ? 'High' : conf >= 55 ? 'Medium' : 'Low'
  const capMinZ = n(deg?.cap_min_z)
  const dcrMaxZ = n(deg?.dcr_max_z)

  const card = (title: string, value: string, suffix = '', color = '#111') => (
    <div style={{ padding: 12, background: 'white', border: '1px solid #e5e7eb', borderRadius: 8 }}>
      <div style={{ fontSize: 11, color: '#666', marginBottom: 4, fontWeight: 800 }}>{title}</div>
      <div style={{ fontSize: 22, fontWeight: 900, color }}>{value} <span style={{ fontSize: 12, opacity: 0.7 }}>{suffix}</span></div>
    </div>
  )

  async function runLive() {
    setInferring(true)
    setInferLog(null)
    const started = Date.now()
    try {
      const out = await getJSON(`/api/live-reinfer/${battery}?r_ratio=${encodeURIComponent(String(rRatio))}&timeout=360`)
      setInferLog({ elapsed_sec_client: ((Date.now() - started) / 1000).toFixed(1), ...out })
      await load()
    } catch (e: any) {
      setInferLog({ ok: false, error: String(e?.message || e) })
    } finally {
      setInferring(false)
    }
  }

  return (
    <div style={{ overflowY: 'auto', paddingRight: 4 }}>
      <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 15, fontWeight: 800 }}>🔍 Prediction Explainability</h3>
      <p style={{ fontSize: 12, color: '#666', margin: '0 0 12px 0', lineHeight: 1.55 }}>
        Model-level explanation and live inference control. Local robust-z anomaly details stay in the Degradation tab.
      </p>

      {loading && <div style={{ padding: 10, color: '#666' }}>Loading explainability data...</div>}
      {error && <div style={{ padding: 10, color: '#d32f2f', background: '#fff5f5', border: '1px solid #ffcaca', borderRadius: 6 }}>Explainability error: {error}</div>}

      <div style={{ marginBottom: 16, padding: 12, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Prediction Confidence</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8 }}>
          {card('Predicted RUL', Number.isFinite(pred) ? pred.toFixed(1) : 'N/A', 'cycles', '#1976d2')}
          {card('True RUL', Number.isFinite(truth) ? truth.toFixed(1) : 'N/A', 'cycles')}
          {card('Uncertainty', Number.isFinite(u2) ? `±${u2.toFixed(1)}` : 'N/A', '2σ', '#ff6f00')}
          {card('Confidence', Number.isFinite(conf) ? `${conf.toFixed(0)}%` : 'N/A', confLabel, confLabel === 'High' ? '#16a34a' : confLabel === 'Medium' ? '#f59e0b' : '#dc2626')}
        </div>
        <div style={{ marginTop: 10, padding: 10, background: 'white', border: '1px solid #e5e7eb', borderRadius: 6, fontSize: 12, lineHeight: 1.55 }}>
          <strong>Residual interpretation:</strong>{' '}
          {Number.isFinite(residual) ? `${residual >= 0 ? '+' : ''}${residual.toFixed(1)} cycles. ` : 'Observed RUL is not available at the selected cycle. '}
          {Number.isFinite(residual) && Number.isFinite(u2) && Math.abs(residual) <= Math.max(5, u2)
            ? 'Prediction is inside the current uncertainty range.'
            : Number.isFinite(residual) && residual > 0
              ? 'This is an over-estimation risk.'
              : Number.isFinite(residual)
                ? 'This is a conservative under-estimation.'
                : ''}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.15fr 0.85fr', gap: 12, marginBottom: 16 }}>
        <div style={{ padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Model Architecture</h4>
          <div style={{ fontSize: 12, lineHeight: 1.65, color: '#444' }}>
            <p style={{ marginTop: 0 }}><strong>Backbone:</strong> CEEMDAN–Transformer–DNN. CEEMDAN decomposes noisy capacity trajectories into local regeneration components and a global residual trend.</p>
            <p><strong>Sequence model:</strong> Transformer modules model local IMF/regeneration components; DNN models the residual degradation trend.</p>
            <p style={{ marginBottom: 0 }}><strong>Meta-learning layer:</strong> BMAML-SVGD is applied on top for few-shot adaptation and uncertainty-aware RUL prediction.</p>
          </div>
        </div>

        <div style={{ padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Initialize & Re-inference</h4>
          <p style={{ fontSize: 12, lineHeight: 1.55, color: '#555' }}>
            Runs the live backend model runner. This endpoint does not silently return precomputed output when the runner is missing.
          </p>
          <button
            onClick={runLive}
            disabled={inferring || !battery}
            style={{ padding: '9px 12px', background: inferring ? '#bbb' : '#1976d2', color: 'white', border: 'none', borderRadius: 6, width: '100%', cursor: inferring ? 'not-allowed' : 'pointer', fontSize: 12, fontWeight: 900 }}
          >
            {inferring ? 'Running live BMAML-SVGD inference...' : 'Initialize & Re-inference'}
          </button>
          {inferLog && <pre style={{ marginTop: 10, maxHeight: 155, overflow: 'auto', padding: 8, background: '#0f172a', color: inferLog.ok === false ? '#fecaca' : '#d1fae5', borderRadius: 6, fontSize: 10, whiteSpace: 'pre-wrap' }}>{JSON.stringify(inferLog, null, 2)}</pre>}
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Global Model Importance</h4>
        <p style={{ fontSize: 12, color: '#555', lineHeight: 1.55, marginTop: 0 }}>
          Model-level feature families are separated from local robust-z drivers to avoid duplicating the Degradation tab.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8 }}>
          {[
            ['Capacity trajectory', 'CEEMDAN separates regeneration and long-term degradation trend.'],
            ['Resistance / DCR', 'Resistance-related signals are diagnostic indicators for lifetime.'],
            ['Thermal / stress', 'Temperature and stress features help explain non-uniform degradation.'],
            ['Usage pattern', 'C-rate/current/voltage dynamics reflect operational load.'],
          ].map(([name, desc]) => (
            <div key={name} style={{ padding: 10, border: '1px solid #eee', borderRadius: 8, background: '#f9fafb' }}>
              <div style={{ fontSize: 12, fontWeight: 900, marginBottom: 5 }}>{name}</div>
              <div style={{ fontSize: 11, color: '#666', lineHeight: 1.45 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Degradation evidence bridge</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
          {card('Degradation status', String(deg?.status || 'N/A'), '', deg?.status === 'major-anomaly' ? '#dc2626' : '#16a34a')}
          {card('Capacity min z', Number.isFinite(capMinZ) ? capMinZ.toFixed(2) : 'N/A', 'prefix', Number.isFinite(capMinZ) && capMinZ <= -3.5 ? '#dc2626' : '#111')}
          {card('DCR max z', Number.isFinite(dcrMaxZ) ? dcrMaxZ.toFixed(2) : 'N/A', 'prefix', Number.isFinite(dcrMaxZ) && dcrMaxZ >= 4 ? '#dc2626' : '#111')}
        </div>
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Methods & References</h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 11, lineHeight: 1.5 }}>
          <div style={{ padding: 10, background: 'white', borderRadius: 6, border: '1px solid #eee' }}><strong>CEEMDAN–Transformer–DNN</strong><br />Early lithium-ion battery RUL prediction using decomposition, Transformer modules, and DNN residual trend modeling.</div>
          <div style={{ padding: 10, background: 'white', borderRadius: 6, border: '1px solid #eee' }}><strong>BMAML-SVGD / uncertainty</strong><br />Few-shot RUL prediction using Bayesian meta-learning, particle uncertainty, and SVGD-style adaptation.</div>
          <div style={{ padding: 10, background: 'white', borderRadius: 6, border: '1px solid #eee' }}><strong>Resistance diagnostics</strong><br />Early resistance signals can be informative diagnostic features for lifetime prediction.</div>
          <div style={{ padding: 10, background: 'white', borderRadius: 6, border: '1px solid #eee' }}><strong>EV battery aging data</strong><br />RPT, HPPC, EIS, and capacity diagnostics motivate multi-feature monitoring.</div>
        </div>
      </div>
    </div>
  )
}
