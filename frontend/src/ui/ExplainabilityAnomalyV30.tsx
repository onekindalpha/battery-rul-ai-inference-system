
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

function eolCycle(pre: any): number {
  const cycles = pre?.cycles ?? []
  const truth = pre?.truth ?? []
  for (let i = 0; i < cycles.length; i += 1) {
    if (Number.isFinite(truth[i]) && truth[i] <= 0) return cycles[i]
  }
  return cycles.length ? cycles[cycles.length - 1] : NaN
}

function issueList(report: any): any[] {
  return [
    ...(Array.isArray(report?.majorAlerts) ? report.majorAlerts : []),
    ...(Array.isArray(report?.major_alerts) ? report.major_alerts : []),
    ...(Array.isArray(report?.issues) ? report.issues : []),
  ]
}

function hasAnomalySignal(report: any): boolean {
  if (!report) return false
  const status = String(report?.status ?? '').toLowerCase()
  const early = report?.earlyWarning?.active || report?.early_warning?.active
  const issues = issueList(report).length > 0
  const cap = Number.isFinite(Number(report?.cap_min_z)) && Number(report.cap_min_z) <= -3.5
  const dcr = Number.isFinite(Number(report?.dcr_max_z)) && Number(report.dcr_max_z) >= 3.5
  return Boolean(early || issues || cap || dcr || (status && status !== 'normal' && status !== 'ok'))
}

function signalLabel(report: any): string {
  const earlyMsg = report?.earlyWarning?.message || report?.early_warning?.message
  if (earlyMsg) return String(earlyMsg)

  const labels = issueList(report).map((x: any) => x?.label || x?.type).filter(Boolean)
  if (labels.length) return labels.join(' / ')

  const status = String(report?.status ?? '')
  if (status && status !== 'normal') return status

  return '특이 신호 없음'
}

function driversFrom(report: any): any[] {
  const drivers = Array.isArray(report?.drivers) ? report.drivers : []
  return drivers
    .map((d: any) => {
      const label = d?.label || d?.tag || d?.feature || 'driver'
      const feature = d?.feature || label
      const zRaw = Number(d?.z ?? d?.absZ ?? d?.abs_z ?? 0)
      const absZRaw = Number(d?.absZ ?? d?.abs_z ?? Math.abs(zRaw))
      return {
        key: String(feature),
        label: String(label),
        feature: String(feature),
        z: Number.isFinite(zRaw) ? zRaw : 0,
        absZ: Number.isFinite(absZRaw) ? Math.abs(absZRaw) : 0,
        action: d?.recommendation || d?.action || '',
      }
    })
    .filter((d: any) => d.key && Number.isFinite(d.absZ))
}

async function fetchReportsInBatches(battery: string, rRatio: string | number, cycles: number[]) {
  const out: { cycle: number; report: any }[] = []
  const batchSize = 8
  for (let i = 0; i < cycles.length; i += batchSize) {
    const batch = cycles.slice(i, i + batchSize)
    const results = await Promise.all(batch.map(async (c) => {
      try {
        const url = `/api/battery/${battery}/degradation-monitoring?r_ratio=${encodeURIComponent(String(rRatio))}&cycle=${encodeURIComponent(String(c))}`
        const report = await getJSON(url)
        return { cycle: c, report }
      } catch {
        return { cycle: c, report: null }
      }
    }))
    out.push(...results)
  }
  return out
}

function buildCumulativeSummary(reports: { cycle: number; report: any }[], eol: number) {
  const signalRows = reports
    .filter((r) => Number.isFinite(r.cycle) && (!Number.isFinite(eol) || r.cycle <= eol))
    .filter((r) => hasAnomalySignal(r.report))

  const first = signalRows.length ? signalRows[0] : null
  const firstCycle = first?.cycle ?? NaN
  const lead = Number.isFinite(eol) && Number.isFinite(firstCycle) ? Math.max(0, eol - firstCycle) : NaN

  const rankMap = new Map<string, any>()

  for (const row of signalRows) {
    for (const d of driversFrom(row.report)) {
      const prev = rankMap.get(d.key) || {
        key: d.key,
        label: d.label,
        feature: d.feature,
        count: 0,
        cycles: [] as number[],
        maxAbsZ: 0,
        sumAbsZ: 0,
        maxZ: d.z,
        action: d.action,
      }
      prev.count += 1
      prev.cycles.push(row.cycle)
      prev.maxAbsZ = Math.max(prev.maxAbsZ, d.absZ)
      prev.sumAbsZ += d.absZ
      if (d.absZ >= Math.abs(prev.maxZ ?? 0)) prev.maxZ = d.z
      if (!prev.action && d.action) prev.action = d.action
      rankMap.set(d.key, prev)
    }
  }

  const ranked = [...rankMap.values()]
    .map((x) => ({
      ...x,
      firstCycle: Math.min(...x.cycles),
      lastCycle: Math.max(...x.cycles),
      meanAbsZ: x.count ? x.sumAbsZ / x.count : 0,
    }))
    .sort((a, b) => (b.count - a.count) || (b.maxAbsZ - a.maxAbsZ) || (a.firstCycle - b.firstCycle))
    .slice(0, 5)

  return {
    signalRows,
    firstCycle,
    firstLabel: first ? signalLabel(first.report) : '수명 종료 전 anomaly report 신호 없음',
    lead,
    ranked,
  }
}

export function ExplainabilityAnomalyV30({ battery, rRatio, cycle }: { battery: string; rRatio: string | number; cycle: number }) {
  const [pre, setPre] = React.useState<any>(null)
  const [summary, setSummary] = React.useState<any>(null)
  const [shap, setShap] = React.useState<any>(null)
  const [warning, setWarning] = React.useState<string | null>(null)
  const [loadingSummary, setLoadingSummary] = React.useState(false)

  React.useEffect(() => {
    let alive = true
    async function run() {
      setWarning(null)
      setSummary(null)
      setLoadingSummary(true)
      try {
        const preRaw = await getJSON(`/api/battery/${battery}/precomputed?r_ratio=${encodeURIComponent(String(rRatio))}`)
        const nextPre = unwrapPrecomputed(preRaw)
        if (!alive) return
        setPre(nextPre)

        const eol = eolCycle(nextPre)
        const cyclesBeforeEol = (nextPre.cycles ?? [])
          .filter((c: number) => Number.isFinite(c) && (!Number.isFinite(eol) || c <= eol))
          .sort((a: number, b: number) => a - b)

        const [reports, shapRaw] = await Promise.all([
          fetchReportsInBatches(battery, rRatio, cyclesBeforeEol),
          getJSON(`/api/fixed4/shap-current`).catch(() => ({ ok: false, items: [] })),
        ])

        if (!alive) return
        setSummary(buildCumulativeSummary(reports, eol))
        setShap(shapRaw)
      } catch (e: any) {
        if (alive) setWarning(String(e?.message || e))
      } finally {
        if (alive) setLoadingSummary(false)
      }
    }

    if (battery) run()
    return () => { alive = false }
  }, [battery, rRatio])

  const idx = pre?.cycles ? nearestIndex(pre.cycles, Number(cycle)) : -1
  const pred = idx >= 0 ? n(pre.pred?.[idx]) : NaN
  const sigma = idx >= 0 ? n(pre.std?.[idx]) : NaN
  const uncertainty2Sigma = Number.isFinite(sigma) ? 2 * sigma : NaN
  const confidence = Number.isFinite(uncertainty2Sigma) && Math.abs(pred) > 1
    ? Math.max(0, Math.min(100, 100 * (1 - Math.min(1, uncertainty2Sigma / Math.abs(pred)))))
    : NaN

  const shapItems = (shap?.items || []).slice(0, 12).slice().reverse()
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
        Uncertainty, cumulative anomaly-report evidence before EOL, global SHAP importance, and model architecture. EOL means the end-of-life point where true RUL reaches 0; anomaly onset is reported separately.
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
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Cumulative anomaly report summary (expected vs observed)</h4>
        {loadingSummary ? (
          <div style={{ padding: 10, background: 'white', border: '1px solid #fed7aa', borderRadius: 6, fontSize: 12 }}>누적 anomaly evidence 계산 중...</div>
        ) : (
          <div style={{ fontSize: 12, lineHeight: 1.6, color: '#444' }}>
            <p style={{ marginTop: 0 }}>
              <strong>First detected signal:</strong> {summary?.firstLabel ?? '수명 종료 전 anomaly report 신호 없음'}
            </p>
            <p>
              <strong>First detectable cycle:</strong>{' '}
              {Number.isFinite(summary?.firstCycle) ? `${summary.firstCycle}` : 'onset 미확정'}{' '}
              {Number.isFinite(summary?.lead) ? `· EOL 약 ${summary.lead.toFixed(0)} cycles 이전 감지${battery === 'B0043' ? '*' : ''}` : ''}
            </p>
            {battery === 'B0043' && (
              <p style={{ marginTop: -4, color: '#92400e', fontSize: 11, lineHeight: 1.45 }}>
                *B0043 transitions to a 4°C/4A load segment around cycle 42, and very-low-capacity discharge is observed from nearly the same point. However, the NASA README states that the root cause of these abrupt low-capacity runs in the low-temperature batch was not fully analyzed, so this dashboard does not claim a confirmed physical root cause.
              </p>
            )}
            <p style={{ marginBottom: 4 }}>
              <strong>Cumulative potential driver ranking before EOL:</strong>
            </p>
            {summary?.ranked?.length ? (
              <ol style={{ marginTop: 4, paddingLeft: 18 }}>
                {summary.ranked.slice(0, 5).map((d: any, i: number) => (
                  <li key={`${d.feature}-${i}`} style={{ marginBottom: 6 }}>
                    <strong>{d.label}</strong>
                    {' '}· appeared {d.count} signal cycles
                    {' '}· first {d.firstCycle}
                    {' '}· max |z| {d.maxAbsZ.toFixed(2)}
                    {d.action ? ` — ${d.action}` : ''}
                  </li>
                ))}
              </ol>
            ) : (
              <div style={{ color: '#777' }}>수명 종료 전 누적 driver ranking 없음</div>
            )}
            <div style={{ marginTop: 8, padding: 8, background: 'white', border: '1px solid #fed7aa', borderRadius: 6 }}>
              Ranking is cumulative: drivers are sorted by how often they appeared in anomaly-report signal cycles before EOL, then by max |z|. This is separate from the current-cycle driver list in the Degradation tab.
            </div>
          </div>
        )}
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Global Model Feature Importance</h4>
        <p style={{ fontSize: 12, color: '#555', lineHeight: 1.55 }}>Global model-level SHAP importance, not local anomaly driver ranking.</p>
        {/* HF_ALWAYS_SHOW_SHAP_IMAGE_V1: show global SHAP image before any prediction/reinference */}
        <div style={{ padding: 12, background: '#0f172a', borderRadius: 10, margin: '10px 0 14px 0' }}>
          {null}</div>
        {shapItems.length ? (
          <Plot data={shapPlot} layout={{ height: 330, margin: { l: 130, r: 20, t: 20, b: 35 }, xaxis: { title: 'Importance' }, yaxis: { title: 'Feature' } }} config={{ responsive: true, displayModeBar: false }} useResizeHandler style={{ width: '100%' }} />
        ) : (
          <div style={{ padding: 12, background: '#0f172a', borderRadius: 10 }}>
            {null}</div>
        )}
      </div>

      <div style={{ marginBottom: 16, padding: 12, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 900 }}>Model Architecture</h4>
        <div style={{ fontSize: 12, lineHeight: 1.65, color: '#444' }}>
          <p style={{ marginTop: 0 }}><strong>Backbone:</strong> CEEMDAN–Transformer–DNN. CEEMDAN decomposes noisy capacity trajectories into local components and residual trend; Transformer/DNN modules model temporal degradation patterns.</p>
          <p><strong>Meta-learning:</strong> BMAML-SVGD provides few-shot adaptation and uncertainty-aware RUL prediction using SVGD particles.</p><p style={{ marginBottom: 0 }}><strong>Dashboard interpretation:</strong> Overview shows forecast/error values; Degradation shows local anomaly monitoring; this Explainability tab summarizes uncertainty and cumulative pre-EOL anomaly evidence.</p>
        </div>
      </div>
    </div>
  )
}
