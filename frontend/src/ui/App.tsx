// HOTFIX_V20_FIXED4_COMPARE_EXPLAIN_SAFE
// HOTFIX_V19_FIXED4_COMPARE_EXPLAIN_LIVE
import React, { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import * as echarts from 'echarts'
import Plot from 'react-plotly.js'
import { Fixed4CompareTabV19, Fixed4ExplainabilityTabV19 } from './Fixed4CompareExplainabilityV19'
import { Fixed4CompareTabV20 } from './Fixed4CompareExplainabilityV20'
import { ExplainabilityAnomalyV30 } from './ExplainabilityAnomalyV30'

type Row = Record<string, any>

type BatchItem = {
  battery_id: string
  split_cycle: number
  len_s?: number
  len_q?: number
  q_pos?: number
  r_ratio_input?: number
  r_ratio_effective?: number
  current_true_rul?: number
  current_pred_mean?: number
  current_cycle_effective?: number
  support?: { cycle?: number[]; rul?: number[] }
  query?: { cycle?: number[]; true_rul?: number[] }
  pred?: { mean?: number[]; std?: number[] }
  metrics?: { rmse?: number; mae?: number }
  // dashboard_export_v2 format properties
  cycles?: number[]
  rul_pred?: number[]
  rul_true?: number[]
  rul_std?: number[]
  rmse?: number
  mae?: number
}

type BatchPayload = {
  tag?: string
  created_at?: string
  items: BatchItem[]
}

type SeriesPack = {
  supportCycles: number[]
  supportRUL: number[]
  cycles: number[]
  trueRUL: number[]
  predRUL: number[]
  std: number[]
  split: number | null
}

function toTag(rr: number) {
  const n = Math.round(rr * 100)
  return `r0p${String(n).padStart(2, '0')}`
}

function safeNum(x: any, fallback = 0) {
  const v = Number(x)
  return Number.isFinite(v) ? v : fallback
}

function buildPack(it: any): SeriesPack {
  // Handle both array and object formats for support
  const sup = it.support
  let supportCycles: number[] = []
  let supportRUL: number[] = []

  if (Array.isArray(sup)) {
    // Array format: [34, 35, 44, ...]
    supportCycles = sup.map((v: any) => safeNum(v))
    // Generate fake RUL values for display (descending from max to min)
    const maxVal = Math.max(...supportCycles)
    supportRUL = supportCycles.map((c: number) => maxVal - c * 0.1)
  } else if (sup && typeof sup === 'object') {
    // Object format: {cycle: [], rul: []}
    supportCycles = (sup.cycle ?? []).map((v: any) => safeNum(v))
    supportRUL = (sup.rul ?? []).map((v: any) => safeNum(v))
  }

  const cycles = (it.query?.cycle ?? []).map((v: any) => safeNum(v))
  const trueRUL = (it.query?.true_rul ?? []).map((v: any) => safeNum(v))
  const predRUL = (it.pred?.mean ?? []).map((v: any) => safeNum(v))
  const std = Array.isArray(it.pred?.std) ? it.pred!.std!.map((v: any) => safeNum(v)) : new Array(cycles.length).fill(0)
  const split = Number.isFinite(Number(it.split_cycle)) ? Number(it.split_cycle) : null
  return { supportCycles, supportRUL, cycles, trueRUL, predRUL, std, split }
}

function Card({ title, value, suffix, compact }: { title: string; value: any; suffix: string; compact?: boolean }) {
  const v = typeof value === 'number' && Number.isFinite(value) ? value : Number(value)
  const isNA = !Number.isFinite(v)
  const isPredictionValue = title.includes('Pred RUL') || title.includes('Predicted RUL') || title.includes('Uncertainty') || title.includes('Confidence')
  const isObservedValue = title.includes('True RUL')
  const isEvaluationValue = title.includes('RMSE') || title.includes('MAE') || title.includes('Abs Error') || title.includes('MAE') || title.includes('Current Abs Error') || title.includes('Error')
  const isMetric = title.includes('RMSE') || title.includes('MAE') || title.includes('Current Abs Error')
  let display = ''
  if (isNA) {
    display = isPredictionValue ? '예측 전' : isObservedValue ? '관측 전' : isEvaluationValue ? '평가 전' : '실행 전'
  } else if (isMetric) {
    display = v.toFixed(2)
  } else {
    display = Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)
  }
  return (
    <div style={{
      border: isNA ? '1px solid #ddd' : '1px solid #eee',
      borderRadius: 4,
      padding: compact ? 4 : 6,
      backgroundColor: isNA ? '#fafafa' : 'white'
    }}>
      <div style={{ fontSize: compact ? 12 : 12, color: '#999', marginBottom: compact ? 3 : 3 }}>{title}</div>
      <div style={{
        fontSize: compact ? 18 : 18,
        fontWeight: 700,
        color: isNA ? '#999' : '#000'
      }}>
        {display} <span style={{ fontSize: compact ? 11 : 11, opacity: 0.7 }}>{suffix}</span>
      </div>
    </div>
  )
}

// RUL trajectory 차트
function RULChart({ rows, cycle, pack }: any) {
  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!chartRef.current || !rows.length || !pack) return

    if (!chart.current) {
      chart.current = echarts.init(chartRef.current)
    }

    const instance = chart.current

    const seriesData = [
      {
        name: '실제 RUL',
        type: 'line',
        data: pack!.trueRUL,
        smooth: false,
        lineStyle: { width: 1.5, type: 'dashed', color: 'rgba(0,0,0,0.5)' },
      },
      {
        name: '관측 RUL (과거)',
        type: 'line',
        data: pack.supportRUL.concat(pack!.trueRUL.slice(0, pack!.cycles.length - pack.supportRUL.length)),
        smooth: false,
        lineStyle: { width: 1.5 },
        color: '#464646',
      },
    ]

    // 예측 RUL 추가 (future region)
    if (pack!.predRUL && pack!.predRUL.length > 0) {
      const predData = new Array(pack!.cycles.length).fill(null)
      pack!.predRUL.forEach((val: number, idx: number) => {
        predData[pack.supportRUL.length + idx] = val
      })
      seriesData.push({
        name: '예측 RUL',
        type: 'line',
        data: predData,
        smooth: false,
        lineStyle: { width: 1.5 },
        color: '#d62728',
      })
    }

    const option: echarts.EChartsOption = {
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0, left: 0, textStyle: { fontSize: 10 } },
      grid: { left: 35, right: 15, top: 20, bottom: 40 },
      xAxis: { type: 'category', data: pack!.cycles, boundaryGap: false },
      yAxis: { type: 'value', splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } } },
      series: seriesData as any,
    }

    instance.setOption(option, true)
  }, [rows, cycle, pack])

  return <div ref={chartRef} style={{ width: '100%', height: 200 }} />
}

// Z-score 차트
function AnomalyZScoreChart({ rows, cycle, bands, commonCycles, xMin, xMax, dcrZSeries, capZSeries }: any) {
  if (!rows?.length || !bands || !commonCycles?.length || !dcrZSeries || !capZSeries) {
    return <div style={{ padding: 12, color: '#999', fontSize: 11 }}>Loading anomaly data...</div>
  }

  // z-score 시계열에서 숫자 배열만 추출 (Anomaly chart용)
  // dcrZSeries와 capZSeries는 { z, cycle } 객체 배열이므로 z값만 뽑기
  const zDcr = dcrZSeries.map((p: any) => p?.z ?? null)
  const zCap = capZSeries.map((p: any) => p?.z ?? null)

  // Plotly data: actual cycle 번호를 x값으로 사용
  const data = [
    {
      x: commonCycles,
      y: zDcr,
      name: 'Robust z(DCR)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#1f77b4', width: 2 },
      showlegend: false,
      hovertemplate: '<b>Robust z(DCR)</b><br>Cycle: %{x}<br>z: %{y:.2f}<extra></extra>',
    },
    {
      x: commonCycles,
      y: zCap,
      name: 'Robust z(Capacity%)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#ff7f0e', width: 2 },
      showlegend: false,
      hovertemplate: '<b>Robust z(Capacity%)</b><br>Cycle: %{x}<br>z: %{y:.2f}<extra></extra>',
    },
    // Threshold +3 line
    {
      x: [xMin, xMax],
      y: [3, 3],
      name: 'z = +3 (HIGH)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#d62728', width: 1, dash: 'dash' },
      showlegend: false,
      hovertemplate: 'Threshold +3<extra></extra>',
    },
    // Threshold -3 line
    {
      x: [xMin, xMax],
      y: [-3, -3],
      name: 'z = -3 (HIGH)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#d62728', width: 1, dash: 'dash' },
      showlegend: false,
      hovertemplate: 'Threshold -3<extra></extra>',
    },
  ]

  const layout: any = {
    title: '',
    height: 220,
    margin: { l: 70, r: 20, t: 10, b: 40 },
    hovermode: 'x unified',
    showlegend: false,
    xaxis: {
      range: [xMin, xMax],
      zeroline: false,
    },
    yaxis: {
      title: 'Robust z-score',
      titlefont: { size: 11 },
      zeroline: true,
      zerolinewidth: 1,
      zerolinecolor: '#ccc',
    },
    shapes: cycle ? [{
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: cycle,
      x1: cycle,
      y0: 0,
      y1: 1,
      line: { color: '#e74c3c', dash: 'dot', width: 2 }
    }] : []
  }

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} useResizeHandler style={{ width: '100%' }} />
}

// Potential Drivers 리스트 (Fixed4 스타일) - targetCycle 기준
function PotentialDriversList({ rows, cycle, bands, targetCycle }: any) {
  const computeRobustZ = (value: number, median: number, q1: number, q3: number) => {
    const iqr = q3 - q1
    const scale = Math.max(iqr / 1.349, 1e-9)
    return (value - median) / scale
  }

  const baseCycle = targetCycle ?? cycle
  const currentRow = rows.find((r: any) => r.cycle_num === baseCycle)
  if (!currentRow) return <div style={{ padding: 12, color: '#999', fontSize: 12 }}>No data for cycle {baseCycle}</div>

  // Driver 정의 (Fixed4 driver_candidates 순서 그대로)
  const driverDefs: any[] = [
    { key: 'thermal_stress', label: '고온/열 스트레스', diagnosis: '열관리 점검(팬/냉각)·고온 구간 제한' },
    { key: 'temperature_mean', label: '고온 노출', diagnosis: '냉각/통풍·고온 운행 제한' },
    { key: 'temp_rise_cycle', label: '셀 발열 증가', diagnosis: '열 runaway 위험 체크·냉각 강화' },
    { key: 'eff_c_rate', label: '고 C-rate(고부하)', diagnosis: '가속/급속충전 제한·부하 분산' },
    { key: 'current_max', label: '고부하(충전/회생)', diagnosis: '피크 전류 제한·회생제동 설정 조정' },
    { key: 'current_min', label: '고부하(방전)', diagnosis: '피크 방전 전류 제한·부하 분산' },
    { key: 'voltage_min', label: '깊은 방전(DoD↑)', diagnosis: '최저 SoC 제한·운영전략 조정' },
    { key: 'dvdt_max_abs', label: '전압 급변', diagnosis: 'BMS 로깅/센서 점검·전력 프로파일 확인' },
    { key: 'dTdt_max', label: '온도 급상승', diagnosis: '열관리/센서 점검·운행 제한' },
  ]

  const drivers = driverDefs
    .map((def: any) => {
      const value = currentRow[def.key]
      if (value === null || value === undefined) return null

      const bandData = bands[def.key]?.find((b: any) => b.cycle === baseCycle)
      if (!bandData) return null

      const z = computeRobustZ(value, bandData.median, bandData.q1, bandData.q3)
      return { ...def, value, z, absZ: Math.abs(z) }
    })
    .filter((d: any) => d !== null)
    .sort((a: any, b: any) => b.absZ - a.absZ)
    .slice(0, 3)

  if (drivers.length === 0) {
    return (
      <div style={{ padding: 12, backgroundColor: '#f5f5f5', borderRadius: 4, border: '1px solid #ddd', fontSize: 11, color: '#666', lineHeight: 1.6 }}>
        driver 후보 피처가 부족하거나 cohort 분포가 충분하지 않아 Top driver를 계산하지 못했습니다.
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {drivers.map((d: any, idx: number) => (
        <div key={idx} style={{ padding: 8, backgroundColor: '#f5f5f5', borderRadius: 4, fontSize: 11, fontFamily: 'monospace' }}>
          <strong>{d.label}</strong> · {d.key}={d.value.toFixed(4)}, z={d.z.toFixed(2)} → {d.diagnosis}
        </div>
      ))}
    </div>
  )
}

// Row별 Custom Legend with 스타일 표현
function RowLegend({ items }: { items: Array<{ type: 'solid' | 'dashed' | 'band' | 'dotted'; color: string; label: string }> }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 11 }}>
      {items.map((item, idx) => (
        <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Line/Band Sample */}
          {item.type === 'solid' && (
            <svg width={40} height={16} style={{ display: 'block', minWidth: 40 }}>
              <line x1={0} y1={8} x2={40} y2={8} stroke={item.color} strokeWidth={2} />
            </svg>
          )}
          {item.type === 'dashed' && (
            <svg width={40} height={16} style={{ display: 'block', minWidth: 40 }}>
              <line x1={0} y1={8} x2={40} y2={8} stroke={item.color} strokeWidth={1} strokeDasharray={'4,3'} />
            </svg>
          )}
          {item.type === 'band' && (
            <div style={{ width: 40, height: 16, backgroundColor: 'rgba(100,100,100,0.1)', border: '1px solid #ccc', borderRadius: 2 }} />
          )}
          {item.type === 'dotted' && (
            <svg width={40} height={16} style={{ display: 'block', minWidth: 40 }}>
              <line x1={20} y1={0} x2={20} y2={16} stroke={item.color} strokeWidth={2} strokeDasharray={'2,2'} />
            </svg>
          )}
          <span style={{ color: '#333', fontSize: 11, whiteSpace: 'nowrap' }}>{item.label}</span>
        </div>
      ))}
    </div>
  )
}

// Row 1: SoH + Capacity Chart
function Row1Chart({ rows, cycle, bands, commonCycles, xMin, xMax, initialCapacity }: any) {
  const initCap = initialCapacity || 1

  const sohData = commonCycles.map((cyc: number) => {
    const row = rows.find((r: any) => r.cycle_num === cyc)
    return row?.soh ? row.soh * 100 : null
  })

  const capData = commonCycles.map((cyc: number) => {
    const row = rows.find((r: any) => r.cycle_num === cyc)
    return row?.capacity_mean ? (row.capacity_mean / initCap) * 100 : null
  })

  const q3Cap = commonCycles.map((c: number) => {
    const b = bands.capacity_mean?.find((bd: any) => bd.cycle === c)
    return b ? b.q3 : null
  })

  const q1Cap = commonCycles.map((c: number) => {
    const b = bands.capacity_mean?.find((bd: any) => bd.cycle === c)
    return b ? b.q1 : null
  })

  const medianCap = commonCycles.map((c: number) => {
    const b = bands.capacity_mean?.find((bd: any) => bd.cycle === c)
    return b ? b.median : null
  })

  const data = [
    { x: commonCycles, y: sohData, name: 'SoH (%)', type: 'scatter', mode: 'lines', line: { color: '#1f77b4', width: 2 }, showlegend: false, hovertemplate: '<b>SoH</b><br>Cycle: %{x}<br>Value: %{y:.2f}%<extra></extra>' },
    { x: commonCycles, y: capData, name: 'Capacity (%)', type: 'scatter', mode: 'lines', line: { color: '#ff7f0e', width: 2 }, showlegend: false, hovertemplate: '<b>Capacity</b><br>Cycle: %{x}<br>Value: %{y:.2f}%<extra></extra>' },
    { x: commonCycles, y: q3Cap, fill: 'tonexty', fillcolor: 'rgba(100,100,100,0.1)', line: { width: 0 }, showlegend: false, hovertemplate: 'Q3: %{y:.2f}<extra></extra>' },
    { x: commonCycles, y: q1Cap, fill: null, line: { width: 0 }, showlegend: false, hovertemplate: 'Q1: %{y:.2f}<extra></extra>' },
    { x: commonCycles, y: medianCap, type: 'scatter', mode: 'lines', line: { color: '#999', dash: 'dash', width: 1 }, showlegend: false, hovertemplate: 'Median: %{y:.2f}<extra></extra>' },
  ]

  const layout: any = {
    height: 220,
    margin: { l: 70, r: 20, t: 10, b: 40 },
    xaxis: { range: [xMin, xMax], showticklabels: false },
    yaxis: { title: 'SoH / Capacity (%)', titlefont: { size: 11 } },
    hovermode: 'x unified',
    showlegend: false,
    shapes: cycle ? [{
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: cycle,
      x1: cycle,
      y0: 0,
      y1: 1,
      line: { color: '#e74c3c', dash: 'dot', width: 2 }
    }] : []
  }

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} useResizeHandler style={{ width: '100%' }} />
}

// Row 2: DCR + Impedance Chart
function Row2Chart({ rows, cycle, bands, commonCycles, xMin, xMax }: any) {
  const dcrData = commonCycles.map((cyc: number) => {
    const row = rows.find((r: any) => r.cycle_num === cyc)
    return row?.dcr ?? null
  })

  const impData = commonCycles.map((cyc: number) => {
    const row = rows.find((r: any) => r.cycle_num === cyc)
    return row?.impedance_sum ?? null
  })

  const q3Imp = commonCycles.map((c: number) => bands.impedance_sum?.find((bd: any) => bd.cycle === c)?.q3 ?? null)
  const q1Imp = commonCycles.map((c: number) => bands.impedance_sum?.find((bd: any) => bd.cycle === c)?.q1 ?? null)
  const medianImp = commonCycles.map((c: number) => bands.impedance_sum?.find((bd: any) => bd.cycle === c)?.median ?? null)

  const data = [
    { x: commonCycles, y: dcrData, name: 'DCR (Ω)', type: 'scatter', mode: 'lines', line: { color: '#d62728', width: 2 }, showlegend: false, hovertemplate: '<b>DCR</b><br>Cycle: %{x}<br>Value: %{y:.4f}Ω<extra></extra>' },
    { x: commonCycles, y: impData, name: 'Impedance (Ω)', type: 'scatter', mode: 'lines', line: { color: '#2ca02c', width: 2 }, showlegend: false, hovertemplate: '<b>Impedance</b><br>Cycle: %{x}<br>Value: %{y:.4f}Ω<extra></extra>' },
    { x: commonCycles, y: q3Imp, fill: 'tonexty', fillcolor: 'rgba(100,100,100,0.1)', line: { width: 0 }, showlegend: false, hovertemplate: 'Q3: %{y:.4f}<extra></extra>' },
    { x: commonCycles, y: q1Imp, fill: null, line: { width: 0 }, showlegend: false, hovertemplate: 'Q1: %{y:.4f}<extra></extra>' },
    { x: commonCycles, y: medianImp, type: 'scatter', mode: 'lines', line: { color: '#999', dash: 'dash', width: 1 }, showlegend: false, hovertemplate: 'Median: %{y:.4f}<extra></extra>' },
  ]

  const layout: any = {
    height: 220,
    margin: { l: 70, r: 20, t: 10, b: 40 },
    xaxis: { range: [xMin, xMax], showticklabels: false },
    yaxis: { title: 'DCR / Impedance (Ω)', titlefont: { size: 11 } },
    hovermode: 'x unified',
    showlegend: false,
    shapes: cycle ? [{
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: cycle,
      x1: cycle,
      y0: 0,
      y1: 1,
      line: { color: '#e74c3c', dash: 'dot', width: 2 }
    }] : []
  }

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} useResizeHandler style={{ width: '100%' }} />
}

// Row 3: Temperature + Thermal Stress Chart
function Row3Chart({ rows, cycle, bands, commonCycles, xMin, xMax }: any) {
  const tempData = commonCycles.map((cyc: number) => {
    const row = rows.find((r: any) => r.cycle_num === cyc)
    return row?.temperature_mean ?? null
  })

  const thermalData = commonCycles.map((cyc: number) => {
    const row = rows.find((r: any) => r.cycle_num === cyc)
    return row?.thermal_stress ?? null
  })

  const q3Thermal = commonCycles.map((c: number) => bands.thermal_stress?.find((bd: any) => bd.cycle === c)?.q3 ?? null)
  const q1Thermal = commonCycles.map((c: number) => bands.thermal_stress?.find((bd: any) => bd.cycle === c)?.q1 ?? null)
  const medianThermal = commonCycles.map((c: number) => bands.thermal_stress?.find((bd: any) => bd.cycle === c)?.median ?? null)

  const data = [
    { x: commonCycles, y: tempData, name: 'Temperature (°C)', type: 'scatter', mode: 'lines', line: { color: '#9467bd', width: 2 }, yaxis: 'y', showlegend: false, hovertemplate: '<b>Temperature</b><br>Cycle: %{x}<br>Value: %{y:.2f}°C<extra></extra>' },
    { x: commonCycles, y: thermalData, name: 'Thermal stress', type: 'scatter', mode: 'lines', line: { color: '#e377c2', width: 2 }, yaxis: 'y2', showlegend: false, hovertemplate: '<b>Thermal stress</b><br>Cycle: %{x}<br>Value: %{y:.4f}<extra></extra>' },
    { x: commonCycles, y: q3Thermal, fill: 'tonexty', fillcolor: 'rgba(100,100,100,0.1)', line: { width: 0 }, yaxis: 'y2', showlegend: false, hovertemplate: 'Q3: %{y:.4f}<extra></extra>' },
    { x: commonCycles, y: q1Thermal, fill: null, line: { width: 0 }, yaxis: 'y2', showlegend: false, hovertemplate: 'Q1: %{y:.4f}<extra></extra>' },
    { x: commonCycles, y: medianThermal, type: 'scatter', mode: 'lines', line: { color: '#999', dash: 'dash', width: 1 }, yaxis: 'y2', showlegend: false, hovertemplate: 'Median: %{y:.4f}<extra></extra>' },
  ]

  const layout: any = {
    height: 220,
    margin: { l: 70, r: 70, t: 10, b: 40 },
    xaxis: { range: [xMin, xMax], title: 'Cycle', titlefont: { size: 11 } },
    yaxis: {
      title: 'Temperature (°C)',
      titlefont: { size: 11 },
      side: 'left'
    },
    yaxis2: {
      overlaying: 'y',
      side: 'right',
      title: 'Thermal stress index',
      titlefont: { size: 11 },
      tickfont: { size: 10 }
    },
    hovermode: 'x unified',
    showlegend: false,
    shapes: cycle ? [{
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: cycle,
      x1: cycle,
      y0: 0,
      y1: 1,
      line: { color: '#e74c3c', dash: 'dot', width: 2 }
    }] : []
  }

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} useResizeHandler style={{ width: '100%' }} />
}

// DegradationChart - 모든 메트릭과 expected band 표시
function DegradationChart({ title, rows, cycle, metrics, bands, normalizeCapacity = false, initialCapacity }: any) {
  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!chartRef.current || !rows.length) return

    if (!chart.current) {
      chart.current = echarts.init(chartRef.current)
    }

    const instance = chart.current
    const cycles = rows.map((r: any) => r.cycle_num)
    const seriesData: any[] = []
    const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    const metricNames: Record<string, string> = {
      'soh': 'SoH (%)',
      'capacity_mean': 'Capacity (% of initial)',
      'impedance_sum': 'Impedance sum (Ω)',
      'dcr': 'DCR (Ω)',
      'temperature_mean': 'Temp mean (°C)',
      'thermal_stress': 'Thermal stress',
    }

    // 각 메트릭별 시리즈 + expected band
    metrics.forEach((metric: string, idx: number) => {
      // 관측값 라인
      let values = rows.map((r: any) => {
        let v = r[metric] ?? null
        if (metric === 'capacity_mean' && normalizeCapacity && initialCapacity) {
          v = v !== null ? (v / initialCapacity) * 100 : null
        } else if (metric === 'soh') {
          v = v !== null ? v * 100 : null
        }
        return v
      })

      seriesData.push({
        name: metricNames[metric] || metric,
        type: 'line',
        data: values,
        smooth: false,
        lineStyle: { width: 1.5 },
        itemStyle: { borderWidth: 0 },
        color: colors[idx % colors.length],
        showSymbol: false,
      })

      // Expected band
      if (bands && bands[metric]) {
        const bandData = bands[metric]
        const q3s = cycles.map((c: number) => {
          const band = bandData.find((b: any) => b.cycle === c)
          return band ? band.q3 : null
        })
        const q1s = cycles.map((c: number) => {
          const band = bandData.find((b: any) => b.cycle === c)
          return band ? band.q1 : null
        })
        const medians = cycles.map((c: number) => {
          const band = bandData.find((b: any) => b.cycle === c)
          return band ? band.median : null
        })

        // Q3 line (invisible, for fill reference)
        seriesData.push({
          name: `Expected Q3 (${metric})`,
          type: 'line',
          data: q3s,
          lineStyle: { width: 0 },
          showSymbol: false,
          color: 'transparent',
        })

        // IQR fill
        seriesData.push({
          name: `Expected IQR (${metric})`,
          type: 'line',
          data: q1s,
          lineStyle: { width: 0 },
          fill: 'tonexty',
          areaStyle: { color: 'rgba(150,150,150,0.1)' },
          showSymbol: false,
        })

        // Median line
        seriesData.push({
          name: `Expected median (${metric})`,
          type: 'line',
          data: medians,
          lineStyle: { width: 1, type: 'dashed', color: '#999' },
          showSymbol: false,
        })
      }
    })

    // 현재 사이클 마커
    const markLineData = [{ xAxis: cycle, lineStyle: { color: 'red', type: 'dashed', width: 2 } }]

    const option: echarts.EChartsOption = {
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0, left: 0, textStyle: { fontSize: 10 }, orient: 'horizontal' },
      grid: { left: 40, right: 15, top: 20, bottom: 40 },
      xAxis: { type: 'category', data: cycles, boundaryGap: false },
      yAxis: { type: 'value', splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } } },
      series: seriesData as any,
      markLine: { data: markLineData, silent: true },
    }

    instance.setOption(option, true)
  }, [rows, cycle, metrics, bands, normalizeCapacity, initialCapacity])

  return <div ref={chartRef} style={{ width: '100%', height: 220 }} />
}

export default function App() {
  // Use relative URLs to leverage Vite proxy in dev and direct requests in production
  const PRECOMP_BASE = ''

  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  // ✅ 배터리 라벨 정의
  const batteryLabels: Record<string, string> = {
    'B0018': '✅ 정상 Residual 패턴',
    'B0042': '⚠️ 제외됨',
    'B0043': '📉 급격한 열화'
  }

  // ✅ viz_meta 파일 기준: 0.10, 0.20, 0.30, 0.40
  const rOptions = [0.10, 0.20, 0.30, 0.40]
  const [rRatio, setRRatio] = useState<number>(0.10)
  const tag = useMemo(() => toTag(rRatio), [rRatio])

  // 배터리별 실험 조건 (README 기반 + 현재 cycle 관측 전류)
  const experimentalConditions: Record<string, { temp: string; profile: string }> = {
    'B0018': { temp: 'room temperature', profile: '2A constant-current discharge, cutoff 2.5V' },
    'B0043': { temp: '4°C', profile: '1A / 4A load profile, cutoff 2.5V' }
  }

  const [battery, setBattery] = useState<string>('')
  const [batteries, setBatteries] = useState<string[]>([])

  // Features/overview
  const [rows, setRows] = useState<Row[]>([])
  const [cycle, setCycle] = useState<number>(0)
  const [meta, setMeta] = useState<any>(null)

  const [autoPlay, setAutoPlay] = useState(false)
  const [smoothed, setSmoothed] = useState(false)
  const [autoZoom, setAutoZoom] = useState(false)

  const [batch, setBatch] = useState<BatchPayload | null>(null)
  const [pack, setPack] = useState<SeriesPack | null>(null)
  const [preErr, setPreErr] = useState<string>('')
  const [inferring, setInferring] = useState(false)
  const [inferResult, setInferResult] = useState<any>(null)
  const [currentTab, setCurrentTab] = useState<'overview' | 'degradation' | 'compare' | 'explainability'>('overview')
  const [selectedBatteriesCompare, setSelectedBatteriesCompare] = useState<string[]>(['B0018', 'B0043'])

  // Degradation analysis
  const [featureBands, setFeatureBands] = useState<any>(null)
  const chartDegradationRef = useRef<HTMLDivElement>(null)
  const chartAnomalyRef = useRef<HTMLDivElement>(null)
  const chartDegradation = useRef<echarts.ECharts | null>(null)
  const chartAnomaly = useRef<echarts.ECharts | null>(null)

  // 1) Load precomputed batch json (no DB needed)
  useEffect(() => {
    setPreErr('')
    setBatch(null)
    setPack(null)

    const url = `${PRECOMP_BASE}/baseline-precomputed-batch/${tag}?baseline=${Date.now()}`
    console.log(`📦 Loading precomputed batch from: ${url}`)
    fetch(url)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`)
        return await r.json()
      })
      .then((d: BatchPayload) => {
        const items = Array.isArray(d?.items) ? d.items : []
        const bs = [...new Set(items.map((x) => x.battery_id))].sort().filter((b) => ['B0018', 'B0043'].includes(b))
        console.log(`✅ Loaded ${items.length} items, filtered batteries: ${bs}`)
        setBatch(d)
        setBatteries(bs)
        setBattery((prev) => {
          const nextBattery = (prev && bs.includes(prev) ? prev : (bs[0] ?? ''))
          console.log(`🔋 Setting battery: ${nextBattery} (previous: ${prev})`)
          return nextBattery
        })
      })
      .catch((e) => {
        setPreErr(`precomputed request failed: ${String(e?.message ?? e)}`)
        setBatteries([])
        setBattery('')
        console.error('❌ Failed to load precomputed:', e)
      })
  }, [tag])

  const batchItem = useMemo(() => {
    if (!batch?.items || !battery) return null
    return batch.items.find((x) => x.battery_id === battery) ?? null
  }, [batch, battery])

  useEffect(() => {
    if (!batchItem) {
      setPack(null)
      return
    }
    setPack(buildPack(batchItem))
  }, [batchItem])

  const metaLine = useMemo(() => {
    if (!batchItem) return ''
    const lenS = batchItem.len_s ?? batchItem.support?.cycle?.length ?? 0
    const lenQ = batchItem.len_q ?? batchItem.query?.cycle?.length ?? 0
    const qPos = batchItem.q_pos ?? (batchItem.query?.true_rul?.filter((v) => v > 0).length ?? 0)
    const rEff = batchItem.r_ratio_effective
    return `split=${batchItem.split_cycle} / support=${lenS} / query=${lenQ} / q_pos=${qPos}` + (rEff != null ? ` / r_eff=${Number(rEff).toFixed(3)}` : '')
  }, [batchItem])

  // 2) Load feature bands for degradation analysis - trigger on battery change OR degradation tab
  useEffect(() => {
    if (!battery) {
      console.log('⚠️ Battery is empty, skipping featureBands load')
      setFeatureBands(null)
      return
    }

    console.log(`🔄 Loading feature bands for battery: ${battery}`)
    const url = `/api/battery/${battery}/feature-bands`
    console.log(`📡 Request URL: ${url}`)

    // Set timeout to allow UI to proceed even if bands don't load
    const timeoutId = setTimeout(() => {
      console.warn('⏱️ Feature bands timeout (5s) - proceeding without bands data')
      setFeatureBands({ bands: null, timeout: true })
    }, 5000)

    axios
      .get(url, { timeout: 10000 })
      .then((r) => {
        clearTimeout(timeoutId)
        console.log('✅ Feature bands loaded:', r.data.battery, 'keys:', Object.keys(r.data))
        console.log('📊 Bands structure:', Object.keys(r.data.bands || {}))
        setFeatureBands(r.data)
      })
      .catch((e) => {
        clearTimeout(timeoutId)
        console.error('❌ Failed to load feature bands:', e.message)
        console.error('   URL:', url)
        if (e.response) console.error('   Status:', e.response.status)
        setFeatureBands({ bands: null, error: true })
      })
  }, [battery, currentTab])

  // 3) Features/meta from backend API (existing behavior)
  useEffect(() => {
    if (!battery) return
    axios.get(`${PRECOMP_BASE}/api/battery/${battery}/meta`).then((r) => setMeta(r.data)).catch(() => setMeta(null))

    // Include all feature columns (고정값 제외)
    const requiredCols = [
      'cycle_num', 'capacity_mean', 'soh', 'ambient_temp_c',
      'temperature_mean', 'temperature_measured_max', 'eff_c_rate', 'current_mean',
      'voltage_measured_mean', 'voltage_min', 'voltage_max', 'temp_rise_cycle',
      'thermal_stress', 'dcr', 'dcr_growth', 'lli', 'lam', 'impedance_sum',
      'impedance_growth', 'capacity_derivative', 'cap_vel',
      'dTdt_max', 'current_min', 'current_max'
    ].join(',')

    axios
      .get(`${PRECOMP_BASE}/api/battery/${battery}/cycles?stride=1&cols=${requiredCols}`)
      .then((r) => {
        const rs = Array.isArray(r.data?.rows) ? r.data.rows as Row[] : []
        if (!rs.length) throw new Error('cycles endpoint returned 0 rows')
        console.log(`✅ Loaded ${rs.length} cycles for ${battery}`)
        setRows(rs)
        setCycle(rs?.[0]?.cycle_num ?? rs?.[0]?.cycle ?? 0)
      })
      .catch((e) => {
        console.warn('⚠️ Primary cycles endpoint failed; trying degradation-monitoring rows:', e.message)
        axios
          .get(`${PRECOMP_BASE}/api/battery/${battery}/degradation-monitoring?r_ratio=${rRatio}`)
          .then((r2) => {
            const rs2 = Array.isArray(r2.data?.rows) ? r2.data.rows as Row[] : []
            if (!rs2.length) throw new Error('degradation-monitoring returned 0 rows')
            console.log(`✅ Loaded ${rs2.length} fallback cycles for ${battery}`)
            setRows(rs2)
            setCycle(rs2?.[0]?.cycle_num ?? rs2?.[0]?.cycle ?? 0)
          })
          .catch((e2) => {
            console.error('❌ Failed to load cycles from both endpoints:', e2.message)
            setRows([])
            setCycle(0)
          })
      })
  }, [battery, rRatio])




  const currentRow = useMemo(() => {
    if (!rows.length) return null
    const target = Number(cycle)
    let best: any = null
    let bestDist = Number.POSITIVE_INFINITY

    for (const r of rows) {
      const c = Number((r as any).cycle_num ?? (r as any).cycle ?? (r as any).anchor_cycle)
      if (!Number.isFinite(c)) continue
      const d = Math.abs(c - target)
      if (!best || d < bestDist) {
        best = r
        bestDist = d
      }
    }

    return best
  }, [rows, cycle])

  const currentExperimentCondition = useMemo(() => {
    const ambient = Number((currentRow as any)?.ambient_temp_c ?? (currentRow as any)?.ambient_temperature)
    const currentMinAbs = Math.abs(Number((currentRow as any)?.current_min))
    const currentMeanAbs = Math.abs(Number((currentRow as any)?.current_mean))
    const currentMaxAbs = Math.abs(Number((currentRow as any)?.current_max))
    const currentLoadMax = Math.abs(Number((currentRow as any)?.current_load_max ?? (currentRow as any)?.Current_load_max ?? (currentRow as any)?.Current_load))

    // Prefer explicit load column if present. Otherwise infer load class from measured current_min.
    // CSV has ambient_temp_c/current_min, so this updates live with the playback cycle without reading MAT at runtime.
    const loadSource = Number.isFinite(currentLoadMax) && currentLoadMax > 0.1
      ? currentLoadMax
      : Number.isFinite(currentMinAbs) && currentMinAbs > 0.1
        ? currentMinAbs
        : Number.isFinite(currentMeanAbs) && currentMeanAbs > 0.1
          ? currentMeanAbs
          : currentMaxAbs

    const loadClass = (() => {
      if (!Number.isFinite(loadSource)) return 'N/A'
      if (loadSource >= 0.7 && loadSource <= 1.3) return '1A'
      if (loadSource >= 1.5 && loadSource <= 2.5) return '2A'
      if (loadSource >= 3.5 && loadSource <= 4.5) return '4A'
      return `${loadSource.toFixed(1)}A`
    })()

    const temp = Number.isFinite(ambient)
      ? `약 ${ambient.toFixed(0)}°C`
      : experimentalConditions[battery]?.temp ?? 'N/A'

    return {
      temp,
      loadClass,
      cutoff: '2.5V',
      source: Number.isFinite(currentLoadMax) && currentLoadMax > 0.1 ? 'load column' : '|current_min|',
    }
  }, [currentRow, battery])



  const cRatePeak = useMemo(() => {
    if (!currentRow || !meta?.c_ref_ahr) return null
    const cm = currentRow.current_min
    if (cm == null) return null
    return Math.abs(Number(cm)) / Number(meta.c_ref_ahr)
  }, [currentRow, meta])

  // 모델 전체 성능 (precomputed)
  const modelMetrics = useMemo(() => {
    if (!batchItem) return { rmse: NaN, mae: NaN }
    const rmse = batchItem.metrics?.rmse ?? NaN
    const mae = batchItem.metrics?.mae ?? NaN
    return { rmse, mae }
  }, [batchItem])

  // 현재까지의 예측 구간 메트릭
  const currentMetrics = useMemo(() => {
    if (!pack || !pack!.cycles.length) return { rmse: NaN, mae: NaN, absError: NaN }

    const idx = pack!.cycles.indexOf(cycle)
    if (idx < 0) return { rmse: NaN, mae: NaN, absError: NaN }

    // 첫 사이클부터 현재 사이클까지의 오차
    const errors: number[] = []
    for (let i = 0; i <= idx; i++) {
      const pred = pack!.predRUL[i]
      const true_val = pack!.trueRUL[i]
      if (Number.isFinite(pred) && Number.isFinite(true_val)) {
        errors.push(pred - true_val)
      }
    }

    if (errors.length === 0) return { rmse: NaN, mae: NaN, absError: NaN }

    const mse = errors.reduce((sum, e) => sum + e * e, 0) / errors.length
    const rmse = Math.sqrt(mse)
    const mae = errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length
    const absError = Math.abs(errors[errors.length - 1])

    return { rmse, mae, absError }
  }, [pack, cycle])

  const handleShowPrecomputedBaseline = async () => {
    if (!battery) return

    try {
      const baselineUrl = `${PRECOMP_BASE}/baseline-precomputed/${battery}_viz_meta_${tag}.json?baseline=${Date.now()}`
      const response = await fetch(baselineUrl, { cache: 'no-store' })
      const contentType = response.headers.get('content-type') || ''

      if (!response.ok) {
        throw new Error(`Could not load precomputed baseline: HTTP ${response.status}`)
      }

      if (!contentType.includes('application/json')) {
        throw new Error(`Could not load precomputed baseline JSON: received ${contentType || 'unknown content-type'}`)
      }

      const baselineItem = await response.json().catch(() => null)

      if (!baselineItem) {
        throw new Error(`Could not parse precomputed baseline JSON`)
      }

      const nextPack = buildPack(baselineItem)
      setPack(nextPack)

      setBatch((prev) => {
        const oldItems = Array.isArray(prev?.items) ? prev!.items : []
        const idx = oldItems.findIndex((x) => x.battery_id === battery)
        const items = idx >= 0
          ? oldItems.map((x, i) => i === idx ? baselineItem : x)
          : [baselineItem, ...oldItems]

        return { ...(prev ?? { tag, r_ratio: rRatio }), items } as BatchPayload
      })

      const qPos = Number(baselineItem.q_pos)
      const c0 = Number.isFinite(qPos) && nextPack.cycles?.[qPos] != null
        ? Number(nextPack.cycles[qPos])
        : Number(baselineItem.current_cycle_effective ?? nextPack.cycles?.[0] ?? rows?.[0]?.cycle_num ?? 0)

      setCycle(Number.isFinite(c0) ? c0 : 0)
      setCurrentTab('overview')

      setInferResult({
        battery,
        r_ratio: rRatio,
        baselineRestored: true,
        source: 'precomputed_static_json',
      })
    } catch (e) {
      setInferResult({ error: `Failed to restore precomputed baseline: ${String(e)}` })
    }
  }


  const handleDownloadPredictionCsv = () => {
    if (!pack || !Array.isArray(pack.cycles) || pack.cycles.length === 0) return

    const stdArr = Array.isArray((pack as any).std) ? (pack as any).std : []
    const rows = pack.cycles.map((cycleValue: number, i: number) => {
      const trueRul = pack.trueRUL?.[i]
      const predRul = pack.predRUL?.[i]
      const std = stdArr?.[i]
      const absError =
        Number.isFinite(trueRul) && Number.isFinite(predRul)
          ? Math.abs(Number(predRul) - Number(trueRul))
          : ''

      return {
        battery_id: battery ?? '',
        r_ratio: Number.isFinite(rRatio) ? rRatio : '',
        cycle: cycleValue,
        true_rul: Number.isFinite(trueRul) ? trueRul : '',
        pred_rul: Number.isFinite(predRul) ? predRul : '',
        pred_std: Number.isFinite(std) ? std : '',
        uncertainty_2sigma: Number.isFinite(std) ? 2 * Number(std) : '',
        abs_error: absError,
      }
    })

    const headers = [
      'battery_id',
      'r_ratio',
      'cycle',
      'true_rul',
      'pred_rul',
      'pred_std',
      'uncertainty_2sigma',
      'abs_error',
    ]

    const escapeCsv = (value: unknown) => {
      const text = String(value ?? '')
      if (/[",\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`
      return text
    }

    const csv = [
      headers.join(','),
      ...rows.map((row) => headers.map((h) => escapeCsv((row as any)[h])).join(',')),
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const safeBattery = battery || 'battery'
    const safeRatio = Number.isFinite(rRatio) ? `r${rRatio.toFixed(2).replace('.', 'p')}` : 'r'
    a.href = url
    a.download = `${safeBattery}_${safeRatio}_prediction_result.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }


  // Reinference handler
  const handleReinfer = async () => {
    if (!battery) return
    setInferring(true)
    setAutoPlay(false)
    setInferResult(null)

    try {
      const runUrl = `${PRECOMP_BASE}/api/battery/${battery}/reinfer?r_ratio=${rRatio}&timeout=900`
      const response = await fetch(runUrl, { method: 'POST' })
      const data = await response.json().catch(() => null)

      if (!response.ok || data?.ok === false || data?.success === false) {
        const detail = data?.detail || data?.error || data?.stderr_tail || `HTTP ${response.status}`
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
      }

      const liveItem = data?.item
      if (!liveItem) throw new Error('live inference response did not include item JSON')

      const nextPack = buildPack(liveItem)
      setPack(nextPack)

      setBatch((prev) => {
        const oldItems = Array.isArray(prev?.items) ? prev!.items : []
        const idx = oldItems.findIndex((x) => x.battery_id === battery)
        const items = idx >= 0 ? oldItems.map((x, i) => i === idx ? liveItem : x) : [liveItem, ...oldItems]
        return { ...(prev ?? { tag, r_ratio: rRatio }), items } as BatchPayload
      })

      const qPos = Number(liveItem.q_pos)
      const c0 = Number.isFinite(qPos) && nextPack.cycles?.[qPos] != null
        ? Number(nextPack.cycles[qPos])
        : Number(liveItem.current_cycle_effective ?? nextPack.cycles?.[0] ?? rows?.[0]?.cycle_num ?? 0)
      setCycle(Number.isFinite(c0) ? c0 : 0)
      setCurrentTab('overview')

      setInferResult({
        ...data,
        battery,
        r_ratio: rRatio,
        sessionOnly: true,
        baselineOverwritten: false,
        overviewUpdated: true,
      })
    } catch (e) {
      setInferResult({ error: `Reinference failed: ${String(e)}` })
    } finally {
      setInferring(false)
    }
  }


  useEffect(() => {
    setAutoPlay(false)
  }, [battery, rRatio, currentTab])







  const playableCyclesV33 = React.useCallback((): number[] => {
    const fromRows = Array.isArray(rows)
      ? rows.map((r: any) => Number(r?.cycle_num ?? r?.cycle)).filter((v: any): v is number => Number.isFinite(v))
      : []
    const fromPack = Array.isArray((pack as any)?.cycles)
      ? (pack as any).cycles.map((c: any) => Number(c)).filter((v: any): v is number => Number.isFinite(v))
      : []
    const xs = (fromRows.length ? fromRows : fromPack).sort((a: number, b: number) => a - b)
    return Array.from(new Set(xs)) as number[]
  }, [rows, pack])

  const jumpCycleV33 = React.useCallback((mode: 'reset' | 'step' | 'play-start') => {
    const xs = playableCyclesV33()
    if (!xs.length) return false

    const cur = Number(cycle)
    const idx = xs.findIndex((c: number) => c === cur)

    if (mode === 'reset') {
      setAutoPlay(false)
      setCycle(xs[0])
      return true
    }

    if (mode === 'step') {
      setAutoPlay(false)
      setCycle(idx >= 0 && idx < xs.length - 1 ? xs[idx + 1] : xs[0])
      return true
    }

    if (!Number.isFinite(cur) || idx < 0 || idx >= xs.length - 1) {
      setCycle(xs[0])
    }
    setAutoPlay(true)
    return true
  }, [playableCyclesV33, cycle])

  useEffect(() => {
    if (!autoPlay) return

    const xs = playableCyclesV33()
    if (!xs.length) {
      setAutoPlay(false)
      return
    }

    const t = window.setInterval(() => {
      setCycle((prev: any) => {
        const cur = Number(prev)
        const idx = xs.findIndex((c: number) => c === cur)
        if (idx < 0) return xs[0]
        if (idx >= xs.length - 1) {
          window.setTimeout(() => setAutoPlay(false), 0)
          return xs[idx]
        }
        return xs[idx + 1]
      })
    }, 420)

    return () => window.clearInterval(t)
  }, [autoPlay, playableCyclesV33])

  useEffect(() => {
    setAutoPlay(false)
    const xs = playableCyclesV33()
    if (!xs.length) return
    const cur = Number(cycle)
    if (!Number.isFinite(cur) || !xs.includes(cur)) {
      setCycle(xs[0])
    }
  }, [battery, rRatio, playableCyclesV33])

  useEffect(() => {
    const onClick = (ev: MouseEvent) => {
      const target = ev.target as HTMLElement | null
      const btn = target?.closest?.('button') as HTMLButtonElement | null
      if (!btn) return

      const text = (btn.textContent || '').trim()
      const isPlay = /\b(Play|Playing|Pause|재생|일시정지)\b/i.test(text)
      const isStep = /\b(Step|Next|다음)\b/i.test(text)
      const isReset = /\b(Reset|리셋|초기화)\b/i.test(text)

      if (!isPlay && !isStep && !isReset) return

      const xs = playableCyclesV33()
      if (!xs.length) return

      ev.preventDefault()
      ev.stopPropagation()
      ev.stopImmediatePropagation()

      if (isReset) {
        jumpCycleV33('reset')
        return
      }

      if (isStep) {
        jumpCycleV33('step')
        return
      }

      if (isPlay) {
        if (autoPlay) {
          setAutoPlay(false)
        } else {
          jumpCycleV33('play-start')
        }
      }
    }

    document.addEventListener('click', onClick, true)
    return () => document.removeEventListener('click', onClick, true)
  }, [autoPlay, playableCyclesV33, jumpCycleV33])



  const playableCyclesV34 = React.useCallback((): number[] => {
    const fromRows = Array.isArray(rows)
      ? rows.map((r: any) => Number(r?.cycle_num ?? r?.cycle)).filter((v: any): v is number => Number.isFinite(v))
      : []
    const fromPack = Array.isArray((pack as any)?.cycles)
      ? (pack as any).cycles.map((c: any) => Number(c)).filter((v: any): v is number => Number.isFinite(v))
      : []
    const xs = (fromRows.length ? fromRows : fromPack).sort((a: number, b: number) => a - b)
    return Array.from(new Set(xs)) as number[]
  }, [rows, pack])

  const jumpCycleV34 = React.useCallback((mode: 'reset' | 'step' | 'play-start') => {
    const xs = playableCyclesV34()
    if (!xs.length) return false

    const cur = Number(cycle)
    const idx = xs.findIndex((c: number) => c === cur)

    if (mode === 'reset') {
      setAutoPlay(false)
      setCycle(xs[0])
      return true
    }

    if (mode === 'step') {
      setAutoPlay(false)
      setCycle(idx >= 0 && idx < xs.length - 1 ? xs[idx + 1] : xs[0])
      return true
    }

    if (!Number.isFinite(cur) || idx < 0 || idx >= xs.length - 1) {
      setCycle(xs[0])
    }
    setAutoPlay(true)
    return true
  }, [playableCyclesV34, cycle])

  useEffect(() => {
    if (!autoPlay) return

    const xs = playableCyclesV34()
    if (!xs.length) {
      setAutoPlay(false)
      return
    }

    const t = window.setInterval(() => {
      setCycle((prev: any) => {
        const cur = Number(prev)
        const idx = xs.findIndex((c: number) => c === cur)
        if (idx < 0) return xs[0]
        if (idx >= xs.length - 1) {
          window.setTimeout(() => setAutoPlay(false), 0)
          return xs[idx]
        }
        return xs[idx + 1]
      })
    }, 420)

    return () => window.clearInterval(t)
  }, [autoPlay, playableCyclesV34])

  // Battery/r_ratio changes should reset to a valid cycle. Tab changes should NOT stop playback.
  useEffect(() => {
    setAutoPlay(false)
    const xs = playableCyclesV34()
    if (!xs.length) return
    const cur = Number(cycle)
    if (!Number.isFinite(cur) || !xs.includes(cur)) {
      setCycle(xs[0])
    }
  }, [battery, rRatio, playableCyclesV34])

  useEffect(() => {
    const onClick = (ev: MouseEvent) => {
      const target = ev.target as HTMLElement | null
      const btn = target?.closest?.('button') as HTMLButtonElement | null
      if (!btn) return

      const text = (btn.textContent || '').trim()
      const isPlay = /\b(Play|Playing|Pause|재생|일시정지)\b/i.test(text)
      const isStep = /\b(Step|Next|다음)\b/i.test(text)
      const isReset = /\b(Reset|리셋|초기화)\b/i.test(text)

      if (!isPlay && !isStep && !isReset) return

      const xs = playableCyclesV34()
      if (!xs.length) return

      ev.preventDefault()
      ev.stopPropagation()
      ev.stopImmediatePropagation()

      if (isReset) {
        jumpCycleV34('reset')
        return
      }

      if (isStep) {
        jumpCycleV34('step')
        return
      }

      if (isPlay) {
        if (autoPlay) {
          setAutoPlay(false)
        } else {
          jumpCycleV34('play-start')
        }
      }
    }

    document.addEventListener('click', onClick, true)
    return () => document.removeEventListener('click', onClick, true)
  }, [autoPlay, playableCyclesV34, jumpCycleV34])

  // B0043 sidebar condition label fix. The source metadata still says "Variable"; display the NASA README condition.
  useEffect(() => {
    const t = window.setTimeout(() => {
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
      let node = walker.nextNode()
      while (node) {
        if (battery === 'B0043' && node.nodeValue?.includes('방전 전류: Variable')) {
          node.nodeValue = node.nodeValue.replace('방전 전류: Variable', '방전 전류: 1A / 4A variable load, cutoff 2.5V')
        }
        if (battery === 'B0043' && node.nodeValue?.includes('Discharge Current: Variable')) {
          node.nodeValue = node.nodeValue.replace('Discharge Current: Variable', 'Discharge Current: 1A / 4A variable load, cutoff 2.5V')
        }
        node = walker.nextNode()
      }
    }, 30)
    return () => window.clearTimeout(t)
  }, [battery, currentTab, rRatio])

  // Fill Overview Signals from the nearest observed cycle row. This fixes stale "실행 전" cards while playback is moving.
  useEffect(() => {
    const cycleNum = Number(cycle)
    const dataRows = Array.isArray(rows) ? rows : []
    if (!dataRows.length || !Number.isFinite(cycleNum)) return

    let nearest: any = null
    let best = Number.POSITIVE_INFINITY
    for (const r of dataRows) {
      const c = Number((r as any)?.cycle_num ?? (r as any)?.cycle)
      if (!Number.isFinite(c)) continue
      const d = Math.abs(c - cycleNum)
      if (d < best) {
        nearest = r
        best = d
      }
    }
    if (!nearest) return

    const val = (keys: string[]) => {
      for (const k of keys) {
        const x = Number((nearest as any)?.[k])
        if (Number.isFinite(x)) return x
      }
      return NaN
    }

    const fmt = (x: number, digits = 3) => Number.isFinite(x) ? x.toFixed(digits) : null
    const pct = (x: number, digits = 3) => Number.isFinite(x) ? x.toFixed(digits) + ' %' : null

    const updates: Array<[string, string | null]> = [
      ['capacity_mean', fmt(val(['capacity_mean', 'capacity']), 4)],
      ['soh', pct(val(['soh', 'soh_pct', 'cap_pct']), 3)],
      ['capacity_derivative', fmt(val(['capacity_derivative', 'cap_derivative']), 5)],
      ['cap_vel', fmt(val(['cap_vel', 'capacity_velocity']), 5)],
      ['eff_c_rate', fmt(val(['eff_c_rate', 'c_rate_eff']), 3)],
      ['current_mean', fmt(val(['current_mean', 'current_measured_mean']), 4)],
      ['c_rate_peak', fmt(val(['c_rate_peak', 'crate_peak']), 3)],
      ['voltage_measured_mean', fmt(val(['voltage_measured_mean', 'voltage_mean']), 3)],
      ['voltage_min', fmt(val(['voltage_min', 'voltage_measured_min']), 3)],
      ['voltage_max', fmt(val(['voltage_max', 'voltage_measured_max']), 3)],
      ['temperature_mean', fmt(val(['temperature_mean', 'temp_mean']), 2)],
      ['temperature_measured_max', fmt(val(['temperature_measured_max', 'temperature_max', 'temp_max']), 2)],
      ['temp_rise_cycle', fmt(val(['temp_rise_cycle', 'temperature_rise_cycle']), 3)],
      ['thermal_stress', fmt(val(['thermal_stress']), 3)],
      ['dT/dt max', fmt(val(['dtdt_max', 'dTdt_max', 'temp_dtdt_max']), 3)],
      ['dcr', fmt(val(['dcr', 'DCR', 'Re']), 4)],
      ['dcr_growth', fmt(val(['dcr_growth', 'dcr_growth_log_ratio']), 4)],
      ['LLI', fmt(val(['lli', 'LLI']), 4)],
      ['LAM', fmt(val(['lam', 'LAM']), 4)],
      ['impedance_sum', fmt(val(['impedance_sum']), 4)],
      ['impedance_growth', fmt(val(['impedance_growth']), 4)],
    ]

    const apply = () => {
      const all = Array.from(document.querySelectorAll('div')) as HTMLElement[]

      for (const [label, display] of updates) {
        if (!display) continue
        const candidates = all
          .filter((el) => {
            const text = (el.textContent || '').trim()
            const rect = el.getBoundingClientRect()
            return text.startsWith(label) && rect.width > 55 && rect.width < 260 && rect.height > 25 && rect.height < 130
          })
          .sort((a, b) => {
            const ar = a.getBoundingClientRect()
            const br = b.getBoundingClientRect()
            return (ar.width * ar.height) - (br.width * br.height)
          })

        const card = candidates[0]
        if (!card) continue

        const children = Array.from(card.children) as HTMLElement[]
        const valueNode = children.find((ch) => {
          const t = (ch.textContent || '').trim()
          return t === '실행 전' || t === '예측 전' || t === '평가 전' || /^-?\d/.test(t)
        }) || children[1]

        if (valueNode) {
          valueNode.textContent = display
        }
      }
    }

    const t = window.setTimeout(apply, 40)
    return () => window.clearTimeout(t)
  }, [cycle, rows, currentTab, battery, rRatio])






  // HOTFIX_V35_SINGLE_PLAYBACK_AND_SIGNAL_SYNC
  // Single controller: one parent cycle state keeps Overview / Degradation / Compare / Explainability in sync.
  const playableCyclesV35 = React.useCallback((): number[] => {
    const fromRows = Array.isArray(rows)
      ? rows.map((r: any) => Number(r?.cycle_num ?? r?.cycle)).filter((v: any): v is number => Number.isFinite(v))
      : []
    const fromPack = Array.isArray((pack as any)?.cycles)
      ? (pack as any).cycles.map((c: any) => Number(c)).filter((v: any): v is number => Number.isFinite(v))
      : []
    const xs = (fromRows.length ? fromRows : fromPack).sort((a: number, b: number) => a - b)
    return Array.from(new Set(xs)) as number[]
  }, [rows, pack])

  const jumpCycleV35 = React.useCallback((mode: 'reset' | 'step' | 'play-start') => {
    const xs = playableCyclesV35()
    if (!xs.length) return false

    const cur = Number(cycle)
    const idx = xs.findIndex((c: number) => c === cur)

    if (mode === 'reset') {
      setAutoPlay(false)
      setCycle(xs[0])
      return true
    }

    if (mode === 'step') {
      setAutoPlay(false)
      setCycle(idx >= 0 && idx < xs.length - 1 ? xs[idx + 1] : xs[0])
      return true
    }

    if (!Number.isFinite(cur) || idx < 0 || idx >= xs.length - 1) {
      setCycle(xs[0])
    }
    setAutoPlay(true)
    return true
  }, [playableCyclesV35, cycle])

  useEffect(() => {
    if (!autoPlay) return

    const xs = playableCyclesV35()
    if (!xs.length) {
      setAutoPlay(false)
      return
    }

    const t = window.setInterval(() => {
      setCycle((prev: any) => {
        const cur = Number(prev)
        const idx = xs.findIndex((c: number) => c === cur)
        if (idx < 0) return xs[0]
        if (idx >= xs.length - 1) {
          window.setTimeout(() => setAutoPlay(false), 0)
          return xs[idx]
        }
        return xs[idx + 1]
      })
    }, 420)

    return () => window.clearInterval(t)
  }, [autoPlay, playableCyclesV35])

  // Battery/rRatio changes realign the cycle. Tab changes intentionally do not stop playback.
  useEffect(() => {
    setAutoPlay(false)
    const xs = playableCyclesV35()
    if (!xs.length) return
    const cur = Number(cycle)
    if (!Number.isFinite(cur) || !xs.includes(cur)) setCycle(xs[0])
  }, [battery, rRatio, playableCyclesV35])

  useEffect(() => {
    const onClick = (ev: MouseEvent) => {
      const target = ev.target as HTMLElement | null
      const btn = target?.closest?.('button') as HTMLButtonElement | null
      if (!btn) return

      const text = (btn.textContent || '').trim()
      const isPlay = /\b(Play|Playing|Pause|재생|일시정지)\b/i.test(text)
      const isStep = /\b(Step|Next|다음)\b/i.test(text)
      const isReset = /\b(Reset|리셋|초기화)\b/i.test(text)
      if (!isPlay && !isStep && !isReset) return

      const xs = playableCyclesV35()
      if (!xs.length) return

      ev.preventDefault()
      ev.stopPropagation()
      ev.stopImmediatePropagation()

      if (isReset) {
        jumpCycleV35('reset')
        return
      }

      if (isStep) {
        jumpCycleV35('step')
        return
      }

      if (autoPlay) setAutoPlay(false)
      else jumpCycleV35('play-start')
    }

    document.addEventListener('click', onClick, true)
    return () => document.removeEventListener('click', onClick, true)
  }, [autoPlay, playableCyclesV35, jumpCycleV35])

  // Display README-backed condition for B0043 even if the raw metadata still says "Variable".
  useEffect(() => {
    const t = window.setTimeout(() => {
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
      let node = walker.nextNode()
      while (node) {
        if (battery === 'B0043' && node.nodeValue?.includes('방전 전류: Variable')) {
          node.nodeValue = node.nodeValue.replace('방전 전류: Variable', '방전 전류: 1A / 4A load profile, cutoff 2.5V')
        }
        if (battery === 'B0043' && node.nodeValue?.includes('Discharge Current: Variable')) {
          node.nodeValue = node.nodeValue.replace('Discharge Current: Variable', 'Discharge Current: 1A / 4A load profile, cutoff 2.5V')
        }
        node = walker.nextNode()
      }
    }, 30)
    return () => window.clearTimeout(t)
  }, [battery, currentTab, rRatio, cycle])

  // Keep observed Overview Signals synced with the current cycle.
  useEffect(() => {
    const cycleNum = Number(cycle)
    const dataRows = Array.isArray(rows) ? rows : []
    if (!dataRows.length || !Number.isFinite(cycleNum)) return

    let nearest: any = null
    let best = Number.POSITIVE_INFINITY
    for (const r of dataRows) {
      const c = Number((r as any)?.cycle_num ?? (r as any)?.cycle)
      if (!Number.isFinite(c)) continue
      const d = Math.abs(c - cycleNum)
      if (d < best) {
        nearest = r
        best = d
      }
    }
    if (!nearest) return

    const get = (keys: string[]) => {
      for (const k of keys) {
        const x = Number((nearest as any)?.[k])
        if (Number.isFinite(x)) return x
      }
      return NaN
    }

    const fmt = (x: number, digits = 3) => Number.isFinite(x) ? x.toFixed(digits) : null
    const pct = (x: number, digits = 3) => Number.isFinite(x) ? x.toFixed(digits) + ' %' : null

    const updates: Array<[string, string | null]> = [
      ['capacity_mean', fmt(get(['capacity_mean', 'capacity']), 4)],
      ['soh', pct(get(['soh', 'soh_pct', 'cap_pct']), 3)],
      ['capacity_derivative', fmt(get(['capacity_derivative', 'cap_derivative']), 5)],
      ['cap_vel', fmt(get(['cap_vel', 'capacity_velocity']), 5)],
      ['eff_c_rate', fmt(get(['eff_c_rate', 'c_rate_eff']), 3)],
      ['current_mean', fmt(get(['current_mean', 'current_measured_mean']), 4)],
      ['c_rate_peak', fmt(get(['c_rate_peak', 'crate_peak']), 3)],
      ['voltage_measured_mean', fmt(get(['voltage_measured_mean', 'voltage_mean']), 3)],
      ['voltage_min', fmt(get(['voltage_min', 'voltage_measured_min']), 3)],
      ['voltage_max', fmt(get(['voltage_max', 'voltage_measured_max']), 3)],
      ['temperature_mean', fmt(get(['temperature_mean', 'temp_mean']), 2)],
      ['temperature_measured_max', fmt(get(['temperature_measured_max', 'temperature_max', 'temp_max']), 2)],
      ['temp_rise_cycle', fmt(get(['temp_rise_cycle', 'temperature_rise_cycle']), 3)],
      ['thermal_stress', fmt(get(['thermal_stress']), 3)],
      ['dT/dt max', fmt(get(['dtdt_max', 'dTdt_max', 'temp_dtdt_max']), 3)],
      ['dcr', fmt(get(['dcr', 'DCR', 'Re']), 4)],
      ['dcr_growth', fmt(get(['dcr_growth', 'dcr_growth_log_ratio']), 4)],
      ['LLI', fmt(get(['lli', 'LLI']), 4)],
      ['LAM', fmt(get(['lam', 'LAM']), 4)],
      ['impedance_sum', fmt(get(['impedance_sum']), 4)],
      ['impedance_growth', fmt(get(['impedance_growth']), 4)],
    ]

    const apply = () => {
      const cards = Array.from(document.querySelectorAll('div')) as HTMLElement[]

      for (const [label, display] of updates) {
        if (!display) continue

        const candidate = cards
          .filter((el) => {
            const text = (el.textContent || '').trim()
            const rect = el.getBoundingClientRect()
            return text.startsWith(label) && rect.width > 55 && rect.width < 280 && rect.height > 25 && rect.height < 140
          })
          .sort((a, b) => {
            const ar = a.getBoundingClientRect()
            const br = b.getBoundingClientRect()
            return (ar.width * ar.height) - (br.width * br.height)
          })[0]

        if (!candidate) continue

        const children = Array.from(candidate.children) as HTMLElement[]
        const valueNode = children.find((ch) => {
          const t = (ch.textContent || '').trim()
          return t === '실행 전' || t === '예측 전' || t === '평가 전' || /^-?\d/.test(t)
        }) || children[1]

        if (valueNode) valueNode.textContent = display
      }
    }

    const t = window.setTimeout(apply, 40)
    return () => window.clearTimeout(t)
  }, [cycle, rows, currentTab, battery, rRatio])


  // HOTFIX_V36_KEEP_PLAYING_ACROSS_TAB_SWITCH
  // If playback is running, switching tabs must not stop it.
  // Existing older effects may still call setAutoPlay(false) on currentTab changes,
  // so this captures the tab click intent and resumes playback after the tab switch render.
  useEffect(() => {
    const onPointerDown = (ev: PointerEvent) => {
      if (!autoPlay) return

      const target = ev.target as HTMLElement | null
      const el = target?.closest?.('button, [role="tab"], div') as HTMLElement | null
      if (!el) return

      const text = (el.textContent || '').trim()
      const isTab =
        text === 'Overview' ||
        text === 'Degradation' ||
        text === 'Compare' ||
        text === 'Explainability'

      if (isTab) {
        ;(window as any).__batteryRulResumePlaybackAfterTabV36 = true
      }
    }

    document.addEventListener('pointerdown', onPointerDown, true)
    return () => document.removeEventListener('pointerdown', onPointerDown, true)
  }, [autoPlay])

  useEffect(() => {
    if (!(window as any).__batteryRulResumePlaybackAfterTabV36) return

    const t = window.setTimeout(() => {
      ;(window as any).__batteryRulResumePlaybackAfterTabV36 = false
      setAutoPlay(true)
    }, 80)

    return () => window.clearTimeout(t)
  }, [currentTab])



  // 3) Render chart
  useEffect(() => {
    if (!chartRef.current || !pack) return

    // Destroy old chart instance when tab changes
    if (chart.current) {
      try {
        chart.current.dispose()
      } catch (e) {}
      chart.current = null
    }

    // Initialize chart fresh
    chart.current = echarts.init(chartRef.current, undefined, { renderer: 'canvas' })

    // Resize chart when tab changes - with multiple retry attempts
    setTimeout(() => chart.current?.resize(), 0)
    setTimeout(() => chart.current?.resize(), 100)
    setTimeout(() => chart.current?.resize(), 300)

    const alpha = 0.25
    const pred = smoothed
      ? pack!.predRUL.reduce((acc: number[], v, i) => {
          if (i === 0) return [v]
          return [...acc, alpha * v + (1 - alpha) * acc[i - 1]]
        }, [])
      : pack!.predRUL

    const upper = pred.map((p, i) => p + 2 * (pack!.std[i] ?? 0))
    const lower = pred.map((p, i) => p - 2 * (pack!.std[i] ?? 0))

    // Convert to pairs, treating NaN/Inf as null for proper line breaking
    const supportPairs = pack!.supportCycles.map((c, i) =>
      [c, Number.isFinite(pack.supportRUL[i]) ? pack.supportRUL[i] : null] as any
    )
    const predPairs = pack!.cycles.map((c, i) =>
      [c, Number.isFinite(pred[i]) ? pred[i] : null] as any
    )
    const truePairs = pack!.cycles.map((c, i) =>
      [c, Number.isFinite(pack!.trueRUL[i]) ? pack!.trueRUL[i] : null] as any
    )
    const upPairs = pack!.cycles.map((c, i) =>
      [c, Number.isFinite(upper[i]) ? upper[i] : null] as any
    )
    const loPairs = pack!.cycles.map((c, i) =>
      [c, Number.isFinite(lower[i]) ? lower[i] : null] as any
    )

    // selected markers (closest query cycle to current cycle)
    let selPred: [number, number] | null = null
    let selTrue: [number, number] | null = null
    if (pack!.cycles.length) {
      let best = 0
      for (let i = 1; i < pack!.cycles.length; i++) {
        if (Math.abs(pack!.cycles[i] - cycle) < Math.abs(pack!.cycles[best] - cycle)) best = i
      }
      selPred = [pack!.cycles[best], pred[best]]
      selTrue = [pack!.cycles[best], pack!.trueRUL[best]]
    }

    const split = pack.split
    const xMin = autoZoom && split != null ? split - 10 : undefined
    const xMax = autoZoom && split != null ? split + 45 : undefined

    chart.current.setOption(
      {
        tooltip: { trigger: 'axis' },
        legend: { data: ['Support', 'True RUL', 'Pred RUL', '+2σ', '-2σ', 'Selected cycle (pred)', 'Selected cycle (true)'] },
        grid: { left: 55, right: 20, top: 30, bottom: 45 },
        xAxis: { type: 'value', name: 'Cycle', min: xMin, max: xMax },
        yAxis: { type: 'value', name: 'RUL (cycles)' },
        series: [
          { name: 'Support', type: 'scatter', data: supportPairs, symbolSize: 6, itemStyle: { color: '#5470c6' } },
          { name: 'True RUL', type: 'line', data: truePairs, showSymbol: false, lineStyle: { width: 2, type: 'dashed', color: '#91cc75' }, smooth: false },
          { name: 'Pred RUL', type: 'line', data: predPairs, showSymbol: false, lineStyle: { width: 2, color: '#fac858' }, smooth: false },
          { name: '+2σ', type: 'line', data: upPairs, showSymbol: false, lineStyle: { width: 1, type: 'dotted', color: '#ee6666' }, smooth: false },
          { name: '-2σ', type: 'line', data: loPairs, showSymbol: false, lineStyle: { width: 1, type: 'dotted', color: '#ee6666' }, smooth: false },
          ...(selPred ? [{ name: 'Selected cycle (pred)', type: 'scatter', data: [selPred], symbolSize: 10, itemStyle: { color: '#73c0de' } }] : []),
          ...(selTrue ? [{ name: 'Selected cycle (true)', type: 'scatter', data: [selTrue], symbolSize: 10, itemStyle: { color: '#ee6666' } }] : []),
        ],
        markLine: {
          symbol: ['none', 'none'],
          label: { show: true, position: 'end' },
          data: [...(split != null ? [{ xAxis: split, label: { formatter: 'split' }, lineStyle: { color: '#999' } }] : [])],
        },
      },
      true
    )
  }, [pack, smoothed, autoZoom, cycle, currentTab])

  return (
    <div style={{ fontFamily: 'ui-sans-serif, system-ui', padding: 16, display: 'grid', gridTemplateColumns: '280px 1fr', gap: 16, height: '100vh' }}>
      {/* 좌측: 사이드바 */}
      <div style={{ overflowY: 'auto', paddingRight: 8 }}>
        <h2>Battery RUL Dashboard</h2>

        <div style={{ marginBottom: 12 }}>
          <label>Test Battery</label>
          <br />
          <select value={battery} onChange={(e) => setBattery(e.target.value)} style={{ width: '100%', padding: 8 }}>
            {batteries.map((b) => (
              <option key={b} value={b}>
                {b} {batteryLabels[b] ? `— ${batteryLabels[b]}` : ''}
              </option>
            ))}
          </select>

          {/* 실험 조건: 현재 playback cycle 기준 */}
          {battery && experimentalConditions[battery] && (
            <div style={{
              marginTop: 8,
              padding: 10,
              backgroundColor: '#f5f7fb',
              borderRadius: 6,
              border: '1px solid #dfe7f3',
              fontSize: 12,
              lineHeight: 1.45
            }}>
              <div style={{ color: '#555', marginBottom: 6, fontWeight: 800 }}>
                실험 조건
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 6,
                marginBottom: 7
              }}>
                <div style={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: 5, padding: 6 }}>
                  <div style={{ color: '#64748b', fontSize: 10, fontWeight: 800 }}>온도</div>
                  <div style={{ color: '#111827', fontSize: 14, fontWeight: 900 }}>{currentExperimentCondition.temp}</div>
                </div>
                <div style={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: 5, padding: 6 }}>
                  <div style={{ color: '#64748b', fontSize: 10, fontWeight: 800 }}>방전 load</div>
                  <div style={{ color: '#111827', fontSize: 14, fontWeight: 900 }}>{currentExperimentCondition.loadClass}</div>
                </div>
              </div>
              <div style={{ color: '#333', marginBottom: 6 }}>
                <strong>cutoff:</strong> {currentExperimentCondition.cutoff}
              </div>
              
            </div>
          )}
        </div>

        <div style={{ marginBottom: 6 }}>
          <label>r_ratio (precomputed)</label>
          <br />
          <select value={rRatio} onChange={(e) => setRRatio(parseFloat(e.target.value))} style={{ width: '100%', padding: 8 }}>
            {rOptions.map((v) => (
              <option key={v} value={v}>
                {v.toFixed(2)}
              </option>
            ))}
          </select>
        </div>

        <div style={{ marginBottom: 12, fontSize: 12, opacity: 0.85, fontFamily: 'monospace' }}>
          {preErr ? `RUL precomputed 로딩 실패: ${preErr}` : metaLine || '—'}
        </div>

        {/* ===== Playback Controls ===== */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>▶️ Playback</div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
            <button
              onClick={() => {
                if (!rows.length) return
                if (autoPlay) {
                  setAutoPlay(false)
                  return
                }
                const cycleList = rows.map((r) => Number(r.cycle_num)).filter(Number.isFinite).sort((a, b) => a - b)
                if (!cycleList.length) return
                const cur = Number(cycle)
                if (!cycleList.includes(cur) || cur >= cycleList[cycleList.length - 1]) setCycle(cycleList[0])
                setAutoPlay(true)
              }}
              style={{
                padding: '6px 12px',
                backgroundColor: autoPlay ? '#4caf50' : '#2196f3',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: 600
              }}
            >
              {autoPlay ? '⏸️ Pause' : '▶️ Play'}
            </button>
            <button
              onClick={() => {
                const cycleList = (rows.length ? rows.map((r) => Number(r.cycle_num)) : (pack?.cycles ?? []).map((c) => Number(c)))
                  .filter(Number.isFinite)
                  .sort((a, b) => a - b)
                if (!cycleList.length) return
                const idx = cycleList.findIndex((c) => c === Number(cycle))
                setCycle(idx >= 0 && idx < cycleList.length - 1 ? cycleList[idx + 1] : cycleList[0])
                setAutoPlay(false)
              }}
              style={{
                padding: '6px 12px',
                backgroundColor: '#ff9800',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: 600
              }}
            >
              ⏭️ Step
            </button>
            <button
              onClick={() => {
                const cycleList = (rows.length ? rows.map((r) => Number(r.cycle_num)) : (pack?.cycles ?? []).map((c) => Number(c)))
                  .filter(Number.isFinite)
                  .sort((a, b) => a - b)
                if (cycleList.length) setCycle(cycleList[0])
                setAutoPlay(false)
              }}
              style={{
                padding: '6px 12px',
                backgroundColor: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: 600
              }}
            >
              ⏮️ Reset
            </button>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <label style={{ fontSize: 12 }}>
              <input type="checkbox" checked={smoothed} onChange={(e) => setSmoothed(e.target.checked)} /> Smoothed
            </label>
            <label style={{ fontSize: 12 }}>
              <input type="checkbox" checked={autoZoom} onChange={(e) => setAutoZoom(e.target.checked)} /> Auto-zoom
            </label>
          </div>
        </div>

        {/* ===== Model Reinference ===== */}
        <hr />
        <div style={{ marginBottom: 12 }}>
          <button
            onClick={handleReinfer}
            disabled={!battery || inferring}
            style={{
              padding: '10px 16px',
              backgroundColor: inferring ? '#ccc' : '#1976d2',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: inferring ? 'not-allowed' : 'pointer',
              fontSize: 13,
              fontWeight: 600,
              width: '100%',
              whiteSpace: 'normal',
              lineHeight: 1.4
            }}
          >
            {inferring
              ? '⏳ Running... ~40s'
              : battery
              ? (
                  <>
                    <div>🧠 Initialize & Reinference</div>
                    <div style={{ fontSize: 11, opacity: 0.85 }}>{battery}, r={rRatio.toFixed(2)}</div>
                  </>
                )
              : '🧠 Initialize & Reinference'}
          </button>

          <button
            type="button"
            onClick={handleDownloadPredictionCsv}
            disabled={!pack || !Array.isArray(pack.cycles) || pack.cycles.length === 0}
            style={{
              marginTop: 8,
              padding: '8px 12px',
              backgroundColor: '#ffffff',
              color: '#1976d2',
              border: '1px solid #1976d2',
              borderRadius: 6,
              cursor: !pack || !Array.isArray(pack.cycles) || pack.cycles.length === 0 ? 'not-allowed' : 'pointer',
              fontSize: 12,
              fontWeight: 600,
              width: '100%'
            }}
          >
            Download current prediction CSV
          </button>

          {inferResult && (
            <div style={{
              marginTop: 10,
              padding: 8,
              borderRadius: 4,
              fontSize: 11,
              fontFamily: 'monospace',
              backgroundColor: inferResult.error ? '#ffebee' : '#e8f5e9',
              color: inferResult.error ? '#c62828' : '#2e7d32',
              border: `1px solid ${inferResult.error ? '#ef5350' : '#66bb6a'}`
            }}>
              {inferResult.error
                ? `❌ ${inferResult.error}`
                : inferResult.baselineRestored
                ? (
                    <div className="space-y-1">
                      <div>
                        ✅ {inferResult.battery} r={inferResult.r_ratio?.toFixed(2)} — Precomputed baseline result restored
                      </div>
                    </div>
                  )
                : (
                    <div className="space-y-1">
                      <div>
                        ✅ {inferResult.battery} r={inferResult.r_ratio?.toFixed(2)} — Live reinference complete
                      </div>
                      <div className="text-xs opacity-80">
                        Latest server-session result is now displayed.
                      </div>
                      <button
                        type="button"
                        onClick={handleShowPrecomputedBaseline}
                        style={{
                          marginTop: 6,
                          padding: '6px 10px',
                          backgroundColor: '#ffffff',
                          color: '#2e7d32',
                          border: '1px solid #66bb6a',
                          borderRadius: 4,
                          cursor: 'pointer',
                          fontSize: 11,
                          fontWeight: 600
                        }}
                      >
                        Show precomputed baseline result
                      </button>
                    </div>
                  )}
            </div>
          )}
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>Cycle</label>
          <br />
          <input
            type="range"
            min={rows?.[0]?.cycle_num ?? 0}
            max={rows?.[rows.length - 1]?.cycle_num ?? 0}
            value={cycle}
            onChange={(e) => setCycle(parseInt(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ fontSize: 14, fontWeight: 600 }}>{cycle}</div>
        </div>

        {/* ===== RUL & Health (좌측에 배치) ===== */}
        <hr />
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, opacity: 0.7, fontWeight: 600, marginBottom: 8 }}>🔋 RUL & Health</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            <Card title="Pred RUL" value={pack?.predRUL?.[pack!.cycles.indexOf(cycle)] ?? 'N/A'} suffix="" compact />
            <Card title="True RUL" value={pack?.trueRUL?.[pack!.cycles.indexOf(cycle)] ?? 'N/A'} suffix="" compact />
            <Card title="SoH %" value={(currentRow?.soh ?? 0) * 100} suffix="%" compact />
          </div>
        </div>

        {/* ===== Model Performance ===== */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, opacity: 0.7, fontWeight: 600, marginBottom: 6 }}>📊 모델 예측 오차 지표</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 8 }}>
            <Card title="RMSE" value={modelMetrics.rmse} suffix="cycles" compact />
            <Card title="MAE" value={modelMetrics.mae} suffix="cycles" compact />
          </div>

          <div style={{ fontSize: 11, opacity: 0.7, fontWeight: 600, marginBottom: 6 }}>⏱️ 현재 예측 성능</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            <Card title="RMSE (current cycles)" value={currentMetrics.rmse} suffix="cycles" compact />
            <Card title="MAE (current cycles)" value={currentMetrics.mae} suffix="cycles" compact />
            <Card title="Current Abs Error" value={currentMetrics.absError} suffix="cycles" compact />
          </div>
        </div>

      </div>

      {/* 우측: 차트 + Overview (세로 레이아웃) */}
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>

        {/* ===== 탭 버튼들 ===== */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12, borderBottom: '1px solid #eee', paddingBottom: 8 }}>
          {(['overview', 'degradation', 'compare', 'explainability'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setCurrentTab(tab)}
              style={{
                padding: '6px 14px',
                backgroundColor: currentTab === tab ? '#1976d2' : '#f5f5f5',
                color: currentTab === tab ? 'white' : '#333',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: 600,
                transition: 'all 0.2s'
              }}
            >
              {tab === 'overview' && 'Overview'}
              {tab === 'degradation' && 'Degradation'}
              {tab === 'compare' && 'Compare'}
              {tab === 'explainability' && 'Explainability'}
            </button>
          ))}
        </div>

        {/* ===== Chart (flex 1로 남은 공간 다 사용) - Overview 탭에만 표시 ===== */}
        {currentTab === 'overview' && (
        <div style={{ flex: 1, minHeight: 300, marginBottom: 2, display: 'flex', flexDirection: 'column' }}>
          <div ref={chartRef} style={{ width: '100%', flex: 1, border: '1px solid #eee', borderRadius: 12 }} />
        </div>
        )}
        <div style={{ fontSize: 9, opacity: 0.6, lineHeight: 1.2, marginBottom: 4 }}>
          💡 Δpred_EOL = pred_EOL(t) − pred_EOL(t−1). Negative → earlier EOL. Positive → later EOL.
        </div>

        {/* ===== 탭 컨텐츠 ===== */}
        {currentTab === 'overview' && (
        <div style={{ overflowY: 'auto', paddingRight: 4 }}>
          <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 15, fontWeight: 600 }}>Overview Signals</h3>

          {/* 4개 카테고리를 가로로 배치 */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 4, marginBottom: 8 }}>

            {/* 카테고리 1: Capacity */}
            <div>
              <div style={{ marginBottom: 3, fontSize: 11, opacity: 0.7, fontWeight: 600 }}>⚡ Capacity</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                <Card title="capacity_mean (Ah)" value={currentRow?.capacity_mean} suffix="" compact />
                <Card title="soh (%)" value={(currentRow?.soh ?? 0) * 100} suffix="%" compact />
                <Card title="capacity_derivative" value={currentRow?.capacity_derivative} suffix="Ah/cycle" compact />
                <Card title="cap_vel" value={currentRow?.cap_vel} suffix="" compact />
              </div>
            </div>

            {/* 카테고리 2: Load/Voltage */}
            <div>
              <div style={{ marginBottom: 3, fontSize: 11, opacity: 0.7, fontWeight: 600 }}>⚙️ Load/Voltage</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                <Card title="eff_c_rate (C)" value={currentRow?.eff_c_rate} suffix="" compact />
                <Card title="current_mean (A)" value={currentRow?.current_mean} suffix="" compact />
                <Card title="c_rate_peak (C)" value={cRatePeak} suffix="" compact />
                <Card title="voltage_measured_mean (V)" value={currentRow?.voltage_measured_mean} suffix="" compact />
                <Card title="voltage_min (V)" value={currentRow?.voltage_min} suffix="" compact />
                <Card title="voltage_max (V)" value={currentRow?.voltage_max} suffix="" compact />
              </div>
            </div>

            {/* 카테고리 3: Temperature */}
            <div>
              <div style={{ marginBottom: 3, fontSize: 11, opacity: 0.7, fontWeight: 600 }}>🌡️ Temperature</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                <Card title="temperature_mean (°C)" value={currentRow?.temperature_mean} suffix="" compact />
                <Card title="temperature_measured_max" value={currentRow?.temperature_measured_max} suffix="°C" compact />
                <Card title="temp_rise_cycle" value={currentRow?.temp_rise_cycle} suffix="°C" compact />
                <Card title="thermal_stress" value={currentRow?.thermal_stress} suffix="" compact />
                <Card title="dT/dt max" value={currentRow?.dTdt_max} suffix="°C/cycle" compact />
              </div>
            </div>

            {/* 카테고리 4: Resistance */}
            <div>
              <div style={{ marginBottom: 3, fontSize: 11, opacity: 0.7, fontWeight: 600 }}>🧱 Resistance</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                <Card title="dcr(Re)" value={currentRow?.dcr} suffix="Ω" compact />
                <Card title="dcr_growth (log ratio)" value={currentRow?.dcr_growth} suffix="" compact />
                <Card title="LLI" value={currentRow?.lli} suffix="" compact />
                <Card title="LAM" value={currentRow?.lam} suffix="" compact />
                <Card title="impedance_sum" value={currentRow?.impedance_sum} suffix="Ω" compact />
                <Card title="impedance_growth" value={currentRow?.impedance_growth} suffix="" compact />
              </div>
            </div>
          </div>
        </div>
        )}

        {/* Degradation Monitoring 탭 */}
        {currentTab === 'degradation' && (
          <DegradationTab battery={battery} rRatio={rRatio} cycle={cycle} />
        )}

                {/* Fixed4 Compare V19 */}
        {false && currentTab === 'compare' && (
          <Fixed4CompareTabV19
            battery={battery}
            cycle={cycle}
            batteries={batteries}
          />
        )}

        {/* Fixed4 Compare V20 */}
        {currentTab === 'compare' && (
          <Fixed4CompareTabV20 battery={battery} cycle={cycle} batteries={batteries} />
        )}

        {/* Fixed4 Explainability V20 */}
        {currentTab === 'explainability' && (
          <ExplainabilityAnomalyV30 battery={battery} rRatio={rRatio} cycle={cycle} />
        )}

{/* Compare Batteries 탭 */}
        {false && (
          <div style={{ overflowY: 'auto', paddingRight: 4 }}>
            <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 15, fontWeight: 600 }}>⚖️ Compare / Fleet view</h3>
            <p style={{ fontSize: 12, color: '#666', margin: '0 0 12px 0' }}>Rank batteries by anomaly z-scores at the current cycle.</p>

            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 6 }}>Select up to 4 batteries:</label>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {batteries.map((b) => (
                  <label key={b} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 10px', backgroundColor: selectedBatteriesCompare.includes(b) ? '#1976d2' : '#f5f5f5', color: selectedBatteriesCompare.includes(b) ? 'white' : '#333', borderRadius: 4, cursor: 'pointer', fontSize: 12, fontWeight: 600, userSelect: 'none' as const }}>
                    <input
                      type="checkbox"
                      checked={selectedBatteriesCompare.includes(b)}
                      onChange={(e) => {
                        if (e.target.checked && selectedBatteriesCompare.length < 4) {
                          setSelectedBatteriesCompare([...selectedBatteriesCompare, b])
                        } else if (!e.target.checked) {
                          setSelectedBatteriesCompare(selectedBatteriesCompare.filter((x) => x !== b))
                        }
                      }}
                      style={{ cursor: 'pointer' }}
                    />
                    {b}
                  </label>
                ))}
              </div>
            </div>

            {selectedBatteriesCompare.length === 0 ? (
              <div style={{ padding: 16, textAlign: 'center', color: '#999' }}>
                Select at least one battery to compare
              </div>
            ) : (
              <div>
                <div style={{ marginBottom: 16, overflowX: 'auto', fontSize: 11 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#e0e0e0', borderBottom: '2px solid #999' }}>
                        <th style={{ padding: '8px', textAlign: 'left', fontWeight: 600 }}>Battery</th>
                        <th style={{ padding: '8px', textAlign: 'center', fontWeight: 600 }}>Pred RUL</th>
                        <th style={{ padding: '8px', textAlign: 'center', fontWeight: 600 }}>RMSE</th>
                        <th style={{ padding: '8px', textAlign: 'center', fontWeight: 600 }}>Health</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedBatteriesCompare.map((bid, idx) => {
                        const item = batch?.items?.find((x) => x.battery_id === bid)
                        const packData = item ? buildPack(item) : null
                        const predRul = packData?.predRUL?.[packData.cycles.length - 1] ?? NaN
                        const rmse = item?.metrics?.rmse ?? NaN
                        const healthStatus = !Number.isFinite(rmse) ? 'N/A' : rmse < 10 ? '✅ Good' : rmse < 20 ? '⚠️ Fair' : '❌ Poor'

                        return (
                          <tr key={bid} style={{ borderBottom: '1px solid #ddd', backgroundColor: idx % 2 === 0 ? '#fafafa' : 'white' }}>
                            <td style={{ padding: '8px', fontWeight: 600, color: '#1976d2' }}>{bid}</td>
                            <td style={{ padding: '8px', textAlign: 'center', color: '#2196f3', fontWeight: 600 }}>
                              {Number.isFinite(predRul) ? predRul.toFixed(0) : 'N/A'}
                            </td>
                            <td style={{ padding: '8px', textAlign: 'center', color: '#ff9800', fontWeight: 600 }}>
                              {Number.isFinite(rmse) ? rmse.toFixed(2) : 'N/A'}
                            </td>
                            <td style={{ padding: '8px', textAlign: 'center', fontWeight: 600 }}>
                              {healthStatus}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                <div style={{ fontSize: 10, opacity: 0.7, padding: 8, backgroundColor: '#fff3e0', borderRadius: 4, border: '1px solid #ffe0b2' }}>
                  💡 Shows predictions and model accuracy (RMSE) for selected batteries.
                </div>
              </div>
            )}
          </div>
        )}

                {/* Fixed4 Explainability V19 */}
        {false && currentTab === 'explainability' && (
          <Fixed4ExplainabilityTabV19
            battery={battery}
            rRatio={rRatio}
            cycle={cycle}
          />
        )}

{/* Explainability 탭 */}
        {false && (
          <div style={{ overflowY: 'auto', paddingRight: 4 }}>
            <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 15, fontWeight: 600 }}>🔍 Prediction Explainability</h3>

            {!pack || !rows.length ? (
              <div style={{ padding: 16, textAlign: 'center', color: '#999' }}>
                No data available. Select a battery and load inference results.
              </div>
            ) : (
              <div>
                {/* Prediction confidence and uncertainty */}
                <div style={{ marginBottom: 16, padding: 12, backgroundColor: '#f9f9f9', borderRadius: 6, border: '1px solid #eee' }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 600 }}>Prediction Confidence (Current Cycle)</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    <div>
                      <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>Predicted RUL</div>
                      <div style={{ fontSize: 22, fontWeight: 700, color: '#2196f3' }}>
                        {pack!.cycles.includes(cycle) ? pack!.predRUL[pack!.cycles.indexOf(cycle)]?.toFixed(0) : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>Uncertainty (±2σ)</div>
                      <div style={{ fontSize: 22, fontWeight: 700, color: '#ff6f00' }}>
                        {pack!.cycles.includes(cycle)
                          ? (2 * (pack!.std[pack!.cycles.indexOf(cycle)] ?? 0)).toFixed(0)
                          : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Residual analysis */}
                <div style={{ marginBottom: 16, padding: 12, backgroundColor: '#f9f9f9', borderRadius: 6, border: '1px solid #eee' }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 600 }}>Residual Analysis (Prediction Error)</h4>
                  {(() => {
                    const idx = pack!.cycles.indexOf(cycle)
                    if (idx < 0) return <div style={{ color: '#999' }}>No prediction at this cycle</div>
                    const pred = pack!.predRUL[idx]
                    const true_rul = pack!.trueRUL[idx]
                    const residual = pred - true_rul
                    const isOverestimate = residual > 0

                    return (
                      <div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 12 }}>
                          <div>
                            <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>Prediction Error</div>
                            <div style={{ fontSize: 18, fontWeight: 700, color: residual > 0 ? '#ff9800' : '#4caf50' }}>
                              {residual > 0 ? '+' : ''}{residual.toFixed(0)} cycles
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>Error Type</div>
                            <div style={{ fontSize: 14, fontWeight: 600, color: isOverestimate ? '#ff9800' : '#4caf50' }}>
                              {isOverestimate ? '⬆️ Overestimate' : '⬇️ Underestimate'}
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>|Error| / σ</div>
                            <div style={{ fontSize: 18, fontWeight: 700, color: '#9c27b0' }}>
                              {(Math.abs(residual) / (pack!.std[idx] || 1)).toFixed(2)}σ
                            </div>
                          </div>
                        </div>

                        <div style={{ fontSize: 10, opacity: 0.7, color: '#666', lineHeight: 1.5, padding: 8, backgroundColor: 'white', borderRadius: 4, border: '1px solid #e0e0e0' }}>
                          {isOverestimate
                            ? '⚠️ Model overestimated RUL (predicts longer life than actual). This may occur early in battery cycle when degradation is unpredictable.'
                            : '⚠️ Model underestimated RUL (predicts shorter life than actual). This may occur later in cycle when degradation stabilizes.'}
                        </div>
                      </div>
                    )
                  })()}
                </div>

                {/* Feature importance proxies */}
                <div style={{ marginBottom: 16, padding: 12, backgroundColor: '#f9f9f9', borderRadius: 6, border: '1px solid #eee' }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 600 }}>Top Degradation Features (Current Cycle)</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {[
                      { name: 'Capacity Loss', value: Math.abs(currentRow?.capacity_derivative ?? 0), unit: 'Ah/cycle', color: '#ff9800' },
                      { name: 'DCR Growth', value: currentRow?.dcr_growth ?? 0, unit: 'Ω/cycle (log)', color: '#f44336' },
                      { name: 'Thermal Stress', value: currentRow?.thermal_stress ?? 0, unit: 'index', color: '#ff5722' },
                      { name: 'Impedance Change', value: currentRow?.impedance_growth ?? 0, unit: 'Ω/cycle', color: '#e91e63' }
                    ].map((feature) => (
                      <div key={feature.name} style={{ padding: 8, backgroundColor: 'white', borderRadius: 4, border: '1px solid #e0e0e0' }}>
                        <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6, color: '#333' }}>{feature.name}</div>
                        <div style={{ fontSize: 14, fontWeight: 700, color: feature.color, marginBottom: 4 }}>
                          {feature.value.toFixed(5)}
                        </div>
                        <div style={{ fontSize: 10, color: '#999' }}>{feature.unit}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Model architecture info */}
                <div style={{ marginBottom: 16, padding: 12, backgroundColor: '#f9f9f9', borderRadius: 6, border: '1px solid #eee' }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 600 }}>Model: Bayesian MAML (Few-shot Learning)</h4>
                  <ul style={{ margin: 0, paddingLeft: 16, fontSize: 11, color: '#666', lineHeight: 1.6 }}>
                    <li><strong>Support set:</strong> {pack!.supportCycles.length} cycles used for rapid adaptation</li>
                    <li><strong>Query set:</strong> {pack!.cycles.length} cycles for RUL prediction</li>
                    <li><strong>Model:</strong> Bayesian Model-Agnostic Meta-Learning (BMAML) with PyTorch</li>
                    <li><strong>Output:</strong> Predicted RUL with epistemic uncertainty estimates (±2σ)</li>
                    <li><strong>Features:</strong> Capacity, DCR, temperature, impedance, and derived metrics</li>
                  </ul>
                </div>

                {/* Calibration guidance */}
                <div style={{ fontSize: 10, opacity: 0.7, padding: 8, backgroundColor: '#e8f5e9', borderRadius: 4, border: '1px solid #c8e6c9', marginTop: 12 }}>
                  💡 <strong>Model calibration:</strong> Uncertainty bounds (±2σ) represent ~95% confidence intervals. Wider bounds = higher uncertainty,
                  typically occurring at early or extreme degradation stages where the model has less training data.
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ===== Degradation Tab Component (새로운 Grid 구조) =====
// HOTFIX_V15B_SAFE_PLAYBACK_NO_ROWS_DECL
// HOTFIX_V13_RESTORE_STABLE_FIXED4_ISSUES_PLAYBACK_LOADING
function DegradationTab({ battery, rRatio, cycle: parentCycle }: any) {
  // ✅ 완전히 독립적: battery와 rRatio만 props로 받음
  const [anomalyReport, setAnomalyReport] = React.useState<any>(null)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  // ✅ cycle은 내부 상태로 관리 (기본값: 1)
  const [selectedCycle, setSelectedCycle] = React.useState<number>(1)
  const [isPlaying, setIsPlaying] = React.useState(false)

  const playbackRunRef = React.useRef(0)

  // HOTFIX_V15B: sync parentCycle into DegradationTab without forcing invalid 0.
  React.useEffect(() => {
    const c = Number(parentCycle)
    if (Number.isFinite(c) && c > 0) setSelectedCycle(c)
  }, [parentCycle])
  const [loadingStartedAt, setLoadingStartedAt] = React.useState<number | null>(null)
  const [loadingNow, setLoadingNow] = React.useState<number>(() => Date.now())

  // Keep Degradation tab in sync with the sidebar Playback slider.
  React.useEffect(() => {
    const c = Number(parentCycle)
    if (Number.isFinite(c) && c > 0) setSelectedCycle(c)
  }, [parentCycle])

  React.useEffect(() => {
    if (!loading) return
    const t = window.setInterval(() => setLoadingNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [loading])

  // Call backend API to get anomaly report
  React.useEffect(() => {
    if (!battery) {
      console.log('🔴 DegradationTab: battery not set, exiting')
      setLoading(false)
      return
    }

    let isMounted = true

    const fetchAnomalyReport = async () => {
      try {
        setLoading(true)
        setLoadingStartedAt(Date.now())
        setLoadingNow(Date.now())
        setError(null)
        // ✅ Unified endpoint that returns all data in one response
        const url = `/api/battery/${battery}/degradation-monitoring?r_ratio=${rRatio}`
        console.log('🟡 DegradationTab: fetching from unified endpoint:', url)
        const response = await axios.get(url)
        if (isMounted) {
          console.log('🟢 DegradationTab: success, cycle:', response.data.cycle, 'series:', response.data.series?.length)
          // Map new response structure to internal format for compatibility
          const mappedData = {
            battery: response.data.battery,
            cycle: response.data.cycle,
            series: response.data.series || response.data.rows || [],
            rows: response.data.series || response.data.rows || [],
            bands: response.data.bands,
            z_scores: response.data.z_scores,
            zSeries: {  // map to old format for compatibility with existing code
              dcr: response.data.z_scores?.dcr || [],
              capacity: response.data.z_scores?.capacity || []
            },
            issues: response.data.issues || [],
            majorAlerts: response.data.majorAlerts || response.data.issues || [],
            status: response.data.status,
            earlyWarning: response.data.earlyWarning || { active: false, message: null },
            drivers: response.data.drivers || response.data.potentialDrivers || [],
            potentialDrivers: response.data.potentialDrivers || response.data.drivers || [],
            reportMarkdown: response.data.reportMarkdown || response.data.markdown || ''
          }
          setAnomalyReport(mappedData)
          const cycleList = (mappedData.rows || [])
            .map((r: any) => Number(r.cycle_num ?? r.cycle ?? r.anchor_cycle))
            .filter(Number.isFinite)
            .sort((a: number, b: number) => a - b)
          const pCycle = Number(parentCycle)
          setSelectedCycle(Number.isFinite(pCycle) && pCycle > 0 ? pCycle : (cycleList[0] || 1))
        }
      } catch (err: any) {
        if (isMounted) {
          console.error('🔴 DegradationTab: Error fetching degradation monitoring:', err.message)
          setError(err.message || 'Failed to fetch degradation monitoring data')
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    fetchAnomalyReport()

    return () => {
      isMounted = false
    }
  }, [battery, rRatio])

  // HOTFIX_V15B: safe playback interval. Loops from the first cycle when Play is pressed at the end.
  React.useEffect(() => {
    if (!isPlaying) return

    const playbackRows = (anomalyReport?.rows || anomalyReport?.series || [])
    const cycleList = playbackRows
      .map((r: any) => Number(r.cycle_num ?? r.cycle ?? r.anchor_cycle))
      .filter(Number.isFinite)
      .sort((a: number, b: number) => a - b)

    if (!cycleList.length) return

    const timer = window.setInterval(() => {
      setSelectedCycle((prev: number) => {
        const cur = Number.isFinite(Number(prev)) ? Number(prev) : cycleList[0]
        const idx = cycleList.findIndex((c: number) => c >= cur)
        if (idx < 0 || idx >= cycleList.length - 1) return cycleList[0]
        return cycleList[idx + 1]
      })
    }, 300)

    return () => window.clearInterval(timer)
  }, [isPlaying, anomalyReport?.rows, anomalyReport?.series])

  if (loading) {
    const elapsedSec = Math.max(0, Math.floor(((loadingNow || Date.now()) - (loadingStartedAt || Date.now())) / 1000))
    return (
      <div style={{ padding: 28, textAlign: 'center', color: '#555', lineHeight: 1.7 }}>
        <div style={{ fontWeight: 800, fontSize: 15, marginBottom: 6 }}>Loading degradation data...</div>
        <div>Elapsed: {elapsedSec}s</div>
        <div style={{ fontSize: 12, opacity: 0.8 }}>초기 CSV/robust z-score 계산 때문에 보통 10~40초 정도 걸릴 수 있습니다.</div>
        {elapsedSec >= 40 && (
          <div style={{ marginTop: 8, color: '#b26a00', fontSize: 12 }}>
            Still working... 서버가 멈춘 것이 아니라 계산이 길어지는 중일 수 있습니다.
          </div>
        )}
      </div>
    )
  }

  if (error) {
    return <div style={{ padding: 20, textAlign: 'center', color: '#f00' }}>Error: {error}</div>
  }

  if (!anomalyReport || !anomalyReport.rows || anomalyReport.rows.length === 0) {
    return <div style={{ padding: 20, textAlign: 'center', color: '#999' }}>No degradation data available</div>
  }

  try {
    // Use API response data
    const dcrZSeries = anomalyReport.zSeries?.dcr || []
    const capZSeries = anomalyReport.zSeries?.capacity || []
    const majorAlerts = anomalyReport.majorAlerts || []
    const earlyWarning = anomalyReport.earlyWarning || { active: false, message: null }
    const potentialDrivers = anomalyReport.potentialDrivers || []
    const apiRows = anomalyReport.rows || []
    const rawBands = anomalyReport.bands || {}

    // ✅ API 응답에서 실제 cycle 값 (최신값일 수 있음)
    const effectiveCycle = anomalyReport.cycle || selectedCycle

    // Convert bands from {feature: {cycle: stats}} to {feature: [{cycle, ...stats}]}
    const apiBands = Object.fromEntries(
      Object.entries(rawBands).map(([feat, cycles]: any) => [
        feat,
        Object.entries(cycles).map(([cycle, stats]: any) => ({
          cycle: Number(cycle),
          ...(typeof stats === 'object' ? stats : {})
        }))
      ])
    )

    // Calculate common cycles from API response data
    const commonCycles = apiRows
      .map((r: any) => Number(r.cycle_num ?? r.cycle ?? r.anchor_cycle))
      .filter(Number.isFinite)
      .sort((a: number, b: number) => a - b)

    if (!commonCycles.length) {
      return <div style={{ padding: 20, textAlign: 'center', color: '#999' }}>No valid cycles found</div>
    }

    const xMin = Math.min(...commonCycles)
    const xMax = Math.max(...commonCycles)

    // ===== RECALCULATE ALERTS AND DRIVERS FOR SELECTED CYCLE =====
    const computeRobustZ = (value: number, median: number, q1: number, q3: number) => {
      const iqr = q3 - q1
      const scale = Math.max(iqr / 1.349, 1e-9)
      return (value - median) / scale
    }

    // Get z-scores for selected cycle
    const selectedDcrZ = dcrZSeries.find((p: any) => Number(p.cycle) === Number(selectedCycle))?.z ?? null
    const selectedCapZ = capZSeries.find((p: any) => Number(p.cycle) === Number(selectedCycle))?.z ?? null

    // Fixed4-compatible dynamic alert rendering:
    // fixed4 online monitoring evaluates the prefix up to selectedCycle, not just one point.
    const numericSelectedCycle = Number(selectedCycle)

    const dcrPrefix = dcrZSeries
      .filter((p: any) => Number.isFinite(Number(p?.z)) && Number(p?.cycle) <= numericSelectedCycle)
      .map((p: any) => ({ cycle: Number(p.cycle), z: Number(p.z) }))

    const capPrefix = capZSeries
      .filter((p: any) => Number.isFinite(Number(p?.z)) && Number(p?.cycle) <= numericSelectedCycle)
      .map((p: any) => ({ cycle: Number(p.cycle), z: Number(p.z) }))

    const dcrWorst = dcrPrefix.length
      ? dcrPrefix.reduce((best: any, p: any) => (p.z > best.z ? p : best), dcrPrefix[0])
      : null

    const capWorst = capPrefix.length
      ? capPrefix.reduce((best: any, p: any) => (p.z < best.z ? p : best), capPrefix[0])
      : null

    const findOnset = (points: any[], threshold: number, direction: 'pos' | 'neg', minRun: number) => {
      let run = 0
      let startCycle: number | null = null
      for (const p of points) {
        const z = Number(p.z)
        const ok = direction === 'pos' ? z >= threshold : z <= -threshold
        if (ok) {
          if (run === 0) startCycle = Number(p.cycle)
          run += 1
          if (run >= minRun) return startCycle
        } else {
          run = 0
          startCycle = null
        }
      }
      return null
    }

    const cycleAlerts: any[] = []
    if (dcrWorst && dcrWorst.z >= 4.0) {
      cycleAlerts.push({
        type: 'dcr_spike',
        label: 'Fault-like anomaly (DCR spike)',
        severity: Math.abs(dcrWorst.z) >= 6 ? 'HIGH' : 'MED',
        onsetCycle: findOnset(dcrPrefix, 3.0, 'pos', 2),
        zValue: dcrWorst.z,
      })
    }

    if (capWorst && capWorst.z <= -3.5) {
      cycleAlerts.push({
        type: 'capacity_drop',
        label: 'Accelerated degradation (Capacity drop)',
        severity: Math.abs(capWorst.z) >= 6 ? 'HIGH' : 'MED',
        onsetCycle: findOnset(capPrefix, 3.0, 'neg', 3),
        zValue: capWorst.z,
      })
    }

    // Drivers for selected cycle
    const driverDefs = [
      { key: 'thermal_stress', label: '고온/열 스트레스' },
      { key: 'temperature_mean', label: '고온 노출' },
      { key: 'temp_rise_cycle', label: '셀 발열 증가' },
      { key: 'eff_c_rate', label: '고 C-rate(고부하)' },
      { key: 'current_max', label: '고부하(충전/회생)' },
      { key: 'current_min', label: '고부하(방전)' },
      { key: 'voltage_min', label: '깊은 방전(DoD↑)' },
      { key: 'dvdt_max_abs', label: '전압 급변' },
      { key: 'dTdt_max', label: '온도 급상승' },
    ]

    const selectedRow = apiRows.find((r: any) => r.cycle_num === selectedCycle)
    const cyclePotentialDrivers = []

    if (selectedRow) {
      const drivers = driverDefs
        .map((def: any) => {
          const value = selectedRow[def.key]
          if (value === null || value === undefined) return null

          const bandData = rawBands[def.key]?.[selectedCycle]
          if (!bandData) return null

          const z = computeRobustZ(value, bandData.median, bandData.q1, bandData.q3)
          return {
            label: def.label,
            feature: def.key,
            value: value,
            z: z,
            absZ: Math.abs(z)
          }
        })
        .filter((d: any) => d !== null)
        .sort((a: any, b: any) => b.absZ - a.absZ)
        .slice(0, 3)

      cyclePotentialDrivers.push(...drivers)
    }

    const significantDrivers = cyclePotentialDrivers.filter((d: any) => Math.abs(d.z) >= 3.0)
    const cycleEarlyWarning = {
      active: cycleAlerts.length === 0 && ((anomalyReport.earlyWarning?.active) || significantDrivers.length > 0),
      message: anomalyReport.earlyWarning?.message || (
        cycleAlerts.length === 0 && significantDrivers.length > 0
          ? '핵심 KPI(DCR/Capacity%) 기준의 \'큰 이탈\'은 아직 없지만, 일부 스트레스 신호(driver)가 cohort 대비 outlier 입니다. (early warning)'
          : null
      )
    }

    // Get target driver cycle (selected cycle)
    const targetDriverCycle = selectedCycle

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, paddingRight: 4 }}>
      {/* HOTFIX_V16_DIRECT_PLAYBACK_FIXED4_PREFIX: direct local playback controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 12px',
        marginBottom: 12,
        borderRadius: 8,
        background: '#0f172a',
        color: '#fff',
        boxShadow: '0 2px 10px rgba(15,23,42,0.18)'
      }}>
        <button
          onClick={async () => {
            if (!commonCycles.length) return

            const token = playbackRunRef.current + 1
            playbackRunRef.current = token
            setIsPlaying(true)

            const last = commonCycles[commonCycles.length - 1]
            const startIndex =
              Number(selectedCycle) >= Number(last)
                ? 0
                : Math.max(0, commonCycles.findIndex((c: number) => Number(c) >= Number(selectedCycle)))

            for (let i = startIndex; i < commonCycles.length; i += 1) {
              if (playbackRunRef.current !== token) return
              setSelectedCycle(Number(commonCycles[i]))
              await new Promise((resolve) => window.setTimeout(resolve, 300))
            }

            if (playbackRunRef.current === token) setIsPlaying(false)
          }}
          style={{ padding: '7px 12px', background: '#1976d2', color: '#fff', border: 0, borderRadius: 5, cursor: 'pointer', fontWeight: 700 }}
        >
          {isPlaying ? '▶ Playing' : '▶ Play'}
        </button>

        <button
          onClick={() => {
            playbackRunRef.current += 1
            setIsPlaying(false)
            const idx = commonCycles.findIndex((c: number) => Number(c) >= Number(selectedCycle))
            if (idx < 0 || idx >= commonCycles.length - 1) setSelectedCycle(Number(commonCycles[0]))
            else setSelectedCycle(Number(commonCycles[idx + 1]))
          }}
          style={{ padding: '7px 12px', background: '#f59e0b', color: '#fff', border: 0, borderRadius: 5, cursor: 'pointer', fontWeight: 700 }}
        >
          ⏭ Step
        </button>

        <button
          onClick={() => {
            playbackRunRef.current += 1
            setIsPlaying(false)
            setSelectedCycle(Number(commonCycles[0]))
          }}
          style={{ padding: '7px 12px', background: '#dc2626', color: '#fff', border: 0, borderRadius: 5, cursor: 'pointer', fontWeight: 700 }}
        >
          ⏮ Reset
        </button>

        <input
          type="range"
          min={xMin}
          max={xMax}
          value={Math.max(xMin, Math.min(Number(selectedCycle) || xMin, xMax))}
          onChange={(e) => {
            playbackRunRef.current += 1
            setIsPlaying(false)
            setSelectedCycle(Number(e.target.value))
          }}
          style={{ flex: 1 }}
        />

        <strong style={{ minWidth: 90, textAlign: 'right' }}>Cycle {Math.max(xMin, Math.min(Number(selectedCycle) || xMin, xMax))}</strong>
      </div>

        {/* === Degradation Monitoring Chart Container === */}
        <div style={{ padding: 12, backgroundColor: '#f5f5f5', borderRadius: 6, border: '1px solid #ddd', flexShrink: 0 }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 600 }}>📈 Degradation monitoring (selected battery)</h4>

          {/* Row 1: SoH + Capacity */}
          <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 180px', gap: 12, marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', fontSize: 12, fontWeight: 600, color: '#333' }}>
              SoH + Capacity
            </div>
            <Row1Chart rows={apiRows} cycle={selectedCycle} bands={apiBands} commonCycles={commonCycles} xMin={xMin} xMax={xMax} initialCapacity={apiRows[0]?.capacity_mean} />
            <RowLegend items={[
              { type: 'solid', color: '#1f77b4', label: 'SoH (%)' },
              { type: 'solid', color: '#ff7f0e', label: 'Capacity (%)' },
              { type: 'dashed', color: '#999', label: 'Expected median' },
              { type: 'band', color: '#ccc', label: 'Expected IQR' },
              { type: 'dotted', color: '#e74c3c', label: 'Current cycle' },
            ]} />
          </div>

          {/* Row 2: DCR + Impedance */}
          <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 180px', gap: 12, marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', fontSize: 12, fontWeight: 600, color: '#333' }}>
              DCR + Impedance
            </div>
            <Row2Chart rows={apiRows} cycle={selectedCycle} bands={apiBands} commonCycles={commonCycles} xMin={xMin} xMax={xMax} />
            <RowLegend items={[
              { type: 'solid', color: '#d62728', label: 'DCR (Ω)' },
              { type: 'solid', color: '#2ca02c', label: 'Impedance (Ω)' },
              { type: 'dashed', color: '#999', label: 'Expected median' },
              { type: 'band', color: '#ccc', label: 'Expected IQR' },
              { type: 'dotted', color: '#e74c3c', label: 'Current cycle' },
            ]} />
          </div>

          {/* Row 3: Temperature + Thermal */}
          <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 180px', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', fontSize: 12, fontWeight: 600, color: '#333' }}>
              Temperature + Thermal
            </div>
            <Row3Chart rows={apiRows} cycle={selectedCycle} bands={apiBands} commonCycles={commonCycles} xMin={xMin} xMax={xMax} />
            <RowLegend items={[
              { type: 'solid', color: '#9467bd', label: 'Temp (°C) [left]' },
              { type: 'solid', color: '#e377c2', label: 'Thermal [right]' },
              { type: 'dashed', color: '#999', label: 'Expected median' },
              { type: 'band', color: '#ccc', label: 'Expected IQR' },
              { type: 'dotted', color: '#e74c3c', label: 'Current cycle' },
            ]} />
          </div>
        </div>

        {/* === Anomaly Report Chart Container === */}
        <div style={{ padding: 12, backgroundColor: '#f5f5f5', borderRadius: 6, border: '1px solid #ddd', flexShrink: 0 }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 600 }}>🧪 Anomaly report (expected vs observed)</h4>

          {/* Anomaly chart with grid layout */}
          <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 180px', gap: 12, marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', fontSize: 12, fontWeight: 600, color: '#333' }}>
              Robust z-score
            </div>
            <div style={{ width: '100%' }}>
              <AnomalyZScoreChart rows={apiRows} cycle={selectedCycle} bands={apiBands} commonCycles={commonCycles} xMin={xMin} xMax={xMax} dcrZSeries={dcrZSeries} capZSeries={capZSeries} />
            </div>
            <RowLegend items={[
              { type: 'solid', color: '#1f77b4', label: 'Robust z(DCR)' },
              { type: 'solid', color: '#ff7f0e', label: 'Robust z(Cap%)' },
              { type: 'dashed', color: '#d62728', label: 'Threshold ±3' },
              { type: 'dotted', color: '#e74c3c', label: 'Current cycle' },
            ]} />
          </div>

          {/* 3-Stage Alert Decision Tree - Based on Selected Cycle */}
          {cycleAlerts.length > 0 ? (
            // Stage 1: Major anomaly detected
            <div style={{ marginBottom: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {cycleAlerts.map((a: any, idx: number) => (
                <div
                  key={idx}
                  style={{
                    padding: 10,
                    backgroundColor: a.severity === 'HIGH' ? '#fff3cd' : '#f0f0f0',
                    borderLeft: `4px solid ${a.severity === 'HIGH' ? '#d4af37' : '#ffc107'}`,
                    borderRadius: 4,
                    fontSize: 11,
                    fontFamily: 'monospace',
                    color: '#333',
                  }}
                >
                  <strong>{a.label}</strong> · severity: {a.severity} · onset: {a.onsetCycle !== null && a.onsetCycle !== undefined ? `cycle ${a.onsetCycle}` : '(onset 미확정)'} · robust z: {Number(a.zValue ?? 0).toFixed(2)}
                </div>
              ))}
            </div>
          ) : cycleEarlyWarning?.active ? (
            // Stage 2: Early warning - driver outlier but no major anomaly
            <div
              style={{
                marginBottom: 12,
                padding: 10,
                backgroundColor: '#e3f2fd',
                borderLeft: '4px solid #42a5f5',
                borderRadius: 4,
                fontSize: 11,
                fontFamily: 'monospace',
                color: '#1565c0',
              }}
            >
              {cycleEarlyWarning.message || '핵심 KPI(DCR/Capacity%) 기준의 \'큰 이탈\'은 아직 없지만, 일부 스트레스 신호(driver)가 cohort 대비 outlier 입니다. (early warning)'}
            </div>
          ) : (
            // Stage 3: Normal - no major anomaly, no driver outliers
            <div
              style={{
                marginBottom: 12,
                padding: 10,
                backgroundColor: '#e8f5e9',
                borderLeft: '4px solid #81c784',
                borderRadius: 4,
                fontSize: 11,
                fontFamily: 'monospace',
                color: '#2e7d32',
              }}
            >
              ✓ 현재 사이클({selectedCycle})은 reference cohort의 기대 범위 내에서 큰 이탈이 관측되지 않았습니다.
            </div>
          )}

          <div style={{ marginTop: 12, marginBottom: 12 }}>
            <h5 style={{ margin: '0 0 8px 0', fontSize: 12, fontWeight: 600 }}>Potential drivers at cycle {selectedCycle} (cohort 대비 robust z-score)</h5>
            {(cycleAlerts.length > 0 ? potentialDrivers : cyclePotentialDrivers).length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {(cycleAlerts.length > 0 ? potentialDrivers : cyclePotentialDrivers).map((driver: any, idx: number) => (
                  <div key={idx} style={{ fontSize: 11, color: '#333', padding: '4px 8px', backgroundColor: '#f9f9f9', borderRadius: 3 }}>
                    <strong>{driver.label || driver.tag || driver.feature}</strong> · z={Number(driver.z ?? 0).toFixed(2)} · value={typeof driver.value === 'number' ? driver.value.toFixed(2) : driver.value}
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: 11, color: '#999' }}>No potential drivers at this cycle</div>
            )}
          </div>

          {(anomalyReport.reportMarkdown || anomalyReport.markdown) && (
            <div style={{ marginTop: 12, marginBottom: 12 }}>
              <button
                onClick={() => {
                  const md = anomalyReport.reportMarkdown || anomalyReport.markdown || ''
                  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `${battery}_anomaly_report.md`
                  document.body.appendChild(a)
                  a.click()
                  a.remove()
                  URL.revokeObjectURL(url)
                }}
                style={{
                  padding: '7px 12px',
                  backgroundColor: '#fff',
                  border: '1px solid #bbb',
                  borderRadius: 4,
                  cursor: 'pointer',
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                📄 Download anomaly report (Markdown)
              </button>
            </div>
          )}

          <div style={{ fontSize: 10, opacity: 0.7, color: '#666', lineHeight: 1.4 }}>
            ※ 위 판정/설명은 데모용(rule + robust z-score)이며, 실제 운영에서는 조건(cohort)·센서 품질·물리 제약을 함께 고려해 임계치/로직을 튜닝합니다.
          </div>
        </div>
      </div>
    )
  } catch (e) {
    console.error('❌ Error in DegradationTab:', e)
    return <div style={{ padding: 20, color: '#f00', fontSize: 12 }}>Error rendering Degradation tab. Check console.</div>
  }
}
