import React, { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

type BatchItem = any;

type BatchPayload = { tag?: string; items: any[] };

const R_CHOICES = ["r0p10","r0p20","r0p30","r0p40","r0p50"];

function bandTrace(x: any, mean: any, std: any) {
  const upper = mean.map((m: any, i: any) => m + 2 * (std?.[i] ?? 0));
  const lower = mean.map((m: any, i: any) => m - 2 * (std?.[i] ?? 0)).reverse();
  return {
    x: x.concat([...x].reverse()),
    y: upper.concat(lower),
    type: "scatter",
    mode: "lines",
    line: { width: 0 },
    fill: "toself",
    name: "±2σ",
    hoverinfo: "skip",
  };
}

function computeXRange(item: any) {
  const split = Number(item.split_cycle);
  const qx = item.query?.cycle ?? [];
  const qy = item.query?.true_rul ?? [];

  // last cycle where GT > 0 (if exists)
  let lastPosCycle = null;
  for (let i = qy.length - 1; i >= 0; i--) {
    if (qy[i] > 0) { lastPosCycle = qx[i]; break; }
  }

  const left = Math.max(0, split - 10);
  const right = lastPosCycle != null ? Math.max(split + 10, lastPosCycle + 5) : split + 30;
  return [left, right];
}

export default function RulPlotlyViewer() {
  const [tag, setTag] = useState("r0p10");      // r0p10/r0p20/...
  const [bid, setBid] = useState("B0018");
  const [batch, setBatch] = useState(null);

  useEffect(() => {
    fetch(`/precomputed/viz_${tag}_v3pos/batch_viz_meta_${tag}.json`)
      .then(r => r.json())
      .then(setBatch);
  }, [tag]);

  const item = useMemo(() => {
    const items = (batch as any)?.items ?? [];
    return items.find((x: any) => x.battery_id === bid) ?? null;
  }, [batch, bid]);

  const meta = useMemo(() => {
    if (!item) return "";
    return `len_s=${item.len_s} / len_q=${item.len_q} / q_pos=${item.q_pos} / split=${item.split_cycle} / r_eff=${Number(item.r_ratio_effective).toFixed(3)}`;
  }, [item]);

  const plotData = useMemo(() => {
    if (!item) return [];
    const sx = item.support.cycle;
    const sy = item.support.rul;
    const qx = item.query.cycle;
    const qy = item.query.true_rul;
    const pm = item.pred.mean;
    const ps = item.pred.std;

    return [
      { x: sx, y: sy, type: "scatter", mode: "lines+markers", name: `Support(${sx.length})` },
      { x: qx, y: qy, type: "scatter", mode: "lines", name: "True Future RUL", line: { dash: "dash" } },
      ...(ps ? [bandTrace(qx, pm, ps)] : []),
      { x: qx, y: pm, type: "scatter", mode: "lines", name: "Pred RUL" },
    ];
  }, [item]);

  const layout = useMemo(() => {
    if (!item) return {};
    const [x0, x1] = computeXRange(item);
    return {
      title: `${bid} (${tag})`,
      xaxis: { title: "cycle", range: [x0, x1], automargin: true },
      yaxis: { title: "RUL", rangemode: "tozero", automargin: true },
      shapes: [
        { // split vertical line
          type: "line",
          x0: item.split_cycle, x1: item.split_cycle,
          y0: 0, y1: 1,
          xref: "x", yref: "paper",
          line: { dash: "dot" },
        }
      ],
      margin: { l: 55, r: 20, t: 55, b: 55 },
      legend: { orientation: "h" },
    };
  }, [item, bid, tag]);

  const bidOptions = useMemo(() => {
    const items = (batch as any)?.items ?? [];
    return [...new Set(items.map((x: any) => x.battery_id))].sort();
  }, [batch]);

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <label>
          Battery&nbsp;
          <select value={bid} onChange={(e) => setBid(e.target.value)}>
            {bidOptions.map((b: any) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>

        <label>
          r_ratio&nbsp;
          <select value={tag} onChange={(e) => setTag(e.target.value)}>
            {R_CHOICES.map((t: any) => <option key={t} value={t}>{t.replace("r0p","0.")}</option>)}
          </select>
        </label>
      </div>

      <div style={{ marginTop: 8, opacity: 0.85, fontFamily: "monospace" }}>
        {item ? meta : "loading..."}
      </div>

      <div style={{ marginTop: 12 }}>
        {item && (
          <Plot
            data={plotData}
            layout={layout}
            config={{ responsive: true, displayModeBar: true }}
            style={{ width: "100%", maxWidth: 1100 }}
          />
        )}
      </div>
    </div>
  );
}