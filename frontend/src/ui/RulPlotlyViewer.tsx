import React, { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

type BatchItem = {
  battery_id: string;
  split_cycle: number;
  len_s?: number;
  len_q?: number;
  q_pos?: number;
  r_ratio_effective?: number;
  support: { cycle: number[]; rul: number[] };
  query: { cycle: number[]; true_rul: number[] };
  pred: { mean: number[]; std?: number[] };
};

type BatchPayload = { tag?: string; created_at?: string; items: BatchItem[] };

type Props = {
  tag: string;          // e.g. "r0p10"
  bid: string;          // e.g. "B0018"
  selectedCycle?: number | null; // actual cycle number (same scale as query cycles)
  autoZoom?: boolean;
  showExplain?: boolean;
};

function bandTrace(x: number[], mean: (number | null)[], std: (number | null)[]) {
  const upper = mean.map((m, i) => (m == null ? null : m + 2 * (std?.[i] ?? 0)));
  const lower = mean.map((m, i) => (m == null ? null : m - 2 * (std?.[i] ?? 0))).reverse();
  const xx = x.concat([...x].reverse());
  const yy = upper.concat(lower);
  return {
    x: xx,
    y: yy,
    type: "scatter",
    mode: "lines",
    line: { width: 0 },
    fill: "toself",
    name: "Pred ±2σ",
    hoverinfo: "skip",
  } as const;
}

function closestIndex(xs: number[], x: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < xs.length; i++) {
    const d = Math.abs(xs[i] - x);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

export default function RulPlotlyViewer(props: Props) {
  const { tag, bid, selectedCycle = null, autoZoom = false, showExplain = true } = props;
  const [batch, setBatch] = useState<BatchPayload | null>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    setErr("");
    setBatch(null);
    fetch(`/precomputed/viz_${tag}_v3pos/batch_viz_meta_${tag}.json`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
      })
      .then(setBatch)
      .catch((e) => setErr(String(e?.message ?? e)));
  }, [tag]);

  const item = useMemo(() => {
    const items = batch?.items ?? [];
    return items.find((x) => x.battery_id === bid) ?? null;
  }, [batch, bid]);

  const meta = useMemo(() => {
    if (!item) return "";
    const lenS = item.len_s ?? item.support?.cycle?.length ?? 0;
    const lenQ = item.len_q ?? item.query?.cycle?.length ?? 0;
    const qPos = item.q_pos ?? (item.query?.true_rul?.filter((v) => v > 0).length ?? 0);
    const rEff = item.r_ratio_effective;
    return `split=${item.split_cycle} / support=${lenS} / query=${lenQ} / q_pos=${qPos}` + (rEff != null ? ` / r_eff=${Number(rEff).toFixed(3)}` : "");
  }, [item]);

  const plotData = useMemo(() => {
    if (!item) return [];

    const sx = item.support.cycle.map(Number);
    const sy = item.support.rul.map((v) => Number(v));

    const qx = item.query.cycle.map(Number);
    const qy = item.query.true_rul.map((v) => Number(v));

    const pm = item.pred.mean.map((v) => Number(v));
    const ps = Array.isArray(item.pred.std)
      ? item.pred.std.map((v) => Number(v))
      : Array(pm.length).fill(0);

    const traces: any[] = [];

    traces.push({
      x: sx,
      y: sy,
      type: "scatter",
      mode: "lines+markers",
      name: `Support(${sx.length})`,
    });

    traces.push({
      x: qx,
      y: qy,
      type: "scatter",
      mode: "lines",
      name: "True RUL",
      line: { dash: "dash" },
    });

    traces.push(bandTrace(qx, pm, ps));

    traces.push({
      x: qx,
      y: pm,
      type: "scatter",
      mode: "lines",
      name: "Pred RUL",
    });

    // Selected cycle markers (true/pred)
    if (selectedCycle != null && qx.length > 0) {
      const idx = closestIndex(qx, selectedCycle);
      const cx = qx[idx];
      traces.push({
        x: [cx],
        y: [qy[idx]],
        type: "scatter",
        mode: "markers",
        name: "Selected cycle (true)",
        marker: { size: 9 },
      });
      traces.push({
        x: [cx],
        y: [pm[idx]],
        type: "scatter",
        mode: "markers",
        name: "Selected cycle (pred)",
        marker: { size: 9, symbol: "diamond" },
      });
    }

    return traces;
  }, [item, selectedCycle]);

  const plotLayout = useMemo(() => {
    if (!item) return { title: "Loading..." };

    const sx = item.support.cycle.map(Number);
    const qx = item.query.cycle.map(Number);
    const xMin = Math.min(...sx, ...qx);
    const xMax = Math.max(...sx, ...qx);

    // Auto-zoom range around split and positive region (if any)
    let xRange: [number, number] | undefined = undefined;
    if (autoZoom && qx.length > 0) {
      const split = Number(item.split_cycle);
      let left = split - 10;
      let right = split + 60;

      // extend right until last positive true_rul if it exists
      const qy = item.query.true_rul.map(Number);
      let lastPos = -1;
      for (let i = 0; i < qy.length; i++) if (qy[i] > 0) lastPos = i;
      if (lastPos >= 0) right = qx[lastPos] + 10;

      left = Math.max(left, xMin);
      right = Math.min(right, xMax);
      if (right <= left) { left = xMin; right = xMax; }
      xRange = [left, right];
    }

    return {
      title: `${item.battery_id} / ${tag}`,
      xaxis: { title: "Cycle", range: xRange as any },
      yaxis: { title: "RUL (cycles)" },
      margin: { l: 55, r: 20, t: 45, b: 45 },
      legend: { orientation: "h" },
      shapes: [
        {
          type: "line",
          x0: item.split_cycle,
          x1: item.split_cycle,
          y0: 0,
          y1: 1,
          xref: "x",
          yref: "paper",
          line: { dash: "dot", width: 2 },
        },
      ],
    } as any;
  }, [item, tag, autoZoom]);

  if (err) {
    return (
      <div style={{ border: "1px solid #f99", borderRadius: 8, padding: 10, color: "#a00" }}>
        precomputed load failed: {err}
      </div>
    );
  }

  return (
    <div>
      <div style={{ fontFamily: "monospace", fontSize: 12, opacity: 0.85, marginBottom: 6 }}>
        {meta || "—"}
      </div>

      <Plot
        data={plotData as any}
        layout={plotLayout as any}
        config={{ responsive: true, displayModeBar: true }}
        style={{ width: "100%", height: 520 }}
      />

      {showExplain && (
        <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>
          <div style={{ marginBottom: 4 }}>
            • <b>Selected cycle (true/pred)</b> markers follow your cycle slider.
          </div>
          <div>
            • Split line indicates the support/query boundary used for precomputed meta-adaptation.
          </div>
        </div>
      )}
    </div>
  );
}
