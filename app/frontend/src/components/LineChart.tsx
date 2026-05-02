import { useState } from "react";

import type { ExperimentCurve } from "../types";

const COLORS = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272"];
const GRID_LINES = [52, 93, 134, 175];
const CHART_WIDTH = 640;
const CHART_HEIGHT = 220;
const CHART_PAD = 28;
const AXIS_LEFT = CHART_PAD;
const AXIS_RIGHT = CHART_WIDTH - CHART_PAD;
const AXIS_BOTTOM = CHART_HEIGHT - CHART_PAD;

type Props = {
  title: string;
  curves: ExperimentCurve[];
  valueKey: "accuracy" | "loss";
};

function yCoordinate(value: number, min: number, max: number): number {
  const range = Math.max(max - min, 1e-6);
  return CHART_HEIGHT - CHART_PAD - ((value - min) / range) * (CHART_HEIGHT - CHART_PAD * 2);
}

function points(values: number[], epochs: number[], min: number, max: number): string {
  const epochMin = Math.min(...epochs);
  const epochMax = Math.max(...epochs);
  const epochRange = Math.max(epochMax - epochMin, 1);

  return values
    .map((value, index) => {
      const x = CHART_PAD + ((epochs[index] - epochMin) / epochRange) * (CHART_WIDTH - CHART_PAD * 2);
      const y = yCoordinate(value, min, max);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export default function LineChart({ title, curves, valueKey }: Props) {
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const active = curves.filter((curve) => curve.epochs.length && curve[valueKey].length);
  if (!active.length) {
    return (
      <section className="panel">
        <h3>{title}</h3>
        <div className="empty-state">暂无训练曲线日志</div>
      </section>
    );
  }

  const values = active.flatMap((curve) => curve[valueKey]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const series = active.map((curve, index) => ({
    curve,
    color: COLORS[index % COLORS.length],
    index,
    points: points(curve[valueKey], curve.epochs, min, max),
  }));
  const highlightedIndex = focusedIndex !== null && focusedIndex < series.length ? focusedIndex : null;
  const orderedSeries =
    highlightedIndex === null ? series : [...series.filter((item) => item.index !== highlightedIndex), series[highlightedIndex]].filter(Boolean);

  function seriesClassName(index: number) {
    if (highlightedIndex === null) {
      return "chart-series";
    }
    return index === highlightedIndex ? "chart-series focused" : "chart-series muted";
  }

  return (
    <section className="panel">
      <div className="section-title">
        <h3>{title}</h3>
      </div>
      <div className="chart-body">
        <svg className="line-chart" viewBox="0 0 640 220" role="img" aria-label={title} onMouseLeave={() => setFocusedIndex(null)}>
          {GRID_LINES.map((y) => (
            <line className="grid-line" key={y} x1={AXIS_LEFT} y1={y} x2={AXIS_RIGHT} y2={y} />
          ))}
          <line className="axis-line" x1={AXIS_LEFT} y1={AXIS_BOTTOM} x2={AXIS_RIGHT} y2={AXIS_BOTTOM} />
          <line className="axis-line" x1={AXIS_LEFT} y1={CHART_PAD} x2={AXIS_LEFT} y2={AXIS_BOTTOM} />
          {orderedSeries.map(({ curve, color, index, points: curvePoints }) => (
            <polyline
              key={curve.experiment}
              className={seriesClassName(index)}
              points={curvePoints}
              stroke={color}
            />
          ))}
          {series.map(({ curve, index, points: curvePoints }) => (
            <polyline
              aria-label={curve.experiment}
              className="chart-hit-area"
              key={`${curve.experiment}-hit-area`}
              onMouseEnter={() => setFocusedIndex(index)}
              points={curvePoints}
              role="presentation"
            />
          ))}
        </svg>
        <div className="legend">
          {series.map(({ curve, color, index }) => (
            <button
              className={`legend-item ${highlightedIndex === null ? "" : highlightedIndex === index ? "focused" : "muted"}`}
              key={curve.experiment}
              onBlur={() => setFocusedIndex(null)}
              onFocus={() => setFocusedIndex(index)}
              onMouseEnter={() => setFocusedIndex(index)}
              onMouseLeave={() => setFocusedIndex(null)}
              type="button"
            >
              <i style={{ backgroundColor: color }} />
              {curve.experiment}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
