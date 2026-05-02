import type { ExperimentResult } from "../types";

function formatDelta(value: number): string {
  if (value > 0) {
    return `↑ ${value.toFixed(1)}`;
  }
  if (value < 0) {
    return `↓ ${Math.abs(value).toFixed(1)}`;
  }
  return "-";
}

function MetricCell({ value, delta, best }: { value: number; delta: number; best: boolean }) {
  const deltaClass = delta > 0 ? "positive" : delta < 0 ? "negative" : "neutral";
  return (
    <span className={best ? "metric best" : "metric"}>
      <strong>{value.toFixed(1)}</strong>
      <small className={deltaClass}>{formatDelta(delta)}</small>
    </span>
  );
}

function ExperimentName({ name }: { name: string }) {
  const suffix = "(full)";
  if (!name.endsWith(suffix)) {
    return <>{name}</>;
  }

  return (
    <>
      {name.slice(0, -suffix.length)}
      <span className="experiment-suffix">{suffix}</span>
    </>
  );
}

export default function MetricTable({ rows }: { rows: ExperimentResult[] }) {
  if (!rows.length) {
    return <div className="empty-state">暂无可解析的实验结果日志</div>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>实验组</th>
            <th className="metric-heading">Rank-1 (%)</th>
            <th className="metric-heading">mAP (%)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.experiment}>
              <td>
                <ExperimentName name={row.experiment} />
              </td>
              <td className={row.is_best_rank1 ? "best-cell numeric-cell" : "numeric-cell"}>
                <MetricCell value={row.rank1} delta={row.rank1_delta} best={row.is_best_rank1} />
              </td>
              <td className={row.is_best_map ? "best-cell numeric-cell" : "numeric-cell"}>
                <MetricCell value={row.map} delta={row.map_delta} best={row.is_best_map} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
