import { ChangeEvent, useEffect, useState } from "react";

import { computeVisualResult, fetchExperimentCurves, fetchExperimentResults, mediaUrl } from "../api/client";
import LineChart from "../components/LineChart";
import MetricTable from "../components/MetricTable";
import type { ExperimentCurve, ExperimentResult, VisualResult } from "../types";

const visualItems: Array<[keyof VisualResult, string]> = [
  ["original", "原图"],
  ["LH", "LH 高频图"],
  ["HL", "HL 高频图"],
  ["HH", "HH 高频图"],
  ["attention_overlay", "Heatmap 叠加图"],
];

const attentionWeightItems = [
  ["branch_1x1", "1x1"],
  ["branch_3x3", "3x3"],
  ["branch_5x5", "5x5"],
] as const;

export default function DataDisplayPage() {
  const [visual, setVisual] = useState<VisualResult | null>(null);
  const [dataset, setDataset] = useState("market1501");
  const [results, setResults] = useState<ExperimentResult[]>([]);
  const [curves, setCurves] = useState<ExperimentCurve[]>([]);
  const [visualLoading, setVisualLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchExperimentCurves().then(setCurves).catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    fetchExperimentResults(dataset).then(setResults).catch((err: Error) => setError(err.message));
  }, [dataset]);

  async function onVisualFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setError("");
    setVisualLoading(true);
    try {
      setVisual(await computeVisualResult(file));
    } catch (err) {
      setVisual(null);
      setError(err instanceof Error ? err.message : "图片可视化计算失败");
    } finally {
      setVisualLoading(false);
    }
  }

  return (
    <div className="page-flow">
      <header className="page-header">
        <div className="title-stack">
          <h2>数据展示</h2>
          <p className="eyebrow">Display</p>
        </div>
        {error ? <span className="status-error">{error}</span> : null}
      </header>

      <section className="panel">
        <div className="section-title">
          <h3>单张图片可视化</h3>
          <label className="file-picker">
            <input accept="image/*" type="file" onChange={onVisualFileChange} />
            <span>选择本地图片</span>
          </label>
        </div>
        {visualLoading ? (
          <div className="empty-state">正在实时计算可视化结果</div>
        ) : visual ? (
          <>
            <div className="visual-grid">
              {visualItems.map(([key, label]) => (
                <figure key={key}>
                  {visual[key] ? <img src={mediaUrl(visual[key] as string)} alt={label} /> : <div className="image-placeholder" />}
                  <figcaption>{label}</figcaption>
                </figure>
              ))}
            </div>
            {visual.attention_weights ? (
              <div className="attention-weights" aria-label="多尺度注意力分支权重">
                {attentionWeightItems.map(([key, label]) => {
                  const value = visual.attention_weights?.[key] ?? 0;
                  return (
                    <div className="weight-row" key={key}>
                      <span>{label}</span>
                      <div className="weight-track">
                        <i style={{ width: `${Math.max(4, value * 100)}%` }} />
                      </div>
                      <strong>{value.toFixed(3)}</strong>
                    </div>
                  );
                })}
              </div>
            ) : null}
          </>
        ) : (
          <div className="empty-state visual-empty">
            <span className="upload-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" focusable="false">
                <path d="M12 17V7" />
                <path d="M8 11l4-4 4 4" />
              </svg>
            </span>
            <span>请选择本地图片生成可视化结果</span>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="section-title">
          <h3>实验结果</h3>
          <div className="segmented" role="group" aria-label="数据集">
            <button className={dataset === "market1501" ? "active" : ""} onClick={() => setDataset("market1501")} type="button">
              Market1501
            </button>
            <button className={dataset === "dukemtmc" ? "active" : ""} onClick={() => setDataset("dukemtmc")} type="button">
              DukeMTMC
            </button>
          </div>
        </div>
        <MetricTable rows={results} />
      </section>

      <div className="chart-grid">
        <LineChart title="Market1501 训练准确率曲线" curves={curves} valueKey="accuracy" />
        <LineChart title="Market1501 训练损失曲线" curves={curves} valueKey="loss" />
      </div>
    </div>
  );
}
