import { ChangeEvent, useMemo, useState } from "react";

import { mediaUrl, searchRetrieval } from "../api/client";
import type { RetrievalResult } from "../types";

export default function RetrievalPage() {
  const [file, setFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setResults([]);
    setError("");
  }

  async function onSearch() {
    if (!file) {
      setError("请选择 Query 图片");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await searchRetrieval(file, topK);
      setResults(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "检索失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-flow">
      <header className="page-header">
        <div className="title-stack">
          <h2>行人检索</h2>
          <p className="eyebrow">Retrieval</p>
        </div>
        {error ? <span className="status-error">{error}</span> : null}
      </header>

      <section className="retrieval-layout">
        <div className="panel">
          <div className="section-title">
            <h3>Query 图片</h3>
            <label className="topk-control">
              <span className="topk-label">Top</span>
              <span className="topk-dash" aria-hidden="true">-</span>
              <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                <option value={1}>1</option>
                <option value={5}>5</option>
                <option value={10}>10</option>
              </select>
            </label>
          </div>
          <label className="upload-box">
            <input accept="image/*" type="file" onChange={onFileChange} />
            {previewUrl ? <img src={previewUrl} alt="Query 预览" /> : <span>选择本地 Query 图片</span>}
          </label>
          <button className="primary-action" disabled={loading} type="button" onClick={onSearch}>
            {loading ? "检索中" : "开始检索"}
          </button>
        </div>

        <div className="panel">
          <div className="section-title">
            <h3>Top-K 结果</h3>
            <span>{results.length} 张</span>
          </div>
          {results.length ? (
            <div className="result-grid">
              {results.map((item) => (
                <figure key={`${item.rank}-${item.image_name}`}>
                  <img src={mediaUrl(item.image_url)} alt={item.image_name} />
                  <figcaption>
                    <strong>Rank {item.rank}</strong>
                    <span>{item.score.toFixed(4)}</span>
                  </figcaption>
                </figure>
              ))}
            </div>
          ) : (
            <div className="empty-state retrieval-empty">检索结果将在这里展示</div>
          )}
        </div>
      </section>
    </div>
  );
}
