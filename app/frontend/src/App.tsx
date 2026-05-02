import { useState } from "react";

import instituteLogo from "./assets/institute-logo.jpg";
import schoolLogo from "./assets/school-logo.jpg";
import DataDisplayPage from "./pages/DataDisplayPage";
import RetrievalPage from "./pages/RetrievalPage";

type Page = "display" | "retrieval";

export default function App() {
  const [page, setPage] = useState<Page>("display");

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="主导航">
        <div className="badge-stack" aria-label="徽章区域">
          <div className="school-badge" aria-label="校徽">
            <img src={schoolLogo} alt="校徽" />
          </div>
          <div className="school-badge institute-badge" aria-label="院徽">
            <img src={instituteLogo} alt="院徽" />
          </div>
        </div>
        <div className="brand-block">
          <p className="brand-mark">MS-ReID</p>
          <h1>实验展示与行人检索</h1>
        </div>
        <nav className={`nav-tabs ${page === "retrieval" ? "is-retrieval" : "is-display"}`}>
          <span className="nav-indicator" aria-hidden="true" />
          <button
            className={page === "display" ? "active" : ""}
            type="button"
            onClick={() => setPage("display")}
          >
            数据展示
          </button>
          <button
            className={page === "retrieval" ? "active" : ""}
            type="button"
            onClick={() => setPage("retrieval")}
          >
            行人检索
          </button>
        </nav>
      </aside>
      <main className="main-panel">{page === "display" ? <DataDisplayPage /> : <RetrievalPage />}</main>
    </div>
  );
}
