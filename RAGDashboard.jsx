import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

// ─── Icons ────────────────────────────────────────────────────────────────────
const SearchIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
  </svg>
);
const UploadIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
);
const BoltIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
  </svg>
);
const DocIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14,2 14,8 20,8"/>
  </svg>
);
const ChipIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
    <rect x="9" y="9" width="6" height="6"/><rect x="2" y="2" width="20" height="20" rx="2"/>
    <line x1="9" y1="2" x2="9" y2="6"/><line x1="15" y1="2" x2="15" y2="6"/>
    <line x1="9" y1="18" x2="9" y2="22"/><line x1="15" y1="18" x2="15" y2="22"/>
    <line x1="2" y1="9" x2="6" y2="9"/><line x1="2" y1="15" x2="6" y2="15"/>
    <line x1="18" y1="9" x2="22" y2="9"/><line x1="18" y1="15" x2="22" y2="15"/>
  </svg>
);

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;500;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Syne', sans-serif;
    background: #0a0a0f;
    color: #e8e8f0;
    min-height: 100vh;
  }

  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a24;
    --border: rgba(255,255,255,0.07);
    --accent: #7c6bff;
    --accent2: #ff6b9d;
    --accent3: #6bffd4;
    --text: #e8e8f0;
    --muted: #7a7a8c;
    --mono: 'JetBrains Mono', monospace;
  }

  .app {
    display: grid;
    grid-template-columns: 280px 1fr;
    grid-template-rows: 60px 1fr;
    height: 100vh;
    overflow: hidden;
  }

  /* Header */
  .header {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    z-index: 10;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.3px;
  }
  .logo-badge {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    width: 28px; height: 28px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
  }
  .header-stats {
    display: flex;
    gap: 20px;
  }
  .hstat {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--muted);
    font-family: var(--mono);
  }
  .hstat-val {
    color: var(--accent3);
    font-weight: 600;
  }
  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent3);
    box-shadow: 0 0 8px var(--accent3);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Sidebar */
  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
  }
  .sidebar-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 12px 8px 6px;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
    transition: all 0.15s;
    border: 1px solid transparent;
  }
  .nav-item:hover { background: var(--surface2); color: var(--text); }
  .nav-item.active {
    background: rgba(124, 107, 255, 0.12);
    border-color: rgba(124, 107, 255, 0.25);
    color: var(--accent);
  }
  .nav-icon { opacity: 0.7; }
  .nav-item.active .nav-icon { opacity: 1; }

  .perf-card {
    margin-top: auto;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
  }
  .perf-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }
  .perf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 12px;
  }
  .perf-label { color: var(--muted); }
  .perf-val {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--accent3);
  }
  .perf-bar {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-bottom: 12px;
    overflow: hidden;
  }
  .perf-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
  }

  /* Main */
  .main {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Tab panels */
  .panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  /* Query panel */
  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--muted);
    text-align: center;
  }
  .empty-icon {
    width: 56px; height: 56px;
    border-radius: 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    margin-bottom: 8px;
  }
  .empty-title { font-size: 17px; font-weight: 700; color: var(--text); }
  .empty-sub { font-size: 13px; max-width: 320px; line-height: 1.6; }
  .suggestion-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 8px;
  }
  .pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .pill:hover { border-color: var(--accent); color: var(--accent); }

  .msg { display: flex; flex-direction: column; gap: 8px; animation: fadeUp 0.3s ease; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, rgba(124,107,255,0.2), rgba(255,107,157,0.1));
    border: 1px solid rgba(124,107,255,0.3);
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    max-width: 70%;
    font-size: 14px;
  }
  .msg-assistant {
    align-self: flex-start;
    max-width: 90%;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .msg-bubble {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px 14px 14px 14px;
    padding: 14px 16px;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
  }
  .sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .source-tag {
    display: flex;
    align-items: center;
    gap: 5px;
    background: rgba(107, 255, 212, 0.07);
    border: 1px solid rgba(107, 255, 212, 0.2);
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 11px;
    font-family: var(--mono);
    color: var(--accent3);
    cursor: pointer;
    transition: all 0.15s;
  }
  .source-tag:hover { background: rgba(107, 255, 212, 0.14); }
  .perf-badge {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
    padding: 0 4px;
  }
  .loading-dots {
    display: flex;
    gap: 5px;
    padding: 14px 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px 14px 14px 14px;
    align-self: flex-start;
  }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: bounce 1.2s infinite;
  }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-8px); }
  }

  /* Input */
  .input-bar {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    background: var(--bg);
  }
  .input-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 10px 12px 10px 16px;
    transition: border-color 0.15s;
  }
  .input-row:focus-within { border-color: rgba(124,107,255,0.4); }
  textarea {
    flex: 1;
    background: none;
    border: none;
    outline: none;
    resize: none;
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    color: var(--text);
    line-height: 1.5;
    min-height: 22px;
    max-height: 120px;
  }
  textarea::placeholder { color: var(--muted); }
  .send-btn {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none;
    border-radius: 9px;
    width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    color: white;
    transition: all 0.15s;
    flex-shrink: 0;
  }
  .send-btn:hover { transform: scale(1.05); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  /* Upload panel */
  .upload-panel {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }
  .drop-zone {
    border: 2px dashed rgba(124,107,255,0.3);
    border-radius: 16px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: rgba(124,107,255,0.03);
  }
  .drop-zone:hover, .drop-zone.drag { 
    border-color: var(--accent);
    background: rgba(124,107,255,0.07);
  }
  .drop-icon {
    width: 52px; height: 52px;
    border-radius: 14px;
    background: rgba(124,107,255,0.15);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px;
    font-size: 22px;
  }
  .drop-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
  .drop-sub { font-size: 13px; color: var(--muted); }
  .file-types {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 16px;
  }
  .ft-badge {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    font-family: var(--mono);
    color: var(--muted);
  }
  .file-list { margin-top: 20px; display: flex; flex-direction: column; gap: 8px; }
  .file-item {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }
  .file-info { flex: 1; min-width: 0; }
  .file-name { font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .file-size { font-size: 11px; color: var(--muted); font-family: var(--mono); margin-top: 2px; }
  .file-status { font-size: 11px; font-family: var(--mono); }
  .status-ready { color: var(--accent); }
  .status-done { color: var(--accent3); }
  .status-error { color: var(--accent2); }
  .upload-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 13px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    color: white;
    cursor: pointer;
    transition: all 0.15s;
    margin-top: 16px;
  }
  .upload-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .upload-btn:not(:disabled):hover { transform: translateY(-1px); }
  .ingest-result {
    margin-top: 20px;
    background: var(--surface2);
    border: 1px solid rgba(107,255,212,0.2);
    border-radius: 12px;
    padding: 16px;
    font-family: var(--mono);
    font-size: 12px;
  }
  .ingest-title { 
    font-size: 11px; 
    font-weight: 700; 
    letter-spacing: 1px; 
    text-transform: uppercase; 
    color: var(--accent3); 
    margin-bottom: 10px; 
  }
  .ingest-row { 
    display: flex; 
    justify-content: space-between; 
    color: var(--muted); 
    margin-bottom: 5px;
  }
  .ingest-val { color: var(--text); }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
`;

// ─── Mock performance data ────────────────────────────────────────────────────
const PERF = {
  accuracy: 92,
  latency: "1.2s",
  quality: "+35%",
  indexed: 0,
};

export default function RAGDashboard() {
  const [tab, setTab] = useState("query");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [ingestResult, setIngestResult] = useState(null);
  const [stats, setStats] = useState({ indexed: 0 });
  const [drag, setDrag] = useState(false);
  const chatRef = useRef(null);
  const fileRef = useRef(null);
  const taRef = useRef(null);

  // Fetch stats
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setStats({ indexed: d.indexed_documents || 0, model: d.llm_provider }))
      .catch(() => {});
  }, [ingestResult]);

  // Auto-scroll
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const handleQuery = async (q) => {
    const question = q || input.trim();
    if (!question) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: question }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5, include_sources: true }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        perf: data.performance || {},
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "⚠️ Could not connect to the RAG backend. Make sure the FastAPI server is running on port 8000.",
        sources: [],
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const handleFileDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    const dropped = Array.from(e.dataTransfer.files);
    addFiles(dropped);
  };

  const addFiles = (newFiles) => {
    const mapped = newFiles.map(f => ({ file: f, status: "ready" }));
    setFiles(prev => [...prev, ...mapped]);
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    const form = new FormData();
    files.forEach(f => form.append("files", f.file));

    try {
      const res = await fetch(`${API_BASE}/ingest`, { method: "POST", body: form });
      const data = await res.json();
      setIngestResult(data);
      setFiles(prev => prev.map(f => ({ ...f, status: "done" })));
    } catch (e) {
      setIngestResult({ error: "Failed to connect to backend" });
      setFiles(prev => prev.map(f => ({ ...f, status: "error" })));
    } finally {
      setUploading(false);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  const suggestions = [
    "Summarize the main findings",
    "What are the key risks mentioned?",
    "Compare the different approaches",
    "List all recommendations",
  ];

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        {/* Header */}
        <header className="header">
          <div className="logo">
            <div className="logo-badge">🔍</div>
            <span>DocRAG</span>
            <span style={{ color: "var(--muted)", fontWeight: 400, fontSize: 13 }}>
              / Intelligent Document Analysis
            </span>
          </div>
          <div className="header-stats">
            <div className="hstat">
              <div className="status-dot" />
              <span style={{ color: "var(--accent3)" }}>Live</span>
            </div>
            <div className="hstat">
              <DocIcon />
              <span><span className="hstat-val">{stats.indexed}</span> docs indexed</span>
            </div>
            <div className="hstat">
              <BoltIcon />
              <span><span className="hstat-val">1.2s</span> avg latency</span>
            </div>
            <div className="hstat">
              <ChipIcon />
              <span><span className="hstat-val">92%</span> accuracy</span>
            </div>
          </div>
        </header>

        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">Navigation</div>
          <div className={`nav-item ${tab === "query" ? "active" : ""}`} onClick={() => setTab("query")}>
            <span className="nav-icon"><SearchIcon /></span>
            Query Documents
          </div>
          <div className={`nav-item ${tab === "upload" ? "active" : ""}`} onClick={() => setTab("upload")}>
            <span className="nav-icon"><UploadIcon /></span>
            Upload & Index
          </div>

          <div className="sidebar-section" style={{ marginTop: 8 }}>Performance</div>
          <div className="perf-card">
            <div className="perf-title">System Metrics</div>
            <div className="perf-row">
              <span className="perf-label">Accuracy</span>
              <span className="perf-val">92%</span>
            </div>
            <div className="perf-bar">
              <div className="perf-bar-fill" style={{ width: "92%" }} />
            </div>
            <div className="perf-row">
              <span className="perf-label">Latency</span>
              <span className="perf-val">1.2s</span>
            </div>
            <div className="perf-bar">
              <div className="perf-bar-fill" style={{ width: "24%", background: "linear-gradient(90deg, var(--accent3), var(--accent))" }} />
            </div>
            <div className="perf-row">
              <span className="perf-label">Quality Δ</span>
              <span className="perf-val">+35%</span>
            </div>
            <div className="perf-bar">
              <div className="perf-bar-fill" style={{ width: "35%", background: "linear-gradient(90deg, var(--accent2), var(--accent))" }} />
            </div>
            <div className="perf-row" style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
              <span className="perf-label">Stack</span>
              <span className="perf-val">LangChain</span>
            </div>
            <div className="perf-row">
              <span className="perf-label">VectorDB</span>
              <span className="perf-val">ChromaDB</span>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="main">
          {tab === "query" && (
            <div className="panel">
              <div className="chat-area" ref={chatRef}>
                {messages.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">🔍</div>
                    <div className="empty-title">Ask anything about your documents</div>
                    <div className="empty-sub">
                      Upload documents first, then ask questions. The RAG pipeline uses hybrid search + reranking for best accuracy.
                    </div>
                    <div className="suggestion-pills">
                      {suggestions.map(s => (
                        <div key={s} className="pill" onClick={() => handleQuery(s)}>{s}</div>
                      ))}
                    </div>
                  </div>
                ) : (
                  messages.map((m, i) => (
                    <div key={i} className="msg">
                      {m.role === "user" ? (
                        <div className="msg-user">{m.content}</div>
                      ) : (
                        <div className="msg-assistant">
                          <div className="msg-bubble">{m.content}</div>
                          {m.sources?.length > 0 && (
                            <div className="sources-row">
                              {m.sources.map((s, j) => (
                                <div key={j} className="source-tag" title={s.excerpt}>
                                  <DocIcon />
                                  {s.filename} · {(s.relevance_score * 100).toFixed(0)}%
                                </div>
                              ))}
                            </div>
                          )}
                          {m.perf && (
                            <div className="perf-badge">
                              <span>⚡ {m.perf.total_time_ms}ms</span>
                              {m.perf.semantic_hits && <span>📚 {m.perf.semantic_hits} chunks searched</span>}
                              {m.perf.cost_usd > 0 && <span>💰 ${m.perf.cost_usd.toFixed(4)}</span>}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))
                )}
                {loading && (
                  <div className="loading-dots">
                    <div className="dot" />
                    <div className="dot" />
                    <div className="dot" />
                  </div>
                )}
              </div>
              <div className="input-bar">
                <div className="input-row">
                  <textarea
                    ref={taRef}
                    rows={1}
                    placeholder="Ask a question about your documents..."
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                  />
                  <button className="send-btn" onClick={() => handleQuery()} disabled={loading || !input.trim()}>
                    <SearchIcon />
                  </button>
                </div>
              </div>
            </div>
          )}

          {tab === "upload" && (
            <div className="upload-panel">
              <div
                className={`drop-zone ${drag ? "drag" : ""}`}
                onDragOver={e => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={handleFileDrop}
                onClick={() => fileRef.current?.click()}
              >
                <input
                  ref={fileRef}
                  type="file"
                  multiple
                  style={{ display: "none" }}
                  accept=".pdf,.docx,.txt,.html,.md,.csv"
                  onChange={e => addFiles(Array.from(e.target.files))}
                />
                <div className="drop-icon">📂</div>
                <div className="drop-title">Drop documents here</div>
                <div className="drop-sub">or click to browse — supports up to 50MB per file</div>
                <div className="file-types">
                  {["PDF", "DOCX", "TXT", "HTML", "MD", "CSV"].map(t => (
                    <span key={t} className="ft-badge">.{t.toLowerCase()}</span>
                  ))}
                </div>
              </div>

              {files.length > 0 && (
                <>
                  <div className="file-list">
                    {files.map((f, i) => (
                      <div key={i} className="file-item">
                        <span style={{ fontSize: 20 }}>
                          {f.file.name.endsWith(".pdf") ? "📄" : f.file.name.endsWith(".docx") ? "📝" : "📃"}
                        </span>
                        <div className="file-info">
                          <div className="file-name">{f.file.name}</div>
                          <div className="file-size">{formatSize(f.file.size)}</div>
                        </div>
                        <span className={`file-status status-${f.status}`}>
                          {f.status === "ready" ? "Ready" : f.status === "done" ? "✓ Indexed" : "✗ Error"}
                        </span>
                      </div>
                    ))}
                  </div>
                  <button className="upload-btn" onClick={handleUpload} disabled={uploading}>
                    <UploadIcon />
                    {uploading ? "Indexing..." : `Index ${files.length} file${files.length > 1 ? "s" : ""}`}
                  </button>
                </>
              )}

              {ingestResult && (
                <div className="ingest-result">
                  <div className="ingest-title">✓ Ingestion Complete</div>
                  {ingestResult.error ? (
                    <div style={{ color: "var(--accent2)" }}>{ingestResult.error}</div>
                  ) : (
                    <>
                      <div className="ingest-row">
                        <span>Files processed</span>
                        <span className="ingest-val">{ingestResult.files_processed}</span>
                      </div>
                      <div className="ingest-row">
                        <span>Chunks created</span>
                        <span className="ingest-val">{ingestResult.chunks_created}</span>
                      </div>
                      <div className="ingest-row">
                        <span>Chunks skipped (dupe)</span>
                        <span className="ingest-val">{ingestResult.chunks_skipped}</span>
                      </div>
                      <div className="ingest-row">
                        <span>Processing time</span>
                        <span className="ingest-val">{ingestResult.processing_time_s}s</span>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </>
  );
}
