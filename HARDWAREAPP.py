import re
import numpy as np
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------
# STREAMLIT CONFIG
# -----------------------
st.set_page_config(
    page_title="Hardware Trojan Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------
# GLOBAL CSS STYLING
# -----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;900&display=swap');

:root {
    --bg-deep:      #050a12;
    --bg-panel:     #0a1628;
    --bg-card:      #0f1f3a;
    --accent-cyan:  #00d4ff;
    --accent-green: #00ff9d;
    --accent-red:   #ff3d6b;
    --accent-gold:  #ffd166;
    --text-primary: #e8f4ff;
    --text-muted:   #7a9bbe;
    --border:       rgba(0, 212, 255, 0.18);
    --glow-cyan:    0 0 24px rgba(0,212,255,0.35);
    --glow-green:   0 0 24px rgba(0,255,157,0.35);
    --glow-red:     0 0 24px rgba(255,61,107,0.4);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 40% at 15% 10%, rgba(0,212,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 30% at 85% 80%, rgba(0,255,157,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }

.hero-header {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
}
.hero-header .badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    color: var(--accent-cyan);
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 2px;
    padding: 4px 14px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}
.hero-header h1 {
    font-family: 'Exo 2', sans-serif !important;
    font-size: clamp(2.2rem, 5vw, 4rem) !important;
    font-weight: 900 !important;
    color: var(--text-primary) !important;
    margin: 0 0 0.6rem !important;
}
.hero-header h1 span { color: var(--accent-cyan); }
.hero-header .subtitle {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.04em;
    max-width: 520px;
    margin: 0 auto;
    font-family: 'Share Tech Mono', monospace;
}
.scan-line {
    width: 180px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    margin: 1.8rem auto 0;
    animation: scanPulse 2.5s ease-in-out infinite;
}
@keyframes scanPulse {
    0%, 100% { opacity: 0.3; transform: scaleX(0.5); }
    50%       { opacity: 1;   transform: scaleX(1); }
}

.stat-bar {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    flex-wrap: wrap;
    padding: 1.2rem 1rem 2rem;
}
.stat-pill {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.7rem 1.6rem;
    text-align: center;
}
.stat-pill .val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    color: var(--accent-cyan);
    display: block;
    line-height: 1;
}
.stat-pill .lbl {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 4px;
    display: block;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed rgba(0,212,255,0.35) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan);
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    color: var(--text-muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(0,212,255,0.1) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    border-radius: 5px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--accent-cyan);
    text-transform: uppercase;
    margin: 2.5rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 2rem 2rem 1.6rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
}
.result-card.danger::before {
    background: linear-gradient(90deg, var(--accent-red), var(--accent-gold));
}
.file-name {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.95rem;
    color: var(--accent-cyan);
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.file-name .dot {
    width: 7px; height: 7px;
    background: var(--accent-cyan);
    border-radius: 50%;
    display: inline-block;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; } 50% { opacity: 0.2; }
}

.verdict-clean {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,255,157,0.08);
    border: 1.5px solid var(--accent-green);
    border-radius: 6px;
    padding: 0.55rem 1.4rem;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--accent-green);
    box-shadow: var(--glow-green);
    letter-spacing: 0.04em;
}
.verdict-trojan {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,61,107,0.1);
    border: 1.5px solid var(--accent-red);
    border-radius: 6px;
    padding: 0.55rem 1.4rem;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--accent-red);
    box-shadow: var(--glow-red);
    letter-spacing: 0.04em;
    animation: redPulse 2s ease-in-out infinite;
}
@keyframes redPulse {
    0%,100% { box-shadow: var(--glow-red); }
    50%      { box-shadow: 0 0 40px rgba(255,61,107,0.6); }
}

.score-block {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 1rem;
}
.score-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
}
.score-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.gauge-bar {
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
    width: 100%;
}
.gauge-fill { height: 100%; border-radius: 3px; }
.gauge-fill.safe  { background: linear-gradient(90deg, #00c97e, #00ff9d); }
.gauge-fill.risky { background: linear-gradient(90deg, #ff9d00, #ff3d6b); }

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: var(--bg-panel) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-testid="stCheckbox"] label {
    color: var(--text-muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
}
[data-testid="stCheckbox"] label:hover { color: var(--accent-cyan) !important; }

.info-strip {
    background: rgba(0,212,255,0.05);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 2rem;
    line-height: 1.7;
}
.upload-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.footer-bar {
    text-align: center;
    padding: 2rem 0 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: rgba(122,155,190,0.45);
    text-transform: uppercase;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# DATA STRUCTURES
# -----------------------
@dataclass
class Signal:
    name: str
    type: str
    width: int = 1
    fanin: int = 0
    fanout: int = 0

@dataclass
class Module:
    name: str
    signals: Dict[str, Signal]
    assignments: List[Tuple[str, str]]
    always_blocks: List[str]

# -----------------------
# VERILOG PARSER
# -----------------------
class VerilogParser:
    def __init__(self):
        self.signal_pattern = re.compile(r'(input|output|wire|reg)\s*(?:\[(\d+):(\d+)\])?\s*([\w,\s]+);')
        self.assign_pattern = re.compile(r'assign\s+(\w+)\s*=\s*([^;]+);')
        self.module_pattern = re.compile(r'module\s+(\w+)')

    def parse(self, code: str) -> Module:
        code = re.sub(r'//.*', '', code)
        module_name = self.module_pattern.search(code)
        module_name = module_name.group(1) if module_name else "unknown"
        signals = {}
        assignments = []
        for m in self.signal_pattern.finditer(code):
            stype = m.group(1)
            hi = int(m.group(2)) if m.group(2) else 0
            lo = int(m.group(3)) if m.group(3) else 0
            width = abs(hi - lo) + 1
            for name in m.group(4).split(","):
                signals[name.strip()] = Signal(name.strip(), stype, width)
        for m in self.assign_pattern.finditer(code):
            assignments.append((m.group(1), m.group(2)))
        return Module(module_name, signals, assignments, [])

# -----------------------
# GRAPH BUILDER (networkx-based, no torch-geometric)
# -----------------------
class GraphBuilder:
    def build(self, module: Module):
        G = nx.DiGraph()
        fanin  = defaultdict(int)
        fanout = defaultdict(int)

        for name, sig in module.signals.items():
            G.add_node(name, signal=sig)

        for tgt, expr in module.assignments:
            sources = re.findall(r'\b\w+\b', expr)
            for src in sources:
                if src in module.signals and tgt in module.signals:
                    G.add_edge(src, tgt)
                    fanout[src] += 1
                    fanin[tgt]  += 1

        for name, sig in module.signals.items():
            sig.fanin  = fanin[name]
            sig.fanout = fanout[name]

        return G

# -----------------------
# GNN REPLACED WITH MLP (scikit-learn)
# — same hybrid logic, same output format
# -----------------------
class GNNLikeDetector:
    """
    Replaces torch-geometric GATConv with an MLP over
    per-node graph features (degree, centrality, signal type).
    Trained on synthetic data so it runs without pretrained weights.
    """
    def __init__(self):
        np.random.seed(42)
        # Synthetic training: clean circuits have low avg degree & no isolated nodes
        X_train, y_train = self._generate_synthetic_data()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        self.mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        self.mlp.fit(X_scaled, y_train)

    def _generate_synthetic_data(self):
        X, y = [], []
        for _ in range(300):
            # Clean sample
            X.append([np.random.uniform(0.5, 3),   # avg_degree
                       np.random.uniform(0, 0.1),   # isolated_ratio
                       np.random.uniform(0.3, 0.7), # input_ratio
                       np.random.uniform(0.1, 0.4), # output_ratio
                       np.random.uniform(0, 0.05)]) # suspicious_name_ratio
            y.append(0)
            # Trojan sample
            X.append([np.random.uniform(0.1, 1.5),
                       np.random.uniform(0.2, 0.8),
                       np.random.uniform(0.1, 0.3),
                       np.random.uniform(0.05, 0.2),
                       np.random.uniform(0.1, 0.9)])
            y.append(1)
        return np.array(X), np.array(y)

    def extract_features(self, module: Module, G: nx.DiGraph) -> np.ndarray:
        n = len(module.signals)
        if n == 0:
            return np.zeros((1, 5))

        degrees   = [G.degree(node) for node in G.nodes()]
        avg_deg   = np.mean(degrees) if degrees else 0
        isolated  = sum(1 for s in module.signals.values() if s.fanin == 0 and s.fanout == 0)
        n_input   = sum(1 for s in module.signals.values() if s.type == "input")
        n_output  = sum(1 for s in module.signals.values() if s.type == "output")
        n_susp    = sum(1 for name in module.signals
                        if re.search(r'trojan|trigger|payload', name, re.I))

        return np.array([[
            avg_deg,
            isolated / n,
            n_input  / n,
            n_output / n,
            n_susp   / n
        ]])

    def predict_proba(self, module: Module, G: nx.DiGraph) -> float:
        feats  = self.extract_features(module, G)
        scaled = self.scaler.transform(feats)
        return self.mlp.predict_proba(scaled)[0][1]

# -----------------------
# STATISTICAL DETECTOR  (unchanged logic)
# -----------------------
class StatisticalDetector:
    def analyze(self, module: Module) -> float:
        score = 0
        for name in module.signals:
            if re.search(r'trojan|trigger|payload', name, re.I):
                score += 0.5
        isolated = [s for s in module.signals.values() if s.fanin == 0 and s.fanout == 0]
        score += len(isolated) * 0.1
        return min(score, 1.0)

# -----------------------
# HYBRID DETECTOR  (same 0.6 / 0.4 blend)
# -----------------------
class HybridDetector:
    def __init__(self):
        self.gnn  = GNNLikeDetector()
        self.stat = StatisticalDetector()

    def predict(self, module: Module, G: nx.DiGraph):
        prob       = self.gnn.predict_proba(module, G)
        stat_score = self.stat.analyze(module)
        final      = 0.6 * prob + 0.4 * stat_score
        pred       = "Trojan Detected 🚨" if final > 0.5 else "Clean ✅"
        return pred, final

# -----------------------
# GRAPH VISUALISATION
# -----------------------
def plot_graph(module: Module, G: nx.DiGraph):
    if len(G.nodes()) == 0:
        st.info("No nodes to display.")
        return

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, labels, colors = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y); labels.append(n)
        sig = module.signals.get(n)
        if sig:
            colors.append("#00d4ff" if sig.type == "input"
                          else "#00ff9d" if sig.type == "output"
                          else "#7a9bbe")
        else:
            colors.append("#7a9bbe")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                             line=dict(color='rgba(0,212,255,0.25)', width=1.5),
                             hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                             text=labels, textposition="top center",
                             textfont=dict(family="Share Tech Mono", size=11, color="#e8f4ff"),
                             marker=dict(size=14, color=colors,
                                         line=dict(width=1.5, color='rgba(0,212,255,0.4)')),
                             hoverinfo='text'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.85)',
        margin=dict(l=20, r=20, t=20, b=20), showlegend=False, height=420,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(family="Share Tech Mono", color="#e8f4ff")
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# HERO HEADER
# -----------------------
st.markdown("""
<div class="hero-header">
    <div class="badge">⬡ Hybrid AI · MLP + Statistical Analysis</div>
    <h1>Hardware <span>Trojan</span> Detection</h1>
    <p class="subtitle">// Scan Verilog netlists for malicious logic insertions</p>
    <div class="scan-line"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-bar">
    <div class="stat-pill"><span class="val">MLP</span><span class="lbl">Neural Network</span></div>
    <div class="stat-pill"><span class="val">0.6/0.4</span><span class="lbl">MLP / Stat Weight</span></div>
    <div class="stat-pill"><span class="val">0.50</span><span class="lbl">Detection Threshold</span></div>
    <div class="stat-pill"><span class="val">.v</span><span class="lbl">Supported Format</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-strip">
    Upload one or more Verilog (.v) source files &nbsp;→&nbsp; the system builds a signal-flow graph,
    runs a Graph-feature MLP, and blends with statistical heuristics to produce
    a per-file threat score and verdict.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="upload-header">⬡ &nbsp;Upload Verilog Files</div>', unsafe_allow_html=True)

# -----------------------
# INIT PIPELINE
# -----------------------
parser   = VerilogParser()
builder  = GraphBuilder()
detector = HybridDetector()

uploaded_files = st.file_uploader(
    "Drop .v files here or click to browse",
    type=["v"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# -----------------------
# RESULTS
# -----------------------
if uploaded_files:
    st.markdown('<div class="section-label">⬡ &nbsp;Analysis Results</div>', unsafe_allow_html=True)

    for file in uploaded_files:
        code   = file.read().decode()
        module = parser.parse(code)
        G      = builder.build(module)
        pred, score = detector.predict(module, G)

        is_trojan  = "Trojan" in pred
        card_class = "result-card danger" if is_trojan else "result-card"
        score_pct  = int(score * 100)
        fill_class = "risky" if is_trojan else "safe"
        score_color = "#ff3d6b" if is_trojan else "#00ff9d"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="file-name"><span class="dot"></span>{file.name}</div>
            <div style="display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
                <div>
                    {"<div class='verdict-trojan'>"+pred+"</div>" if is_trojan
                      else "<div class='verdict-clean'>"+pred+"</div>"}
                </div>
                <div class="score-block" style="min-width:180px;">
                    <span class="score-label">Confidence Score</span>
                    <span class="score-value" style="color:{score_color};">{score:.3f}</span>
                    <div class="gauge-bar">
                        <div class="gauge-fill {fill_class}" style="width:{score_pct}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_graph, col_signals = st.columns([1, 1], gap="medium")

        with col_graph:
            if st.checkbox(f"Signal-Flow Graph  ·  {file.name}", key=f"g_{file.name}"):
                st.markdown('<div class="section-label">⬡ &nbsp;Circuit Graph</div>', unsafe_allow_html=True)
                plot_graph(module, G)

        with col_signals:
            if st.checkbox(f"Signal Table  ·  {file.name}", key=f"s_{file.name}"):
                st.markdown('<div class="section-label">⬡ &nbsp;Parsed Signals</div>', unsafe_allow_html=True)
                data = [{
                    "Signal":  s.name,
                    "Type":    s.type,
                    "Width":   s.width,
                    "Fan-in":  s.fanin,
                    "Fan-out": s.fanout
                } for s in module.signals.values()]
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 1rem; color:rgba(122,155,190,0.5);">
        <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">⬡</div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                    letter-spacing:0.18em; text-transform:uppercase;">
            No files uploaded · Awaiting scan target
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-bar">
    Hardware Trojan Detection System &nbsp;·&nbsp; Hybrid MLP + Statistical Analysis &nbsp;·&nbsp; Verilog Netlist Scanner
</div>
""", unsafe_allow_html=True)