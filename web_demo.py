"""CivilityLens - Standalone Web Demonstration Interface.

Zero-dependency local web application served via standard library `http.server`.
Provides an instant demonstration environment for BTech viva and evaluations,
featuring the dark navy/teal theme, real dataset metrics, and live model inference.
"""

import http.server
import json
from pathlib import Path
import socketserver
import sys
import urllib.parse

from src.config import (
    BEST_MODEL_PATH,
    DATASET_METADATA_PATH,
    ERROR_ANALYSIS_PATH,
    EXPERIMENT_RESULTS_PATH,
    METRICS_PATH,
)
from src.predict import get_predictor
from src.preprocessing import (
    apply_lemmatization,
    apply_stemming,
    clean_text_conservative,
)

PORT = 8501
PROJECT_ROOT = Path(__file__).resolve().parent


def load_data():
    metadata = {}
    metrics = {}
    error_cases = {}
    experiments = []

    if DATASET_METADATA_PATH.exists():
        with open(DATASET_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    if ERROR_ANALYSIS_PATH.exists():
        with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            error_cases = json.load(f)

    if EXPERIMENT_RESULTS_PATH.exists():
        import pandas as pd
        experiments = pd.read_csv(EXPERIMENT_RESULTS_PATH).to_dict(orient="records")

    return metadata, metrics, error_cases, experiments


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CivilityLens | NLP Cyberbullying Detection</title>
    <style>
        :root {
            --bg-main: #0d1117;
            --bg-card: #161b22;
            --bg-card-alt: #21262d;
            --border: #30363d;
            --text-main: #e6edf3;
            --text-muted: #8b949e;
            --accent-teal: #38bdf8;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-amber: #e3b341;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-main); color: var(--text-main); display: flex; min-height: 100vh; }
        
        /* Sidebar */
        aside {
            width: 270px;
            background: #090d13;
            border-right: 1px solid var(--border);
            padding: 24px 16px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .brand-icon { font-size: 2rem; }
        .brand h2 { font-size: 1.25rem; color: #fff; }
        .nav-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 10px 14px;
            border-radius: 6px;
            text-align: left;
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.2s;
            width: 100%;
        }
        .nav-btn:hover, .nav-btn.active {
            background: var(--bg-card);
            border-color: var(--border);
            color: var(--accent-teal);
            font-weight: 600;
        }

        /* Content Area */
        main { flex: 1; padding: 36px 48px; overflow-y: auto; max-width: 1200px; margin: 0 auto; }
        .hero {
            background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
            border-left: 5px solid var(--accent-teal);
            padding: 24px;
            border-radius: 8px;
            margin-bottom: 28px;
        }
        .hero h1 { font-size: 2rem; margin-bottom: 6px; }
        .hero p { color: var(--accent-teal); font-size: 1.05rem; }

        /* Metric Grid */
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; margin-bottom: 28px; }
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        .metric-title { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
        .metric-val { font-size: 2rem; font-weight: 700; color: var(--accent-teal); margin: 6px 0; }
        .metric-sub { font-size: 0.75rem; color: var(--text-muted); }

        /* Tables & Lists */
        table { width: 100%; border-collapse: collapse; margin-top: 14px; background: var(--bg-card); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
        th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: #1c2128; color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; }
        tr:last-child td { border-bottom: none; }
        .highlight { font-weight: 600; color: var(--accent-teal); }

        /* Live Detection Box */
        .input-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
        }
        textarea {
            width: 100%;
            height: 120px;
            background: #090d13;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            color: #fff;
            font-size: 1rem;
            resize: vertical;
            outline: none;
            margin-bottom: 14px;
        }
        textarea:focus { border-color: var(--accent-teal); }
        .btn-analyze {
            background: #0284c7;
            color: #fff;
            border: none;
            padding: 10px 24px;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-analyze:hover { background: #0369a1; }

        .result-box {
            padding: 20px;
            border-radius: 8px;
            margin-top: 18px;
            display: none;
        }
        .result-box.flagged {
            display: block;
            background: rgba(248, 81, 73, 0.1);
            border: 1px solid var(--accent-red);
        }
        .result-box.safe {
            display: block;
            background: rgba(63, 185, 80, 0.1);
            border: 1px solid var(--accent-green);
        }

        .disclaimer {
            background: rgba(227, 179, 65, 0.1);
            border-left: 4px solid var(--accent-amber);
            padding: 14px 18px;
            border-radius: 6px;
            color: #f0c665;
            font-size: 0.88rem;
            margin-top: 20px;
        }

        .section-view { display: none; }
        .section-view.active { display: block; }
    </style>
</head>
<body>
    <aside>
        <div class="brand">
            <span class="brand-icon">🛡️</span>
            <div>
                <h2>CivilityLens</h2>
                <small style="color: var(--text-muted);">NLP Lab Project</small>
            </div>
        </div>
        <hr style="border-color: var(--border);">
        <button class="nav-btn active" onclick="showTab('overview')">🏛️ Overview</button>
        <button class="nav-btn" onclick="showTab('dataset')">📊 Dataset Explorer</button>
        <button class="nav-btn" onclick="showTab('pipeline')">⚙️ NLP Pipeline</button>
        <button class="nav-btn" onclick="showTab('models')">🔬 Model Lab</button>
        <button class="nav-btn" onclick="showTab('live')">⚡ Live Detection</button>
        <button class="nav-btn" onclick="showTab('errors')">🔍 Error Analysis</button>

        <div style="margin-top: auto; font-size: 0.8rem; color: var(--text-muted); line-height: 1.4;">
            <strong>University Evaluation Build</strong><br>
            • 47,692 tweets evaluated<br>
            • Unbiased Test Split (15%)<br>
            • Real ML predictions
        </div>
    </aside>

    <main>
        <!-- TAB 1: OVERVIEW -->
        <section id="tab-overview" class="section-view active">
            <div class="hero">
                <h1>CivilityLens</h1>
                <p>Detect harmful patterns. Reduce false alarms. Support safer online communities.</p>
            </div>

            <h3 style="margin-bottom: 14px;">Evaluated Test-Set Metrics (Untouched 6,657 Samples)</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-title">Test Accuracy</div>
                    <div class="metric-val">__ACCURACY__%</div>
                    <div class="metric-sub">Overall correct classifications</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Test Precision</div>
                    <div class="metric-val">__PRECISION__%</div>
                    <div class="metric-sub">Harassment flag reliability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Test Recall</div>
                    <div class="metric-val">__RECALL__%</div>
                    <div class="metric-sub">Harassment pattern coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Test F1-Score</div>
                    <div class="metric-val">__F1__</div>
                    <div class="metric-sub">Harmonic mean of P & R</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                <div class="metric-card" style="text-align: left;">
                    <h4 style="color: var(--accent-teal); margin-bottom: 10px;">Active Production Model</h4>
                    <p style="margin-bottom: 6px;"><strong>Classifier:</strong> __MODEL_NAME__</p>
                    <p style="margin-bottom: 6px;"><strong>Feature Extraction:</strong> __VECTORIZER__</p>
                    <p style="margin-bottom: 6px;"><strong>Selection Criteria:</strong> Highest Validation F1 (__VAL_F1__)</p>
                </div>
                <div class="metric-card" style="text-align: left;">
                    <h4 style="color: var(--accent-teal); margin-bottom: 10px;">Dataset Scope</h4>
                    <p style="margin-bottom: 6px;"><strong>Raw Tweet Count:</strong> 47,692 records</p>
                    <p style="margin-bottom: 6px;"><strong>Cleaned Records:</strong> __CLEAN_COUNT__ samples</p>
                    <p style="margin-bottom: 6px;"><strong>Target Mapping:</strong> 6 Original Classes → Binary</p>
                </div>
            </div>

            <div class="disclaimer">
                <strong>⚠️ Academic Principle:</strong> CivilityLens is an assistive moderation-support tool. It computes statistical linguistic affinities and cannot guarantee definitive intent or comprehend complex sarcasm. Human review is recommended for consequential decisions.
            </div>
        </section>

        <!-- TAB 2: DATASET -->
        <section id="tab-dataset" class="section-view">
            <h2>📊 Dataset Explorer & Data Cleaning</h2>
            <p style="color: var(--text-muted); margin-top: 6px; margin-bottom: 20px;">Analysis of the Kaggle Cyberbullying Classification Dataset.</p>

            <div class="metric-grid">
                <div class="metric-card"><div class="metric-title">Raw Records</div><div class="metric-val">47,692</div></div>
                <div class="metric-card"><div class="metric-title">Duplicates Purged</div><div class="metric-val">__DEDUP_COUNT__</div></div>
                <div class="metric-card"><div class="metric-title">Retained Cleaned</div><div class="metric-val">__CLEAN_COUNT__</div></div>
                <div class="metric-card"><div class="metric-title">Original Classes</div><div class="metric-val">6</div></div>
            </div>

            <h3 style="margin-top: 20px;">Original Category Breakdown</h3>
            <table>
                <thead><tr><th>Category</th><th>Count</th><th>Derived Target</th></tr></thead>
                <tbody>
                    <tr><td>Religion</td><td>7,998</td><td>1 (Cyberbullying-related)</td></tr>
                    <tr><td>Age</td><td>7,992</td><td>1 (Cyberbullying-related)</td></tr>
                    <tr><td>Gender</td><td>7,973</td><td>1 (Cyberbullying-related)</td></tr>
                    <tr><td>Ethnicity</td><td>7,961</td><td>1 (Cyberbullying-related)</td></tr>
                    <tr><td>Not Cyberbullying</td><td>7,945</td><td>0 (Non-cyberbullying)</td></tr>
                    <tr><td>Other Cyberbullying</td><td>7,823</td><td>1 (Cyberbullying-related)</td></tr>
                </tbody>
            </table>
        </section>

        <!-- TAB 3: PIPELINE -->
        <section id="tab-pipeline" class="section-view">
            <h2>⚙️ End-to-End NLP Architecture</h2>
            <p style="color: var(--text-muted); margin-top: 6px; margin-bottom: 20px;">The 10-stage processing pipeline that prepares and classifies raw text.</p>
            <table>
                <thead><tr><th>Stage</th><th>Mechanism</th><th>Purpose</th></tr></thead>
                <tbody>
                    <tr><td><strong>1. Ingestion</strong></td><td>Pandas CSV reader & schema validation</td><td>Guarantees valid text and category columns</td></tr>
                    <tr><td><strong>2. Cleaning</strong></td><td>Deduplication & conflicting label purge</td><td>Prevents evaluation contamination & label leakage</td></tr>
                    <tr><td><strong>3. Target Mapping</strong></td><td>Six original classes mapped to binary {0, 1}</td><td>Prioritizes high-recall moderation triage</td></tr>
                    <tr><td><strong>4. Normalisation</strong></td><td>Lowercase, regex for URLs and @mentions</td><td>Normalizes non-discriminative Twitter metadata</td></tr>
                    <tr><td><strong>5. Stopwords</strong></td><td>Controlled filtering preserving negations</td><td>Retains 'not', 'no', 'never' to maintain sentence polarity</td></tr>
                    <tr><td><strong>6. Vectorisation</strong></td><td>TF-IDF Character N-grams (3,5)</td><td>Resistant to deliberate spelling alterations and leetspeak</td></tr>
                    <tr><td><strong>7. Splitting</strong></td><td>70% Train, 15% Val, 15% Test</td><td>Prevents data snooping by locking test data</td></tr>
                    <tr><td><strong>8. Inference</strong></td><td>Fitted Scikit-learn Pipeline</td><td>Combines extraction and Logistic Regression weights</td></tr>
                </tbody>
            </table>
        </section>

        <!-- TAB 4: MODEL LAB -->
        <section id="tab-models" class="section-view">
            <h2>🔬 Model Laboratory & Experiments</h2>
            <p style="color: var(--text-muted); margin-top: 6px; margin-bottom: 20px;">Empirical validation comparisons across architectures and feature extractors.</p>
            <table>
                <thead>
                    <tr>
                        <th>Exp ID</th>
                        <th>Model</th>
                        <th>Vectorizer</th>
                        <th>Val Accuracy</th>
                        <th>Val Precision</th>
                        <th>Val Recall</th>
                        <th>Val F1</th>
                    </tr>
                </thead>
                <tbody>
                    __EXPERIMENT_ROWS__
                </tbody>
            </table>

            <h3 style="margin-top: 30px;">Untouched Test Confusion Matrix</h3>
            <div style="max-width: 500px; margin-top: 14px;">
                <table>
                    <thead><tr><th></th><th>Pred Non-Bullying</th><th>Pred Bullying</th></tr></thead>
                    <tbody>
                        <tr><td><strong>Actual Non-Bullying</strong></td><td class="highlight">TN: __TN__</td><td>FP: __FP__</td></tr>
                        <tr><td><strong>Actual Bullying</strong></td><td>FN: __FN__</td><td class="highlight">TP: __TP__</td></tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- TAB 5: LIVE DETECTION -->
        <section id="tab-live" class="section-view">
            <h2>⚡ Live Cyberbullying Detection Workbench</h2>
            <p style="color: var(--text-muted); margin-top: 6px; margin-bottom: 20px;">Enter any message below to run real-time inference through the trained pipeline.</p>

            <div class="input-card">
                <textarea id="liveInput" placeholder="Enter social media message or comment..."></textarea>
                <button class="btn-analyze" onclick="analyzeText()">🚀 Analyze Text</button>
                <div id="resultBox" class="result-box">
                    <h3 id="resTitle" style="margin-bottom: 8px;"></h3>
                    <p id="resDesc" style="margin-bottom: 8px;"></p>
                    <p id="resStats" style="font-size: 0.9rem; color: var(--text-muted);"></p>
                </div>
            </div>

            <div class="disclaimer">
                <strong>Notice:</strong> This live workbench communicates with the locally saved Scikit-learn pipeline artifact. No external APIs or cloud models are contacted.
            </div>
        </section>

        <!-- TAB 6: ERROR ANALYSIS -->
        <section id="tab-errors" class="section-view">
            <h2>🔍 Error Analysis & Failure Taxonomy</h2>
            <p style="color: var(--text-muted); margin-top: 6px; margin-bottom: 20px;">Actual misclassified test cases from the 6,657-sample evaluation set.</p>

            <h3 style="color: var(--accent-red); margin-top: 14px;">False Positives (False Alarms)</h3>
            <p style="color: var(--text-muted); font-size: 0.9rem;">Harmless text misclassified as harassment due to sensitive keywords.</p>
            <table>
                <thead><tr><th>Original Category</th><th>Tweet Text</th></tr></thead>
                <tbody>__FP_ROWS__</tbody>
            </table>

            <h3 style="color: var(--accent-amber); margin-top: 24px;">False Negatives (Missed Attacks)</h3>
            <p style="color: var(--text-muted); font-size: 0.9rem;">Cyberbullying instances missed by the classifier due to subtle or implicit phrasing.</p>
            <table>
                <thead><tr><th>Original Category</th><th>Tweet Text</th></tr></thead>
                <tbody>__FN_ROWS__</tbody>
            </table>
        </section>
    </main>

    <script>
        function showTab(tabName) {
            document.querySelectorAll('.section-view').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabName).classList.add('active');
            event.target.classList.add('active');
        }

        async function analyzeText() {
            const text = document.getElementById('liveInput').value.trim();
            if (!text) { alert('Please enter text to analyze.'); return; }

            const resBox = document.getElementById('resultBox');
            resBox.style.display = 'block';
            resBox.className = 'result-box';
            document.getElementById('resTitle').innerText = 'Analyzing...';
            document.getElementById('resDesc').innerText = 'Processing through NLP pipeline...';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const data = await response.json();

                if (data.is_cyberbullying) {
                    resBox.className = 'result-box flagged';
                    document.getElementById('resTitle').innerText = '⚠️ ' + (data.headline || 'Cyberbullying detected.');
                    document.getElementById('resTitle').style.color = '#f85149';
                    document.getElementById('resDesc').innerText = 'This message exhibits linguistic patterns strongly associated with online harassment.';
                } else {
                    resBox.className = 'result-box safe';
                    document.getElementById('resTitle').innerText = '✅ Non-Cyberbullying Content';
                    document.getElementById('resTitle').style.color = '#3fb950';
                    document.getElementById('resDesc').innerText = 'No significant indicators of harassment were identified in this message.';
                }

                const subInfo = data.subtype_confidence ? ` | Subtype: ${data.subtype_name} (${data.subtype_confidence}%)` : '';
                document.getElementById('resStats').innerText = 
                    `Confidence: ${data.confidence_display} | Risk: ${data.risk_level}${subInfo} | Model: ${data.model_architecture}`;
            } catch (err) {
                document.getElementById('resTitle').innerText = 'Error';
                document.getElementById('resDesc').innerText = err.message;
            }
        }
    </script>
</body>
</html>
"""


class CivilityRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            metadata, metrics, error_cases, experiments = load_data()

            test_m = metrics.get("final_test_metrics", {})
            cm = test_m.get("confusion_matrix", {})

            # Prepare experiment rows
            exp_rows = ""
            for e in experiments:
                exp_rows += (
                    f"<tr><td>{e.get('Experiment ID')}</td>"
                    f"<td>{e.get('Model')}</td>"
                    f"<td>{e.get('Vectorizer')}</td>"
                    f"<td>{e.get('Val Accuracy'):.4f}</td>"
                    f"<td>{e.get('Val Precision'):.4f}</td>"
                    f"<td>{e.get('Val Recall'):.4f}</td>"
                    f"<td class='highlight'>{e.get('Val F1-Score'):.4f}</td></tr>"
                )

            # Prepare FP and FN rows
            fp_rows = "".join(
                f"<tr><td>{c.get('original_category')}</td><td>{c.get('text')}</td></tr>"
                for c in error_cases.get("false_positives", [])[:6]
            )
            fn_rows = "".join(
                f"<tr><td>{c.get('original_category')}</td><td>{c.get('text')}</td></tr>"
                for c in error_cases.get("false_negatives", [])[:6]
            )

            html = HTML_TEMPLATE
            html = html.replace("__ACCURACY__", f"{test_m.get('accuracy', 0)*100:.1f}")
            html = html.replace("__PRECISION__", f"{test_m.get('precision', 0)*100:.1f}")
            html = html.replace("__RECALL__", f"{test_m.get('recall', 0)*100:.1f}")
            html = html.replace("__F1__", f"{test_m.get('f1_score', 0):.4f}")
            html = html.replace("__MODEL_NAME__", metrics.get("best_model_name", "Logistic Regression"))
            html = html.replace("__VECTORIZER__", metrics.get("best_vectorizer", "TF-IDF Char N-grams (3,5)"))
            html = html.replace("__VAL_F1__", f"{metrics.get('validation_metrics', {}).get('f1_score', 0):.4f}")
            html = html.replace("__CLEAN_COUNT__", f"{metadata.get('cleaning_summary', {}).get('final_records_retained', 0):,}")
            html = html.replace("__DEDUP_COUNT__", f"{metadata.get('cleaning_summary', {}).get('total_records_removed', 0):,}")
            html = html.replace("__TN__", f"{cm.get('tn', 0):,}")
            html = html.replace("__FP__", f"{cm.get('fp', 0):,}")
            html = html.replace("__FN__", f"{cm.get('fn', 0):,}")
            html = html.replace("__TP__", f"{cm.get('tp', 0):,}")
            html = html.replace("__EXPERIMENT_ROWS__", exp_rows)
            html = html.replace("__FP_ROWS__", fp_rows)
            html = html.replace("__FN_ROWS__", fn_rows)

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                text = data.get("text", "")
                predictor = get_predictor()
                result = predictor.predict(text)
                response_bytes = json.dumps(result).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server():
    """Start local web demo server."""
    with socketserver.TCPServer(("", PORT), CivilityRequestHandler) as httpd:
        print(f"===========================================================")
        print(f"CivilityLens Web Application running at http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop.")
        print(f"===========================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    start_server()
