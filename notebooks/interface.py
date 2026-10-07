"""Interface: reads every results/*.json written by the model files and builds model_comparison.html"""
import json
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = sorted((HERE / 'results').glob('*.json'))
if not files:
    raise SystemExit("No results found. Run the model files first "
                     "(decisiontreeclassification.py, xgboostclassification.py, "
                     "logisticregressor.py, gradientboostingclassifier.py).")

results = [json.loads(f.read_text(encoding='utf-8')) for f in files]
results.sort(key=lambda r: (r['cv_score'], r['val_acc']), reverse=True)   # best CV accuracy first
best_name = results[0]['name']
print(f"BEST MODEL : {best_name}  ({len(results)} models compared)")

INFO = {
    'Decision Tree': {
        'how': 'Splits the data with a series of if/else questions (e.g. "Sex = female?", "Age < 10?") until it reaches a prediction.',
        'pros': ['Very easy to read and explain', 'No scaling needed', 'Captures non-linear rules'],
        'cons': ['Overfits easily', 'Unstable: small data change = different tree', 'Usually less accurate than ensembles'],
        'type': 'Single tree',
    },
    'Logistic Regression': {
        'how': 'Learns one weight per feature and combines them linearly, then converts the score into a survival probability (sigmoid).',
        'pros': ['Fast, simple, hard to overfit', 'Interpretable coefficients', 'Gives calibrated probabilities'],
        'cons': ['Only linear boundaries', 'Needs scaling / encoding', 'Misses feature interactions (e.g. Sex x Pclass)'],
        'type': 'Linear model',
    },
    'Random Forest': {
        'how': 'Trains many deep trees in parallel on random samples of rows and features, then averages their votes.',
        'pros': ['Robust, few tuning needs', 'Reduces tree overfitting', 'Handles non-linear + interactions'],
        'cons': ['Less interpretable than one tree', 'Larger and slower at prediction', 'Usually slightly behind boosting'],
        'type': 'Bagged trees (parallel)',
    },
    'Gradient Boosting': {
        'how': 'Builds many small trees one after another; each new tree corrects the errors of the previous ones (sklearn implementation).',
        'pros': ['High accuracy on tabular data', 'Handles non-linear + interactions', 'Flexible loss and parameters'],
        'cons': ['Slower to train', 'Many hyperparameters', 'Can overfit if learning_rate / n_estimators badly set'],
        'type': 'Boosted trees (sequential)',
    },
    'XGBoost': {
        'how': 'Optimized gradient boosting: adds L1/L2 regularization, second-order gradients, column subsampling and parallel tree building.',
        'pros': ['Often the best accuracy', 'Built-in regularization', 'Fast and scalable, handles missing values'],
        'cons': ['Most hyperparameters to tune', 'Less interpretable', 'Extra dependency (xgboost)'],
        'type': 'Regularized boosted trees',
    },
}

WHY_BEST = {
    'Decision Tree': 'A single shallow tree generalized best here. The Titanic rules (Sex, Pclass, Age) are simple enough that a small tree captures them without needing an ensemble.',
    'Logistic Regression': 'The linear model generalized best. On a small dataset (~900 rows) with few features, a simple model has low variance and avoids overfitting.',
    'Random Forest': 'Averaging many decorrelated trees gave the most stable predictions, reducing the variance a single tree suffers from.',
    'Gradient Boosting': 'Sequential boosting captured the non-linear interactions (Sex x Pclass x Age) while the tuned learning rate kept overfitting under control.',
    'XGBoost': 'Regularized boosting captured non-linear interactions while its built-in L1/L2 penalties and subsampling limited overfitting.',
}

QUICK = {   # name: (learns, non-linear, interpretability, overfit risk, scaling, speed, tuning)
    'Decision Tree': ('Rules (1 tree)', 'Yes', 'High', 'High', 'No', 'Very fast', 'Low'),
    'Logistic Regression': ('Linear weights', 'No', 'High', 'Low', 'Yes', 'Very fast', 'Low'),
    'Random Forest': ('Trees in parallel', 'Yes', 'Medium', 'Low-Medium', 'No', 'Fast', 'Medium'),
    'Gradient Boosting': ('Trees in sequence', 'Yes', 'Low', 'Medium', 'No', 'Slow', 'High'),
    'XGBoost': ('Regularized trees in sequence', 'Yes', 'Low', 'Medium (regularized)', 'No', 'Fast', 'Highest'),
}
QUICK_LABELS = ['Learns', 'Non-linear', 'Interpretability', 'Overfitting risk', 'Needs scaling', 'Training speed', 'Tuning effort']


def pct(v):
    return f"{v * 100:.2f}%"


rows, bars, cards = "", "", ""
for i, r in enumerate(results):
    cls = ' class="best"' if i == 0 else ''
    badge = ' <span class="badge">BEST</span>' if i == 0 else ''
    gap_cls = 'warn' if r['overfit_gap'] > 0.10 else 'ok'
    rows += (f"<tr{cls}><td>{i + 1}</td><td>{r['name']}{badge}</td><td>{pct(r['cv_score'])} ± {r['cv_std'] * 100:.2f}</td>"
             f"<td>{pct(r['val_acc'])}</td><td>{r['val_f1']:.3f}</td>"
             f"<td class='{gap_cls}'>{r['overfit_gap'] * 100:+.1f} pts</td><td>{r['time']:.1f}s</td></tr>")
    bars += (f"<div class='barrow'><span>{r['name']}</span><div class='track'>"
             f"<div class='fill{' top' if i == 0 else ''}' style='width:{r['val_acc'] * 100:.1f}%'></div></div>"
             f"<b>{pct(r['val_acc'])}</b></div>")
    info = INFO.get(r['name'], {'how': '', 'pros': [], 'cons': [], 'type': ''})
    pros = "".join(f"<li>{p}</li>" for p in info['pros'])
    cons = "".join(f"<li>{c}</li>" for c in info['cons'])
    params = ", ".join(f"{k}={v}" for k, v in r['params'].items())
    cards += (f"<div class='card'><h3>{r['name']} <small>{info['type']}</small></h3><p>{info['how']}</p>"
              f"<div class='cols'><div><h4>Strengths</h4><ul class='pro'>{pros}</ul></div>"
              f"<div><h4>Weaknesses</h4><ul class='con'>{cons}</ul></div></div>"
              f"<p class='params'><b>Best params:</b> {params}</p></div>")

names = [r['name'] for r in results]
quick_head = "".join(f"<th>{n}</th>" for n in names)
quick_rows = ""
for idx, label in enumerate(QUICK_LABELS):
    cells = "".join(f"<td>{QUICK.get(n, ('-',) * 7)[idx]}</td>" for n in names)
    quick_rows += f"<tr><td>{label}</td>{cells}</tr>"

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Titanic - Model Comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f7fb;--fg:#1c2030;--card:#fff;--mut:#667;--acc:#4f46e5;--good:#16a34a;--bad:#dc2626;--line:#e3e6ef}
@media(prefers-color-scheme:dark){:root{--bg:#12141c;--fg:#e8eaf2;--card:#1b1e2a;--mut:#9aa0b5;--line:#2a2e3e}}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--fg);padding:24px;max-width:1100px;margin:auto}
h1{margin:0 0 4px}h2{margin-top:32px}.sub{color:var(--mut);margin-bottom:20px}
.hero{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;border-radius:16px;padding:22px 26px}
.hero h2{margin:0 0 6px;color:#fff}.hero p{margin:4px 0;opacity:.95}
table{width:100%;border-collapse:collapse;background:var(--card);border-radius:12px;overflow:hidden}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:14px}
th{background:var(--line)}tr.best td{font-weight:600;background:rgba(79,70,229,.10)}
.badge{background:var(--acc);color:#fff;border-radius:6px;padding:1px 7px;font-size:11px}
.ok{color:var(--good)}.warn{color:var(--bad)}
.barrow{display:grid;grid-template-columns:170px 1fr 70px;gap:10px;align-items:center;margin:8px 0;font-size:14px}
.track{background:var(--line);border-radius:8px;height:16px;overflow:hidden}
.fill{height:100%;background:#94a3b8}.fill.top{background:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.card h3{margin:0 0 6px}.card small{color:var(--mut);font-weight:400;font-size:12px;margin-left:6px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px}.cols h4{margin:8px 0 2px;font-size:13px}
ul{margin:0;padding-left:18px;font-size:13px}ul.pro li::marker{color:var(--good)}ul.con li::marker{color:var(--bad)}
.params{font-size:12px;color:var(--mut);word-break:break-word}
.box{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 18px;font-size:14px}
.tbl-wrap{overflow-x:auto}
@media(max-width:560px){.grid{grid-template-columns:1fr}.barrow{grid-template-columns:110px 1fr 60px}}
</style></head><body>
<h1>Titanic - Model Comparison</h1>
<div class="sub">__NAMES__ | GridSearchCV, 5-fold</div>

<div class="hero"><h2>Best model: __BEST__</h2>
<p>CV accuracy <b>__BESTCV__</b> | Validation accuracy <b>__BESTVAL__</b></p>
<p>__WHY__</p></div>

<h2>Results</h2>
<div class="tbl-wrap"><table><tr><th>#</th><th>Model</th><th>CV accuracy</th><th>Validation acc.</th><th>F1</th><th>Overfit gap (train - valid)</th><th>Search time</th></tr>__ROWS__</table></div>
<h2>Validation accuracy</h2><div class="box">__BARS__</div>

<h2>Differences between the models</h2>
<div class="grid">__CARDS__</div>

<h2>Quick comparison</h2>
<div class="tbl-wrap"><table><tr><th></th>__QHEAD__</tr>__QROWS__</table></div>

<h2>How to choose</h2>
<div class="box"><ul>
<li><b>Best accuracy</b> on tabular data: XGBoost or Gradient Boosting.</li>
<li><b>Need to explain decisions</b> (reports, stakeholders): Decision Tree or Logistic Regression.</li>
<li><b>Small dataset</b> like Titanic: differences are small; prefer the simplest model within ~1 pt of the best (check the std of CV).</li>
<li>Selection criterion used here: highest CV accuracy (more reliable than a single validation split), validation accuracy as tie-breaker.</li>
</ul></div>
</body></html>"""

html = (HTML.replace('__NAMES__', ' - '.join(names))
        .replace('__BEST__', best_name)
        .replace('__BESTCV__', pct(results[0]['cv_score']))
        .replace('__BESTVAL__', pct(results[0]['val_acc']))
        .replace('__WHY__', WHY_BEST.get(best_name, ''))
        .replace('__ROWS__', rows)
        .replace('__BARS__', bars)
        .replace('__CARDS__', cards)
        .replace('__QHEAD__', quick_head)
        .replace('__QROWS__', quick_rows))

report_path = HERE / 'model_comparison.html'
report_path.write_text(html, encoding='utf-8')
print(f"Report saved to {report_path}")
webbrowser.open(report_path.as_uri())
