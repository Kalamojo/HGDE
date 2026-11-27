"""Visualize baseline and transformed embeddings with separate PCA bases.

By default, creates one PNG containing both baseline and transformed views. Add
``--interactive`` to also write one HTML page containing both interactive views.
Each embedding set uses its own PCA basis.

Run from the repo root:
    python visualization/visualize_embeddings_using_pca.py
    python visualization/visualize_embeddings_using_pca.py --interactive
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "results" / "triplet_by_column" / "color_3c_all"
PALETTE = ["#3584c5", "#a96e5c", "#64bdcf", "#d75c62", "#9d78c1", "#d6a448"]
SEMANTIC_COLORS = {"blue": "#3182bd", "brown": "#8c564b", "green": "#2ca02c"}
LABEL_COLORS = SEMANTIC_COLORS


def build_view(name: str, coords: np.ndarray, labels: np.ndarray, clusters: np.ndarray, variance: np.ndarray) -> dict:
    return {
        "name": name,
        "variance": (variance * 100).round(2).tolist(),
        "bounds": [[float(coords[:, i].min()), float(coords[:, i].max())] for i in range(3)],
        "points": [
            {"xyz": np.round(coords[i], 6).tolist(), "label": str(labels[i]), "cluster": str(clusters[i])}
            for i in range(len(coords))
        ],
    }


def view_section(view: dict) -> str:
    data = json.dumps(view, separators=(",", ":")).replace("</", "<\\/")
    view_id = "pca-" + view["name"].lower().replace(" ", "-")
    return f'''<section class="pca-view" id="{view_id}">
  <h2>{html.escape(view["name"])} embeddings</h2>
  <div class="legend" aria-label="Plot legend"></div>
  <div class="stage">
    <svg class="plot" viewBox="0 0 720 430" role="img" aria-label="Interactive 3D PCA scatter plot for {html.escape(view["name"])} embeddings"></svg>
    <div class="tooltip" hidden></div>
  </div>
  <div class="hint">Drag to rotate · Scroll to zoom · Hover for Pokémon and cluster details</div>
  <script type="application/json" class="plot-data">{data}</script>
</section>'''


STYLE = '''
:root{color-scheme:dark;--page:#111318;--text:#e7e9ee;--muted:#aab1bd;--surface:#1b1f26;--border:#454c58;--grid:#68717e}
body{font:16px system-ui,sans-serif;margin:24px;color:var(--text);background:var(--page)}
h1{font-size:1.3em;font-weight:500;margin:0 0 .7em}h2{font-size:1.1em;font-weight:500;margin:.8em 0 .4em}
.color-control{display:flex;align-items:center;gap:.6em;margin:0 0 1em;color:var(--muted)}
.color-control select{font:inherit;color:var(--text);background:var(--surface);border:1px solid var(--border);border-radius:5px;padding:.35em .55em}
.pca-view{margin:0 0 1.5em}.legend{display:flex;flex-wrap:wrap;gap:.4em 1em;margin-bottom:.5em}
.legend-item{display:inline-flex;align-items:center;gap:.35em}.swatch{width:.72em;height:.72em;border-radius:50%;display:inline-block}
.stage{position:relative;width:100%;height:min(62vh,560px);min-height:360px}.plot{width:100%;height:100%;display:block;touch-action:none;overflow:hidden}
.tooltip{position:absolute;padding:.45em .65em;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:4px;box-shadow:0 2px 10px #0008;pointer-events:none;white-space:nowrap}
.hint{font-size:.85em;color:var(--muted);margin-top:.3em}
@media(max-width:480px){body{margin:12px}.stage{height:420px;min-height:340px}}
'''


SCRIPT = r'''<script>
(() => {
  const palette = __PALETTE__;
  const semantic = __SEMANTIC__;
  const ns = 'http://www.w3.org/2000/svg';
  const modeSelect = document.getElementById('color-mode');
  const views = [];
  const make = (tag, attrs, text) => {
    const el = document.createElementNS(ns, tag);
    for (const key of Object.keys(attrs)) el.setAttribute(key, attrs[key]);
    if (text) el.textContent = text;
    return el;
  };
  for (const root of document.querySelectorAll('.pca-view')) {
    const data = JSON.parse(root.querySelector('.plot-data').textContent);
    const svg = root.querySelector('.plot');
    const tooltip = root.querySelector('.tooltip');
    const labels = [...new Set(data.points.map(p => p.label))].sort();
    const center = data.bounds.map(([lo, hi]) => (lo + hi) / 2);
    const span = Math.max(...data.bounds.map(([lo, hi]) => hi - lo)) || 1;
    const axes = data.variance.map((v, i) => `PC${i + 1} (${v.toFixed(1)}%)`);
    const view = {yaw: -.72, pitch: .48, zoom: 1, projected: []};
    const colors = new Map();
    function project(v) {
      const x = (v[0] - center[0]) / span, y = (v[1] - center[1]) / span, z = (v[2] - center[2]) / span;
      const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
      const rx = x * cy - y * sy, ry = x * sy + y * cy;
      const px = rx, py = ry * cp - z * sp, depth = ry * sp + z * cp;
      const scale = Math.min(720, 430) * 1.2 * view.zoom;
      return {x: 360 + px * scale, y: 215 - py * scale, depth};
    }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const half = span * .55;
      const edges = [
        [[center[0]-half,center[1],center[2]],[center[0]+half,center[1],center[2]],axes[0]],
        [[center[0],center[1]-half,center[2]],[center[0],center[1]+half,center[2]],axes[1]],
        [[center[0],center[1],center[2]-half],[center[0],center[1],center[2]+half],axes[2]]
      ];
      for (const [a,b,label] of edges) {
        const p=project(a), q=project(b);
        svg.appendChild(make('line',{x1:p.x,y1:p.y,x2:q.x,y2:q.y,stroke:'#59616c','stroke-width':1,fill:'none'}));
        svg.appendChild(make('text',{x:q.x+5,y:q.y-5,fill:'#aab1bd','font-size':12},label));
      }
      view.projected = data.points.map((p,i)=>({...project(p.xyz),i})).sort((a,b)=>a.depth-b.depth);
      for (const p of view.projected) {
        const d=data.points[p.i];
        const colorKey = modeSelect.value === 'cluster' ? d.cluster : d.label;
        svg.appendChild(make('circle',{cx:p.x,cy:p.y,r:4.2,fill:colors.get(colorKey),opacity:.84,'data-i':p.i}));
      }
    }
    svg.addEventListener('pointerdown', e => {view.dragging=true;view.lastX=e.clientX;view.lastY=e.clientY;if(svg.setPointerCapture)svg.setPointerCapture(e.pointerId)});
    svg.addEventListener('pointerup',()=>{view.dragging=false});
    svg.addEventListener('pointercancel',()=>{view.dragging=false});
    svg.addEventListener('pointermove',e=>{
      const rect=svg.getBoundingClientRect();
      if(view.dragging){view.yaw+=(e.clientX-view.lastX)*.008;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-view.lastY)*.008));view.lastX=e.clientX;view.lastY=e.clientY;draw();tooltip.hidden=true;return}
      const x=(e.clientX-rect.left)*720/rect.width,y=(e.clientY-rect.top)*430/rect.height;
      let best=null,bestD=15*15;
      for(const p of view.projected){const d=(p.x-x)**2+(p.y-y)**2;if(d<bestD){best=p;bestD=d}}
      if(!best){tooltip.hidden=true;return}
      const p=data.points[best.i], rowIndex=best.i;
      tooltip.textContent=`${p.name || `row ${rowIndex}`} · ground truth: ${p.label} · predicted cluster: ${p.cluster}`;
      tooltip.style.left=`${Math.min(e.clientX-rect.left+12,rect.width-tooltip.offsetWidth-4)}px`;
      tooltip.style.top=`${Math.max(4,e.clientY-rect.top-34)}px`;tooltip.hidden=false;
    });
    svg.addEventListener('wheel',e=>{e.preventDefault();view.zoom=Math.max(.45,Math.min(3.5,view.zoom*(e.deltaY<0?1.08:.92)));draw()},{passive:false});
    views.push({root, data, colors, draw});
  }
  function updateColorMode() {
    const mode = modeSelect.value;
    const allValues = [...new Set(views.flatMap(v => v.data.points.map(p => mode === 'cluster' ? p.cluster : p.label)))];
    allValues.sort((a,b) => {
      if (mode === 'cluster' && Number.isFinite(Number(a)) && Number.isFinite(Number(b))) return Number(a)-Number(b);
      return String(a).localeCompare(String(b));
    });
    const colorMap = new Map(allValues.map((value,i) => [value,
      mode === 'color' && semantic[String(value).toLowerCase()] ? semantic[String(value).toLowerCase()] : palette[i % palette.length]
    ]));
    for (const view of views) {
      view.colors.clear();
      for (const [value,color] of colorMap) view.colors.set(value,color);
      const legend = view.root.querySelector('.legend');
      while (legend.firstChild) legend.removeChild(legend.firstChild);
      const localCounts = new Map();
      for (const point of view.data.points) {
        const value = mode === 'cluster' ? point.cluster : point.label;
        localCounts.set(value,(localCounts.get(value)||0)+1);
      }
      for (const value of allValues.filter(v => localCounts.has(v))) {
        const item=document.createElement('span'); item.className='legend-item';
        const swatch=document.createElement('span'); swatch.className='swatch'; swatch.style.backgroundColor=colorMap.get(value);
        item.appendChild(swatch); item.appendChild(document.createTextNode(`${mode === 'cluster' ? `Cluster ${value}` : value} (n=${localCounts.get(value)})`));
        legend.appendChild(item);
      }
      view.draw();
    }
  }
  modeSelect.addEventListener('change', updateColorMode);
  updateColorMode();
})();
</script>'''


def write_page(path: Path, title: str, views: list[dict]) -> None:
    content = "".join(view_section(v) for v in views)
    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        '<label class="color-control" for="color-mode">Color and legend by '
        '<select id="color-mode"><option value="color">Pokémon color</option>'
        '<option value="cluster">Cluster number</option></select></label>'
        f"{content}"
        f"{SCRIPT.replace('__PALETTE__', json.dumps(PALETTE)).replace('__SEMANTIC__', json.dumps(SEMANTIC_COLORS))}"
        "</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def write_static_pngs(
    output_dir: Path,
    baseline: np.ndarray,
    transformed: np.ndarray,
    labels: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = sorted(np.unique(labels))
    fig = plt.figure(figsize=(15, 7))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, name, embeddings in zip(axes, ("Baseline", "Transformed"), (baseline, transformed)):
        pca = PCA(n_components=3)
        coordinates = pca.fit_transform(embeddings)
        for i, label in enumerate(categories):
            mask = labels == label
            color = LABEL_COLORS.get(str(label).lower(), f"C{i % 10}")
            ax.scatter(
                coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2],
                color=color, alpha=0.78, s=24, depthshade=True,
                label=f"{label} (n={int(mask.sum())})",
            )
        ratios = pca.explained_variance_ratio_ * 100
        ax.set_title(f"{name} embeddings · 3D PCA")
        ax.set_xlabel(f"PC1 ({ratios[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ratios[1]:.1f}%)")
        ax.set_zlabel(f"PC3 ({ratios[2]:.1f}%)")
        ax.view_init(elev=22, azim=-55)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=max(1, len(categories)), frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    output_path = output_dir / "pca_baseline_transformed_3d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_DIR / "baseline_embeddings.npy")
    parser.add_argument("--transformed", type=Path, default=DEFAULT_DIR / "personalized_embeddings.npy")
    parser.add_argument("--assignments", type=Path, default=DEFAULT_DIR / "assignments.csv")
    parser.add_argument("--label-column", default="color")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--interactive", action="store_true",
        help="Also create one standalone interactive HTML page containing both views",
    )
    args = parser.parse_args()

    baseline = np.load(args.baseline, allow_pickle=False)
    transformed = np.load(args.transformed, allow_pickle=False)
    assignments = pd.read_csv(args.assignments)
    if baseline.ndim != 2 or transformed.shape != baseline.shape:
        parser.error(f"baseline and transformed must have matching N x D shapes, got {baseline.shape} and {transformed.shape}")
    if len(assignments) != len(baseline):
        parser.error("assignment CSV rows do not match embeddings")
    if "row_index" in assignments and not np.array_equal(assignments.row_index.to_numpy(), np.arange(len(baseline))):
        parser.error("assignment row_index values do not match embedding row order")
    if args.label_column not in assignments:
        parser.error(f"label column {args.label_column!r} not found in assignments")
    labels = assignments[args.label_column].fillna("unknown").astype(str).to_numpy()
    write_static_pngs(args.output_dir, baseline, transformed, labels)
    if not args.interactive:
        return

    needed = ("cluster_before", "cluster_after")
    missing = [column for column in needed if column not in assignments]
    if missing:
        parser.error(f"interactive output requires assignment columns: {missing}")

    names = assignments["name"].astype(str).to_numpy() if "name" in assignments else np.array([f"row {i}" for i in range(len(labels))])
    baseline_pca = PCA(n_components=3).fit(baseline)
    transformed_pca = PCA(n_components=3).fit(transformed)
    baseline_view = build_view("Baseline", baseline_pca.transform(baseline), labels,
                               assignments.cluster_before.astype(str).to_numpy(), baseline_pca.explained_variance_ratio_)
    transformed_view = build_view("Transformed", transformed_pca.transform(transformed), labels,
                                  assignments.cluster_after.astype(str).to_numpy(), transformed_pca.explained_variance_ratio_)
    for view, view_names in ((baseline_view, names), (transformed_view, names)):
        for point, name in zip(view["points"], view_names):
            point["name"] = str(name)

    output_path = args.output_dir / "pca_baseline_transformed_interactive.html"
    write_page(output_path, "Baseline and transformed embeddings · 3D PCA", [baseline_view, transformed_view])
    print(f"Wrote one self-contained interactive HTML page to {output_path.resolve()}")
    print(f"Baseline first-three-PC variance: {baseline_pca.explained_variance_ratio_.sum():.1%}")
    print(f"Transformed first-three-PC variance: {transformed_pca.explained_variance_ratio_.sum():.1%}")


if __name__ == "__main__":
    main()
