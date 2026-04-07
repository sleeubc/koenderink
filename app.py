"""
Koenderink Scale-Space Explorer
================================
Interactive visualization of Jan J. Koenderink's "Structure of Images" (1984)
scale-space theory, demonstrating how data structure is rigorously aggregated
and simplified across scales without creating spurious artifacts.
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
import json

# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Koenderink Scale-Space Explorer",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
    height: 100%;
    overflow: hidden;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0rem !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
    max-width: none !important;
}
.stApp {
    font-family: 'Inter', sans-serif;
    background: #f3f4f6;
    color: #111827;
}
[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
[data-testid="stHeader"] {
    background: transparent;
    height: 0 !important;
    min-height: 0 !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
}
[data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stStatusWidget"] {
    display: none !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0 !important;
}
[data-testid="element-container"]:has(iframe[title="streamlit-component"]) {
    min-height: 100vh;
    display: flex;
    align-items: center;
}
[data-testid="stExpander"] {
    margin-top: 0.4rem;
    background: rgba(255, 255, 255, 0.82) !important;
    border: 1px solid rgba(17, 24, 39, 0.08) !important;
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}
[data-testid="stExpander"] summary {
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
}
.stApp a {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Mathematical Engine
# ─────────────────────────────────────────────────────────────

def create_landscape(N=150, domain=(-4.5, 4.5)):
    """
    Create a synthetic 2D scalar field L(x, y) as a sum of Gaussians.
    Three peaks: one dominant, two subordinate on its slopes.
    This ensures saddle points exist between the peaks.
    """
    x = np.linspace(domain[0], domain[1], N)
    y = np.linspace(domain[0], domain[1], N)
    X, Y = np.meshgrid(x, y)

    # Peak 1: Primary central peak
    L = 5.0 * np.exp(-((X - 0.0)**2 + (Y - 0.0)**2) / (2 * 0.6**2))

    # Peak 2: Very close subordinate peak (northeast)
    # Positions (1.0484, 1.0484) with spread 0.5. 
    # This results in the combined peak being at exactly (1.0, 1.0) at fine scale.
    L += 4.5 * np.exp(-((X - 1.0484)**2 + (Y - 1.0484)**2) / (2 * 0.5**2))

    # Peak 3: Distant subordinate peak (southwest)
    L += 4.0 * np.exp(-((X + 2.5)**2 + (Y + 2.5)**2) / (2 * 0.6**2))

    # Add a tiny base elevation
    L += 0.1 

    return X, Y, L, x, y


def apply_scale_space(L, t, pixel_spacing):
    """
    Apply Gaussian blur to simulate the diffusion equation.
    sigma = sqrt(2t) in continuous domain.
    Convert to pixel-space sigma for scipy.
    """
    if t < 1e-8:
        return L.copy()
    sigma_continuous = np.sqrt(2 * t)
    sigma_pixels = sigma_continuous / pixel_spacing
    # Use 'constant' with the baseline 0.1 to avoid artifacts at edges.
    # This ensures that areas outside the domain are treated as the background level.
    return gaussian_filter(L, sigma=sigma_pixels, mode='constant', cval=0.1)


def compute_derivatives(K, dx):
    """Compute first and second spatial derivatives using central differences."""
    Kx = np.gradient(K, dx, axis=1)
    Ky = np.gradient(K, dx, axis=0)
    Kxx = np.gradient(Kx, dx, axis=1)
    Kyy = np.gradient(Ky, dx, axis=0)
    Kxy = np.gradient(Kx, dx, axis=0)
    return Kx, Ky, Kxx, Kyy, Kxy


def find_critical_points(K, X, Y, Kx, Ky, Kxx, Kyy, Kxy, dx, domain):
    """
    Find critical points where the gradient is approximately zero.
    Uses a refined approach: find local minima of gradient magnitude,
    then classify using the Hessian.
    """
    grad_mag = np.sqrt(Kx**2 + Ky**2)
    N = K.shape[0]

    # Hessian determinant at every point
    H_det = Kxx * Kyy - Kxy**2

    extrema = []
    saddles = []

    # 1. Better peak/pit detection: look for local max/min directly in K
    window = 2
    for i in range(window, N - window):
        for j in range(window, N - window):
            patch_K = K[i-window:i+window+1, j-window:j+window+1]
            # Is it a strict local extremum in this patch?
            if K[i, j] == patch_K.max() or K[i, j] == patch_K.min():
                # Verify it has positive Hessian (true extremum)
                if H_det[i, j] > 0 and grad_mag[i, j] < 0.6:
                    extrema.append((X[i, j], Y[i, j], K[i, j], H_det[i, j]))

    # 2. Saddle detection: local minima of gradient magnitude
    for i in range(window, N - window):
        for j in range(window, N - window):
            patch_grad = grad_mag[i-window:i+window+1, j-window:j+window+1]
            if grad_mag[i, j] == patch_grad.min() and grad_mag[i, j] < 0.4:
                if H_det[i, j] < 0:
                    saddles.append((X[i, j], Y[i, j], K[i, j], H_det[i, j]))

    # Remove duplicates (points too close together)
    extrema = _remove_duplicates(extrema, min_dist=0.4)
    saddles = _remove_duplicates(saddles, min_dist=0.4)

    # Filter to ensure we don't pick up boundary artifacts (staying inside the main field)
    # domain is (-4.5, 4.5)
    margin = 1.0 
    extrema = [p for p in extrema if abs(p[0]) < domain[1] - margin and abs(p[1]) < domain[1] - margin]
    saddles = [p for p in saddles if abs(p[0]) < domain[1] - margin and abs(p[1]) < domain[1] - margin]

    return extrema, saddles


def _remove_duplicates(points, min_dist=0.4):
    """Remove duplicate critical points that are too close together."""
    if not points:
        return points
    filtered = [points[0]]
    for p in points[1:]:
        too_close = False
        for f in filtered:
            dist = np.sqrt((p[0] - f[0])**2 + (p[1] - f[1])**2)
            if dist < min_dist:
                too_close = True
                # Keep the one with smaller gradient magnitude (= better critical point)
                break
        if not too_close:
            filtered.append(p)
    return filtered



def render_persistent_plotly(fig, storage_key, height):
    """Render a Plotly figure with browser-side UI state persistence."""
    div_id = f"{storage_key}_div"
    key_json = json.dumps(storage_key)
    post_script = f"""
    const plotDiv = document.getElementById('{div_id}');
    const storageKey = {key_json};

    function saveState() {{
        const state = {{
            relayout: plotDiv._persistentRelayout || {{}},
            visibility: plotDiv.data.map((trace) => trace.visible ?? true),
        }};
        localStorage.setItem(storageKey, JSON.stringify(state));
    }}

    function restoreState() {{
        const raw = localStorage.getItem(storageKey);
        if (!raw) return;

        try {{
            const state = JSON.parse(raw);
            if (state.visibility) {{
                state.visibility.forEach((visible, index) => {{
                    if (typeof visible !== 'undefined') {{
                        Plotly.restyle(plotDiv, {{ visible: [visible] }}, [index]);
                    }}
                }});
            }}
            if (state.relayout && Object.keys(state.relayout).length > 0) {{
                Plotly.relayout(plotDiv, state.relayout);
            }}
        }} catch (error) {{
            console.warn('Failed to restore Plotly state', error);
        }}
    }}

    restoreState();

    plotDiv.on('plotly_relayout', (eventData) => {{
        plotDiv._persistentRelayout = Object.assign(
            {{}},
            plotDiv._persistentRelayout || {{}},
            eventData
        );
        saveState();
    }});

    plotDiv.on('plotly_restyle', () => {{
        saveState();
    }});

    plotDiv.on('plotly_doubleclick', () => {{
        setTimeout(() => {{
            plotDiv._persistentRelayout = {{}};
            saveState();
        }}, 0);
    }});
    """

    html = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        default_width="100%",
        default_height=f"{height}px",
        div_id=div_id,
        post_script=post_script,
    )
    components.html(html, height=height, scrolling=False)


@st.cache_data
def get_landscape():
    N = 150
    domain = (-4.5, 4.5)
    X, Y, L, x, y = create_landscape(N=N, domain=domain)
    dx = (domain[1] - domain[0]) / (N - 1)
    return X, Y, L, x, y, dx, N, domain


# ─────────────────────────────────────────────────────────────
@st.cache_data
def get_scale_space_frames():
    X, Y, L, x_arr, y_arr, dx, N, domain = get_landscape()
    step = 2
    x_sub = np.round(x_arr[::step], 3)
    y_sub = np.round(y_arr[::step], 3)
    log_t_values = np.round(np.arange(-3.0, 1.5 + 1e-9, 0.05), 2)

    frames = []
    prev_extrema_count = None
    for log_t in log_t_values:
        t = float(10 ** float(log_t))
        K = apply_scale_space(L, t, dx)
        Kx, Ky, Kxx, Kyy, Kxy = compute_derivatives(K, dx)
        extrema, saddles = find_critical_points(K, X, Y, Kx, Ky, Kxx, Kyy, Kxy, dx, domain)
        
        # Compute Watershed Boundaries for "Cities"
        peak_idx = peak_local_max(K, min_distance=1)
        markers = np.zeros_like(K, dtype=int)
        for idx, (i, j) in enumerate(peak_idx, start=1):
            markers[i, j] = idx
        # Mask out virtually flat areas (base elevation is ~0.1)
        mountain_mask = K > 0.3 # Higher threshold to tighten boundaries
        labels = watershed(-K, markers, mask=mountain_mask)
        boundaries = find_boundaries(labels, mode='inner')
        # Clear domain grid edges to prevent artifact bounding box
        boundaries[0, :] = boundaries[-1, :] = boundaries[:, 0] = boundaries[:, -1] = False
        
        bx = X[boundaries].tolist()
        by = Y[boundaries].tolist()
        bz = K[boundaries].tolist()
        K_sub = np.round(K[::step, ::step], 4).tolist()

        event = None
        extrema_count = len(extrema)
        if prev_extrema_count is not None and extrema_count < prev_extrema_count:
            event = {"from": prev_extrema_count, "to": extrema_count}
        prev_extrema_count = extrema_count

        frames.append({
            "log_t": float(log_t),
            "t": round(t, 6),
            "sigma": round(float(np.sqrt(2 * t)), 4),
            "surface_z": K_sub,
            "contour_z": K_sub,
            "extrema": [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 4)] for p in extrema],
            "saddles": [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 4)] for p in saddles],
            "boundary_x": [round(float(v), 3) for v in bx],
            "boundary_y": [round(float(v), 3) for v in by],
            "boundary_z": [round(float(v), 4) for v in bz],
            "extrema_count": extrema_count,
            "saddle_count": len(saddles),
            "critical_count": extrema_count + len(saddles),
            "annihilation": event,
        })

    return {
        "domain": [float(domain[0]), float(domain[1])],
        "x": x_sub.tolist(),
        "y": y_sub.tolist(),
        "frames": frames,
    }


def render_client_scale_space(explorer_data):
    explorer_json = json.dumps(explorer_data, separators=(",", ":"))
    plotly_js = get_plotlyjs()
    html = f"""
    <div id="koenderink-client-app">
      <style>
        html, body {{
          margin: 0;
          height: 100%;
          overflow: hidden;
          background: #f3f4f6;
        }}
        #koenderink-client-app {{
          --panel-gap: 0.85rem;
          --panel-radius: 20px;
          --panel-title-size: 1.14rem;
          --plot-font-size: 18px;
          font-family: 'Inter', sans-serif;
          color: #111827;
          position: relative;
          width: 100%;
          height: 100%;
          overflow: hidden;
          background: #f3f4f6;
          font-size: 1.08rem;
          display: flex;
          flex-direction: column;
          gap: 0.45rem;
        }}
        #koenderink-client-app .explanation {{
          font-size: 1.04rem;
          line-height: 1.35;
          color: #374151;
          padding: 0 0.15rem;
        }}

        #koenderink-client-app .plots-grid {{
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: var(--panel-gap);
          flex: 1;
          min-height: 0;
          padding: 0;
          box-sizing: border-box;
        }}
        #koenderink-client-app .plot-panel {{
          position: relative;
          min-width: 0;
          height: 100%;
          border: 1px solid rgba(17, 24, 39, 0.08);
          border-radius: var(--panel-radius);
          background: rgba(255, 255, 255, 0.92);
          box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
          overflow: hidden;
        }}
        #koenderink-client-app .plot-host {{
          width: 100%;
          height: 100%;
        }}
        #koenderink-client-app .plot-title {{
          position: absolute;
          top: 0.85rem;
          left: 1rem;
          z-index: 10;
          font-size: var(--panel-title-size);
          font-weight: 600;
          color: rgba(17, 24, 39, 0.42);
          pointer-events: none;
          text-transform: uppercase;
          letter-spacing: 0.16em;
        }}

        #koenderink-client-app .control-bar {{
          width: 100%;
          background: rgba(255, 255, 255, 0.94);
          border: 1px solid rgba(17, 24, 39, 0.08);
          border-radius: 18px;
          padding: 0.55rem 0.8rem;
          display: flex;
          align-items: center;
          gap: 0.8rem;
          box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
          box-sizing: border-box;
        }}
        #koenderink-client-app .control-button,
        #koenderink-client-app .speed-select {{
          height: 2rem;
          border-radius: 999px;
          border: 1px solid rgba(17, 24, 39, 0.1);
          background: #ffffff;
          color: #111827;
          font: inherit;
        }}
        #koenderink-client-app .control-button {{
          min-width: 4.8rem;
          padding: 0 0.9rem;
          font-size: 0.94rem;
          font-weight: 600;
          cursor: pointer;
        }}
        #koenderink-client-app .control-button:hover,
        #koenderink-client-app .speed-select:hover {{
          border-color: rgba(17, 24, 39, 0.18);
        }}
        #koenderink-client-app .speed-select {{
          padding: 0 0.7rem;
          font-size: 0.92rem;
          cursor: pointer;
        }}
        #koenderink-client-app .sigma-label {{
          font-size: 1.22rem;
          font-weight: 600;
          color: #111827;
          line-height: 1;
          white-space: nowrap;
        }}

        #koenderink-client-app .slider-container {{
          display: flex;
          align-items: center;
          flex: 1;
          pointer-events: auto;
        }}
        #koenderink-client-app input[type="range"] {{
          width: 100%;
          accent-color: #111827;
          cursor: pointer;
          pointer-events: auto;
          position: relative;
          z-index: 60;
        }}

        #koenderink-client-app .metrics-inline {{
          display: flex;
          gap: 0.9rem;
          font-size: 0.76rem;
          font-weight: 500;
          color: #4b5563;
          white-space: nowrap;
        }}

        #koenderink-client-app .toast-container {{
          position: absolute;
          top: 3.75rem;
          right: 0.75rem;
          z-index: 100;
          pointer-events: none;
        }}
        #koenderink-client-app .annihilation-toast {{
          background: rgba(255, 255, 255, 0.96);
          border: 1px solid rgba(185, 28, 28, 0.14);
          border-left: 3px solid #b91c1c;
          border-radius: 12px;
          padding: 0.65rem 0.9rem;
          color: #991b1b;
          font-size: 0.76rem;
          font-weight: 600;
          box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
          animation: slide-in 0.3s ease-out forwards;
        }}

        @keyframes slide-in {{
          from {{ opacity: 0; transform: translateY(-10px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}

        @media (max-width: 980px) {{
          #koenderink-client-app .plots-grid {{
            grid-template-columns: 1fr 1fr;
            gap: 0.6rem;
          }}
          #koenderink-client-app .control-bar {{
            gap: 0.75rem;
            padding: 0.75rem 0.9rem;
          }}
          #koenderink-client-app .metrics-inline {{
            gap: 0.8rem;
            font-size: 0.8rem;
          }}
        }}

        @media (max-width: 820px) {{
          #koenderink-client-app {{
            --panel-gap: 0.55rem;
            --panel-radius: 16px;
            --panel-title-size: 1rem;
            --plot-font-size: 16px;
          }}
        }}

        @media (max-width: 720px) {{
          #koenderink-client-app .plot-title {{
            font-size: 0.88rem;
            letter-spacing: 0.12em;
          }}
          #koenderink-client-app .control-bar {{
            flex-direction: column;
            align-items: stretch;
          }}
          #koenderink-client-app .metrics-inline {{
            justify-content: space-between;
          }}
        }}

        @media (max-width: 700px) {{
          #koenderink-client-app .plots-grid {{
            grid-template-columns: 1fr;
            grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
          }}
        }}
      </style>

      <div class="plots-grid">
        <div class="plot-panel">
          <div class="plot-title">Smoothed Density Surface</div>
          <div id="plot3d" class="plot-host"></div>
        </div>
        <div class="plot-panel">
          <div class="plot-title">Topology</div>
          <div id="plot2d" class="plot-host"></div>
        </div>
      </div>

      <div class="toast-container" id="status-toast"></div>

      <div class="control-bar">
        <div class="sigma-label">σ</div>
        <div class="slider-container">
          <input id="scale-slider" type="range" min="0" max="0" step="1" value="0" title="Smoothing Parameter (σ)" />
        </div>
        <button id="play-toggle" class="control-button" type="button">Play</button>
        <select id="speed-select" class="speed-select" title="Animation speed">
          <option value="900">0.5x</option>
          <option value="600" selected>1x</option>
          <option value="300">2x</option>
          <option value="150">4x</option>
        </select>
      </div>

      <div class="explanation">
        Koenderink’s 1984 paper, “The Structure of Images,” has two main results. First, Gaussian smoothing across scale is the unique natural way to simplify an image without creating new peaks. For example, in delineating cities and subcenters, a city cannot spring up from an empty field as σ increases.
      </div>

      <div class="explanation">
        Second, as σ increases, peaks and saddle points merge and annihilate, leaving a smooth slope behind. This app demonstrates the second result.
      </div>

      <div class="explanation">
        These two results together imply a hierarchical structure of blobs (i.e., peak-centered regions) as we change σ.
      </div>

    </div>

    <script>{plotly_js}</script>
    <script>
      const explorer = {explorer_json};
      const slider = document.getElementById('scale-slider');
      const playToggle = document.getElementById('play-toggle');
      const speedSelect = document.getElementById('speed-select');
      const statusToast = document.getElementById('status-toast');
      const plot3d = document.getElementById('plot3d');
      const plot2d = document.getElementById('plot2d');
      const domain = explorer.domain;
      const xVals = explorer.x;
      const yVals = explorer.y;
      const root = document.getElementById('koenderink-client-app');

      const plotState = {{
        plot3d: {{
          camera: null,
          visibility: [true, true, true, true],
        }},
        plot2d: {{
          visibility: [true, true, true, true, true],
          relayout: {{}},
        }},
      }};
      let animationTimer = null;

      slider.max = String(explorer.frames.length - 1);

      function isCompactViewport() {{
        return window.innerWidth < 820;
      }}

      function legendConfig() {{
        if (window.innerWidth < 700) {{
          return {{
            visible: false,
          }};
        }}
        if (isCompactViewport()) {{
          return {{
            visible: true,
            orientation: 'h',
            x: 0.02,
            y: 0.98,
            xanchor: 'left',
            yanchor: 'top',
            itemwidth: 30,
          }};
        }}
        return {{
          visible: true,
          orientation: 'v',
          x: 0.02,
          y: 0.98,
          xanchor: 'left',
          yanchor: 'top',
        }};
      }}

      function build3dTraces(frame) {{
        const extrema = frame.extrema;
        const saddles = frame.saddles;
        return [
          {{
            type: 'surface',
            x: xVals,
            y: yVals,
            z: frame.surface_z,
            colorscale: 'Viridis',
            opacity: 0.95,
            showscale: false,
            lighting: {{ ambient: 0.5, diffuse: 0.7, specular: 0.2, roughness: 0.6 }},
            contours: {{ z: {{ show: true, usecolormap: true, project: {{ z: false }}, highlightcolor: '#94a3b8', width: 1 }} }},
            name: 'Scale Space Surface',
            hovertemplate: 'Peak Area<extra></extra>',
            visible: plotState.plot3d.visibility[0] ?? true,
          }},
          {{
            type: 'scatter3d',
            mode: 'markers',
            x: extrema.map((p) => p[0]),
            y: extrema.map((p) => p[1]),
            z: extrema.map((p) => p[2] + 0.15),
            customdata: extrema.map((p) => p[2]),
            marker: {{
              size: 6,
              color: '#4338ca',
              symbol: 'circle',
              line: {{ width: 1.5, color: '#e2e8f0' }},
            }},
            name: 'Peaks',
            hovertemplate: 'Peak<extra></extra>',
            visible: plotState.plot3d.visibility[1] ?? true,
          }},
          {{
            type: 'scatter3d',
            mode: 'markers',
            x: frame.boundary_x,
            y: frame.boundary_y,
            z: frame.boundary_z.map((z) => z + 0.05), // slightly elevate above surface
            marker: {{
              size: 2.5,
              color: '#fbbf24',
              opacity: 0.8,
            }},
            name: 'Watershed Boundaries',
            hovertemplate: 'Territory Boundary<extra></extra>',
            visible: plotState.plot3d.visibility[2] ?? true,
          }},
          {{
            type: 'scatter3d',
            mode: 'markers',
            x: saddles.map((p) => p[0]),
            y: saddles.map((p) => p[1]),
            z: saddles.map((p) => p[2] + 0.15),
            customdata: saddles.map((p) => p[2]),
            marker: {{
              size: 6,
              color: '#b45309',
              symbol: 'x',
              line: {{ width: 1.5, color: '#e2e8f0' }},
            }},
            name: 'Saddle Points',
            hovertemplate: 'Saddle Point<extra></extra>',
            visible: plotState.plot3d.visibility[3] ?? true,
          }},
        ];
      }}

      function build2dTraces(frame) {{
        const traces = [
          {{
            type: 'contour',
            x: xVals,
            y: yVals,
            z: frame.contour_z,
            colorscale: 'Viridis',
            contours: {{ coloring: 'heatmap', showlines: true, showlabels: false }},
            line: {{ width: 1, color: 'rgba(0, 0, 0, 0.05)' }},
            showscale: false,
            name: 'Surface',
            hovertemplate: 'Structure<extra></extra>',
            visible: plotState.plot2d.visibility[0] ?? true,
          }},
          {{
            type: 'contour',
            x: xVals,
            y: yVals,
            z: frame.contour_z,
            contours: {{
              coloring: 'none',
              showlines: true,
              showlabels: false,
            }},
            line: {{ width: 0.8, color: 'rgba(0, 0, 0, 0.1)' }},
            showscale: false,
            hoverinfo: 'skip',
            name: 'Isolines',
            visible: plotState.plot2d.visibility[1] ?? true,
          }},
        ];

        traces.push({{
          type: 'scatter',
          mode: 'markers',
          x: frame.boundary_x,
          y: frame.boundary_y,
          marker: {{
            size: 3,
            color: '#fbbf24',
          }},
          name: 'Watershed Boundaries',
          hovertemplate: 'Territory Boundary<extra></extra>',
          visible: plotState.plot2d.visibility[2] ?? true,
        }});

        traces.push(
          {{
            type: 'scatter',
            mode: 'markers',
            x: frame.extrema.map((p) => p[0]),
            y: frame.extrema.map((p) => p[1]),
            marker: {{
              size: 10,
              color: '#4338ca',
              symbol: 'circle',
              line: {{ width: 2, color: '#ffffff' }},
            }},
            name: 'Peaks',
            hovertemplate: 'Peak<extra></extra>',
            visible: plotState.plot2d.visibility[3] ?? true,
          }},
          {{
            type: 'scatter',
            mode: 'markers',
            x: frame.saddles.map((p) => p[0]),
            y: frame.saddles.map((p) => p[1]),
            marker: {{
              size: 10,
              color: '#b45309',
              symbol: 'x',
              line: {{ width: 2, color: '#ffffff' }},
            }},
            name: 'Saddle Points',
            hovertemplate: 'Saddle Point<extra></extra>',
            visible: plotState.plot2d.visibility[4] ?? true,
          }}
        );

        return traces;
      }}

      function build3dLayout() {{
        const compact = isCompactViewport();
        const legend = legendConfig();
        return {{
          uirevision: 'client-3d',
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: {{ family: "'Inter', 'Helvetica', 'Arial', sans-serif", color: '#4b5563', size: parseInt(getComputedStyle(root).getPropertyValue('--plot-font-size'), 10) || 13 }},
          margin: compact ? {{ l: 0, r: 0, t: 24, b: 0 }} : {{ l: 0, r: 0, t: 30, b: 0 }},
          autosize: true,
          legend: {{
            orientation: legend.orientation,
            bgcolor: 'rgba(255, 255, 255, 0.92)',
            bordercolor: 'rgba(17, 24, 39, 0.08)',
            borderwidth: 1,
            font: {{ family: "'Inter', 'Helvetica', 'Arial', sans-serif", color: '#111827', size: Math.max(14, (parseInt(getComputedStyle(root).getPropertyValue('--plot-font-size'), 10) || 18) - 1) }},
            x: legend.x,
            y: legend.y,
            xanchor: legend.xanchor,
            yanchor: legend.yanchor,
            itemwidth: legend.itemwidth,
            itemsizing: 'constant',
          }},
          showlegend: legend.visible,
          scene: {{
            uirevision: 'client-3d-scene',
            bgcolor: 'rgba(0,0,0,0)',
            camera: plotState.plot3d.camera || {{ eye: {{ x: 0.0, y: -1.35, z: 0.95 }} }},
            xaxis: {{
              title: '',
              range: domain,
              showticklabels: false,
              backgroundcolor: 'rgba(0,0,0,0)',
              gridcolor: '#e5e7eb',
              zeroline: false,
            }},
            yaxis: {{
              title: '',
              range: domain,
              showticklabels: false,
              backgroundcolor: 'rgba(0,0,0,0)',
              gridcolor: '#e5e7eb',
              zeroline: false,
            }},
            zaxis: {{
              title: '',
              range: [0, 8.5],
              showticklabels: false,
              backgroundcolor: 'rgba(0,0,0,0)',
              gridcolor: '#e5e7eb',
              zeroline: false,
            }},
          }},
        }};
      }}

      function build2dLayout() {{
        const compact = isCompactViewport();
        const legend = legendConfig();
        return Object.assign({{
          uirevision: 'client-2d',
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: {{ family: "'Inter', 'Helvetica', 'Arial', sans-serif", color: '#4b5563', size: parseInt(getComputedStyle(root).getPropertyValue('--plot-font-size'), 10) || 13 }},
          margin: compact ? {{ l: 8, r: 8, t: 24, b: 8 }} : {{ l: 14, r: 14, t: 30, b: 14 }},
          autosize: true,
          legend: {{
            orientation: legend.orientation,
            bgcolor: 'rgba(255, 255, 255, 0.92)',
            bordercolor: 'rgba(17, 24, 39, 0.08)',
            borderwidth: 1,
            font: {{ family: "'Inter', 'Helvetica', 'Arial', sans-serif", color: '#111827', size: Math.max(14, (parseInt(getComputedStyle(root).getPropertyValue('--plot-font-size'), 10) || 18) - 1) }},
            yanchor: legend.yanchor,
            y: legend.y,
            xanchor: legend.xanchor,
            x: legend.x,
            itemwidth: legend.itemwidth,
            itemsizing: 'constant',
          }},
          showlegend: legend.visible,
          xaxis: {{
            title: '',
            range: domain,
            scaleanchor: 'y',
            scaleratio: 1,
            showticklabels: false,
            gridcolor: '#e5e7eb',
            zeroline: false,
          }},
          yaxis: {{
            title: '',
            range: domain,
            showticklabels: false,
            gridcolor: '#e5e7eb',
            zeroline: false,
          }},
        }}, plotState.plot2d.relayout);
      }}

      function updateStatus(frame) {{
        const alertHtml = frame.annihilation
          ? `<div class="annihilation-toast">Topological annihilation occurred</div>`
          : ``;
        if (typeof statusToast !== 'undefined' && statusToast) {{
          statusToast.innerHTML = alertHtml;
	          if (frame.annihilation) {{
	            setTimeout(() => {{
	              if (statusToast.innerHTML.includes('Topological annihilation')) {{
	                statusToast.innerHTML = '';
	              }}
	            }}, 3000);
	          }}
        }}
      }}

      function updateReadout(frame) {{
        updateStatus(frame);
      }}

      function captureVisibility(gd, key) {{
        plotState[key].visibility = gd.data.map((trace) => trace.visible === undefined ? true : trace.visible);
      }}

      function merge2dRelayout(eventData) {{
        const next = {{ ...plotState.plot2d.relayout }};
        const allowedKeys = [
          'xaxis.range[0]', 'xaxis.range[1]', 'xaxis.autorange',
          'yaxis.range[0]', 'yaxis.range[1]', 'yaxis.autorange',
        ];
        allowedKeys.forEach((key) => {{
          if (Object.prototype.hasOwnProperty.call(eventData, key)) {{
            next[key] = eventData[key];
          }}
        }});
        if (eventData['xaxis.autorange']) {{
          delete next['xaxis.range[0]'];
          delete next['xaxis.range[1]'];
        }}
        if (eventData['yaxis.autorange']) {{
          delete next['yaxis.range[0]'];
          delete next['yaxis.range[1]'];
        }}
        plotState.plot2d.relayout = next;
      }}

      function renderFrame(index) {{
        const frame = explorer.frames[index];
        updateReadout(frame);
        Plotly.react(plot3d, build3dTraces(frame), build3dLayout(), {{ responsive: true, displaylogo: false }});
        Plotly.react(plot2d, build2dTraces(frame), build2dLayout(), {{ responsive: true, displaylogo: false }});
      }}

      function stopAnimation() {{
        if (animationTimer !== null) {{
          window.clearInterval(animationTimer);
          animationTimer = null;
        }}
        if (playToggle) playToggle.textContent = 'Play';
      }}

      function startAnimation() {{
        stopAnimation();
        if (playToggle) playToggle.textContent = 'Pause';
        const stepMs = Number(speedSelect?.value || 600);
        animationTimer = window.setInterval(() => {{
          const currentIndex = Number(slider.value);
          const nextIndex = currentIndex >= explorer.frames.length - 1 ? 0 : currentIndex + 1;
          slider.value = String(nextIndex);
          renderFrame(nextIndex);
        }}, stepMs);
      }}

      function toggleAnimation() {{
        if (animationTimer === null) {{
          startAnimation();
        }} else {{
          stopAnimation();
        }}
      }}

      function attachHandlers() {{
        plot3d.on('plotly_relayout', (eventData) => {{
          if (eventData['scene.camera']) {{
            plotState.plot3d.camera = eventData['scene.camera'];
          }}
        }});
        plot3d.on('plotly_restyle', () => captureVisibility(plot3d, 'plot3d'));

        plot2d.on('plotly_relayout', (eventData) => {{
          merge2dRelayout(eventData);
        }});
        plot2d.on('plotly_restyle', () => captureVisibility(plot2d, 'plot2d'));
      }}

      renderFrame(0);
      attachHandlers();

      slider.addEventListener('input', (event) => {{
        renderFrame(Number(event.target.value));
      }});
      playToggle.addEventListener('click', () => {{
        toggleAnimation();
      }});
      speedSelect.addEventListener('change', () => {{
        if (animationTimer !== null) {{
          startAnimation();
        }}
      }});
      window.addEventListener('resize', () => {{
        renderFrame(Number(slider.value));
      }});
    </script>
    """
    components.html(html, height=980, scrolling=False)



def main():
    # ─────────────────────────────────────────────────────────────
    # Session state initialization
    # ─────────────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []
    if "prev_extrema_count" not in st.session_state:
        st.session_state.prev_extrema_count = None
    if "annihilation_events" not in st.session_state:
        st.session_state.annihilation_events = []
    if "plot_defaults_initialized" not in st.session_state:
        st.session_state.plot_defaults_initialized = False

    # ─────────────────────────────────────────────────────────────
    # Render Application Component
    # ─────────────────────────────────────────────────────────────
    render_client_scale_space(get_scale_space_frames())


if __name__ == "__main__":
    main()
