import csv
import os
import re
import random
from io import BytesIO

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from pymcdm.methods import PROMETHEE_II, TOPSIS

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ELEMENT_COLUMNS = [f"Element_{i}" for i in range(1, 8)]
FILTER_OPTIONS = [
    "Reserve (ton)", "Production (ton)", "HHI (USGS)",
    "ESG Score", "CO2 footprint max (kg/kg)",
    "Embodied energy max (MJ/kg)", "Water usage max (l/kg)",
    "Toxicity", "Companionality",
]
CRITERIA_OPTIONS = {
    "Reserve (ton)": 1,  "Production (ton)": 1,  "HHI (USGS)": -1,
    "ESG Score": -1,     "CO2 footprint max (kg/kg)": -1,
    "Embodied energy max (MJ/kg)": -1, "Water usage max (l/kg)": -1,
    "Toxicity": -1,      "Companionality": -1,
}

def _sid(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", s).strip("-")

FILTER_SID = {f: _sid(f) for f in FILTER_OPTIONS}

CRIT_SID   = {f: _sid(f) for f in CRITERIA_OPTIONS}


# ─────────────────────────────────────────────────────────────────────────────
# PERIODIC TABLE — element data, styles, grid builder
# ─────────────────────────────────────────────────────────────────────────────
_PT_CSV = os.path.join(os.path.dirname(__file__), "elements.csv")

def _load_elements(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{"Z": int(r["Z"]), "sym": r["sym"], "name": r["name"],
              "mass": float(r["mass"]), "cat": r["cat"],
              "period": int(r["period"]), "group": int(r["group"])}
            for r in rows]

PT_ELEMENTS = _load_elements(_PT_CSV)
PT_BY_Z     = {e["Z"]: e for e in PT_ELEMENTS}
PT_SYM_SET  = {e["sym"] for e in PT_ELEMENTS}

PT_CATS = {
    "alkali metal":    {"bg": "#e74c3c", "text": "#fff"},
    "alkaline earth":  {"bg": "#e67e22", "text": "#fff"},
    "transition":      {"bg": "#2980b9", "text": "#fff"},
    "post-transition": {"bg": "#27ae60", "text": "#fff"},
    "metalloid":       {"bg": "#16a085", "text": "#fff"},
    "nonmetal":        {"bg": "#f39c12", "text": "#1a1a1a"},
    "halogen":         {"bg": "#8e44ad", "text": "#fff"},
    "noble gas":       {"bg": "#2c3e50", "text": "#ecf0f1"},
    "lanthanide":      {"bg": "#c0392b", "text": "#fff"},
    "actinide":        {"bg": "#d35400", "text": "#fff"},
    "unknown":         {"bg": "#7f8c8d", "text": "#fff"},
}
PT_CAT_LABELS = {
    "alkali metal": "Alkali Metal", "alkaline earth": "Alkaline Earth Metal",
    "transition": "Transition Metal", "post-transition": "Post-Transition Metal",
    "metalloid": "Metalloid", "nonmetal": "Nonmetal", "halogen": "Halogen",
    "noble gas": "Noble Gas", "lanthanide": "Lanthanide",
    "actinide": "Actinide", "unknown": "Unknown",
}
_CW, _CH = 46, 46

# OPT: pre-compute the base (no-selection) style for every element once at
#      startup — cell style callbacks shallow-copy this and only mutate the
#      keys that need to change (border, boxShadow, opacity, transform, zIndex).
_PT_BASE_STYLES = {}
for _e in PT_ELEMENTS:
    _PT_BASE_STYLES[_e["Z"]] = {
        "gridRow": _e["period"], "gridColumn": _e["group"],
        "width": f"{_CW}px", "height": f"{_CH}px",
        "padding": "2px 2px", "borderRadius": "4px",
        "cursor": "pointer", "userSelect": "none",
        "display": "flex", "flexDirection": "column",
        "justifyContent": "space-between",
        "overflow": "hidden", "boxSizing": "border-box",
        "transition": "transform 0.12s, box-shadow 0.12s, opacity 0.12s",
        "backgroundColor": PT_CATS[_e["cat"]]["bg"],
        "color":           PT_CATS[_e["cat"]]["text"],
        "border": "2px solid transparent",
        "opacity": "1", "transform": "scale(1)",
        "zIndex": "1", "boxShadow": "none",
    }


def _pt_cell_style(elem, active_set: set, mode: str) -> dict:
    """
    OPT: accepts a pre-built set for O(1) membership test instead of list.
    Shallow-copies the pre-built base style and only mutates changed keys.
    """
    s = dict(_PT_BASE_STYLES[elem["Z"]])   # shallow copy — fast
    is_active = elem["sym"] in active_set
    if is_active:
        if mode == "exclude":
            s["border"]    = "2px solid #ff4757"
            s["boxShadow"] = "0 0 0 2px rgba(255,71,87,0.8)"
        else:
            s["border"]    = "2px solid #f9ca24"
            s["boxShadow"] = "0 0 0 2px rgba(249,202,36,0.8)"
        s["transform"] = "scale(1.08)"
        s["zIndex"]    = "15"
    elif active_set and not is_active:
        s["opacity"] = "0.35"
    return s


def _pt_make_cell(elem, prefix: str) -> html.Div:
    return html.Div(
        id={"type": f"{prefix}-elem", "Z": elem["Z"]},
        n_clicks=0,
        style=dict(_PT_BASE_STYLES[elem["Z"]]),  # use pre-built base
        children=[
            html.Div(str(elem["Z"]),
                     style={"fontSize": "7px", "opacity": "0.8", "lineHeight": "1"}),
            html.Div(elem["sym"],
                     style={"fontSize": "15px", "fontWeight": "700",
                             "textAlign": "center", "lineHeight": "1.1"}),
            html.Div(elem["name"],
                     style={"fontSize": "5.5px", "textAlign": "center",
                             "overflow": "hidden", "whiteSpace": "nowrap",
                             "textOverflow": "ellipsis", "lineHeight": "1"}),
        ],
    )


def _pt_make_legend() -> html.Div:
    chips = [
        html.Div([
            html.Div(style={"width": "11px", "height": "11px", "borderRadius": "2px",
                             "backgroundColor": PT_CATS[k]["bg"], "flexShrink": "0"}),
            html.Span(PT_CAT_LABELS[k], style={"fontSize": "10px", "whiteSpace": "nowrap"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px"})
        for k in PT_CATS
    ]
    return html.Div(chips, style={"display": "flex", "flexWrap": "wrap",
                                   "gap": "5px 14px", "marginTop": "8px"})


def build_periodic_table(prefix: str, mode: str = "include") -> html.Div:
    verb  = "include" if mode == "include" else "exclude"
    color = "#f9ca24" if mode == "include" else "#ff4757"
    hint  = html.Div([
        html.Span("● ", style={"color": color, "fontWeight": "700"}),
        html.Span(f"Click elements to {verb}. Click again to deselect. "),
        html.Span("Selected elements dimly grey out the rest.", style={"color": "#6c757d"}),
    ], style={"fontSize": "12px", "marginBottom": "8px"})

    cells = [_pt_make_cell(e, prefix) for e in PT_ELEMENTS]

    for row, label, cat in [(6, "*", "lanthanide"), (7, "**", "actinide")]:
        cells.append(html.Div(label, style={
            "gridRow": row, "gridColumn": 3,
            "width": f"{_CW}px", "height": f"{_CH}px",
            "backgroundColor": PT_CATS[cat]["bg"], "color": PT_CATS[cat]["text"],
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "borderRadius": "4px", "fontSize": "14px", "fontWeight": "700",
            "cursor": "default", "userSelect": "none",
        }))

    for p in range(1, 8):
        cells.append(html.Div(str(p), style={
            "gridRow": p, "gridColumn": "1", "marginLeft": "-20px",
            "color": "#adb5bd", "fontSize": "10px", "fontWeight": "600",
            "display": "flex", "alignItems": "center", "justifyContent": "flex-end",
            "width": "16px", "height": f"{_CH}px", "paddingRight": "3px",
        }))
    for frow, lbl in [(9, "6"), (10, "7")]:
        cells.append(html.Div(lbl, style={
            "gridRow": frow, "gridColumn": "3", "marginLeft": "-20px",
            "color": "#adb5bd", "fontSize": "9px",
            "display": "flex", "alignItems": "center", "justifyContent": "flex-end",
            "width": "16px", "height": f"{_CH}px", "paddingRight": "3px",
        }))
    for g in range(1, 19):
        cells.append(html.Div(str(g), style={
            "gridRow": "1", "gridColumn": g,
            "color": "#adb5bd", "fontSize": "9px", "fontWeight": "600",
            "textAlign": "center", "marginTop": "-16px", "height": "16px",
            "display": "flex", "alignItems": "flex-end", "justifyContent": "center",
        }))

    grid = html.Div(cells, style={
        "display":             "grid",
        "gridTemplateColumns": f"repeat(18, {_CW}px)",
        "gridTemplateRows":    f"repeat(7, {_CH}px) 8px repeat(2, {_CH}px)",
        "gap":                 "2px",
        "width":               "fit-content",
        "marginTop":           "16px",
    })
    return html.Div([
        hint,
        html.Div(grid, style={"overflowX": "auto", "paddingBottom": "6px"}),
        _pt_make_legend(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DATA — loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────
DF1 = pd.read_excel("7_materials_properties.xlsx")
DF2 = pd.read_excel("materials_high_confidence_cleaned.xlsx").iloc[:, 1:]
DF2["Date"] = pd.to_datetime(DF2["Date"], errors="coerce")

# OPT: pre-clean Element columns in DF1 once so filter functions skip
#      per-call .fillna("").astype(str).str.strip() on every row.
for _c in ELEMENT_COLUMNS:
    if _c in DF1.columns:
        DF1[_c] = DF1[_c].fillna("").astype(str).str.strip()

# OPT: pre-convert DF2 numeric columns once — avoids repeated pd.to_numeric
#      coercion inside bg_update_all on every element click.
if "Value" in DF2.columns:
    DF2["Value"] = pd.to_numeric(DF2["Value"], errors="coerce")

ELEM_COLS_DF2 = DF2.columns[3:-2].tolist()
DF2_DATE_MIN  = DF2["Date"].min().date()
DF2_DATE_MAX  = DF2["Date"].max().date()


# ─────────────────────────────────────────────────────────────────────────────
# PURE DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def format_tons(v: float) -> str:
    if v >= 1e12: return f"{v/1e12:.1f}TB tons"
    if v >= 1e9:  return f"{v/1e9:.1f}B tons"
    if v >= 1e6:  return f"{v/1e6:.1f}M tons"
    if v >= 1e3:  return f"{v/1e3:.1f}K tons"
    return f"{v:.0f} tons"


def parse_elements(text: str) -> list:
    if not text or not text.strip():
        return []
    return [e.strip() for e in text.split(",") if e.strip()]


def filter_by_excluded(df: pd.DataFrame, excluded: list) -> pd.DataFrame:
    if not excluded:
        return df
    # OPT: DF1 element columns are pre-cleaned at startup — plain isin, no str ops
    excl  = set(excluded)
    valid = [c for c in ELEMENT_COLUMNS if c in df.columns]
    mask  = pd.concat([df[c].isin(excl) for c in valid], axis=1).any(axis=1)
    return df[~mask]


def filter_by_included_df1(df: pd.DataFrame, included: list) -> pd.DataFrame:
    if not included:
        return df
    # OPT: columns pre-cleaned — empty string check is enough, no astype/strip
    inc  = set(included)
    mask = pd.Series(True, index=df.index)
    for c in ELEMENT_COLUMNS:
        if c in df.columns:
            mask &= df[c].isin(inc) | (df[c] == "")
    return df[mask]


def filter_by_included_df2(df: pd.DataFrame, included: list) -> pd.DataFrame:
    if not included:
        return df.iloc[0:0]
    inc  = set(included)
    mask = pd.Series(True, index=df.index)
    for c in ELEM_COLS_DF2:
        if c in df.columns and c not in inc:
            mask &= df[c] == 0
    return df[mask]


def filter_df(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in filters.items():
        if col in df.columns:
            mask &= df[col].between(lo, hi, inclusive="both")
    return df[mask]


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    n, m = matrix.shape
    prob = np.zeros_like(matrix, dtype=float)
    for j in range(m):
        s = matrix[:, j].sum()
        prob[:, j] = matrix[:, j] / s if s > 1e-10 else 1.0 / n
    divs = np.array([
        1.0 - (-np.sum(np.where(p > 1e-10, p, 1e-10) *
                       np.log(np.where(p > 1e-10, p, 1e-10))) / np.log(n))
        for p in prob.T
    ])
    s = divs.sum()
    return divs / s if s > 1e-10 else np.ones(m) / m


def pick_palette(n: int) -> list:
    c = px.colors.qualitative.Set3 if n <= 12 else px.colors.qualitative.Light24
    return c[:n]


def professional_scatter(df, x_col, y_col, title, x_label, y_label,
                          log_x=False, log_y=False) -> go.Figure:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return go.Figure()
    primary, accent = "#3498db", "#c5301f"
    # OPT: only copy the DataFrame when log-clipping is actually needed
    dp = df.copy() if (log_x or log_y) else df
    if log_x: dp[x_col] = dp[x_col].clip(lower=1e-10)
    if log_y: dp[y_col] = dp[y_col].clip(lower=1e-10)
    hi = np.random.choice(len(dp), min(10, len(dp)), replace=False)
    hi_mask = np.zeros(len(dp), dtype=bool)
    hi_mask[hi] = True
    reg = dp[~hi_mask]
    hil = dp[hi_mask]
    fig = go.Figure([
        go.Scatter(
            x=reg[x_col], y=reg[y_col], mode="markers", name="All Materials",
            marker=dict(size=8, color=primary, opacity=0.6), text=reg["Name"],
            hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>",
        ),
        go.Scatter(
            x=hil[x_col], y=hil[y_col], mode="markers+text", name="Highlighted",
            marker=dict(size=12, color=accent), text=hil["Name"],
            textposition="top center", textfont=dict(color=accent, size=10),
            hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>",
        ),
    ])
    fig.update_layout(
        title=title,
        xaxis_title=f"log({x_label})" if log_x else x_label,
        yaxis_title=f"log({y_label})" if log_y else y_label,
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
        hovermode="closest", template="plotly_white", height=500,
        legend=dict(x=0.99, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def build_excel(filtered_df, results_df, weights_df, filters: dict) -> bytes:
    out = BytesIO()
    rr  = results_df.reset_index()
    fd  = filtered_df.copy()
    if "Score" in rr.columns:
        fd["TOPSIS_Score"] = fd["Name"].map(dict(zip(rr["Material"], rr["Score"])))
        fd["TOPSIS_Rank"]  = fd["Name"].map(dict(zip(rr["Material"], rr["Rank"])))
    else:
        fd["PROMETHEE_Net_Flow"] = fd["Name"].map(dict(zip(rr["Material"], rr["Net Flow"])))
        fd["PROMETHEE_Rank"]     = fd["Name"].map(dict(zip(rr["Material"], rr["Rank"])))
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        fd.to_excel(w, sheet_name="Full Data", index=False)
        rr.to_excel(w, sheet_name="Rankings",  index=False)
        weights_df.reset_index().to_excel(w, sheet_name="Weights", index=False)
        if filters:
            pd.DataFrame(
                [{"Filter": k, "Min": v[0], "Max": v[1]} for k, v in filters.items()]
            ).to_excel(w, sheet_name="Filter Settings", index=False)
    return out.getvalue()


# Filters shown as a single-handle (minimum only) — max is always data max
SINGLE_MIN_FILTERS = {"Reserve (ton)", "Production (ton)"}
# Filters whose values are always whole numbers
INTEGER_FILTERS    = {"Companionality", "Toxicity"}

# ─────────────────────────────────────────────────────────────────────────────
# PRE-RENDERED FILTER BANKS
#
# SINGLE_MIN_FILTERS → dcc.Slider (one handle = minimum threshold)
# INTEGER_FILTERS    → dcc.RangeSlider, integer step
# HHI                → dcc.RangeSlider, fixed 0–1, step 0.001
# others             → dcc.RangeSlider, step 0.01  (prevents tooltip rounding)
# ─────────────────────────────────────────────────────────────────────────────
def _make_filter_slider_bank(prefix: str) -> list:
    out = []
    for fname in FILTER_OPTIONS:
        sid  = FILTER_SID[fname]
        fmin = float(DF1[fname].min())
        fmax = float(DF1[fname].max())

        if fname in SINGLE_MIN_FILTERS:
            # Plain number input for minimum threshold — no slider snapping.
            # apply_filters pairs this with the data maximum: [user_min, data_max]
            inner = [
                dbc.Label(f"Minimum {fname}"),
                dbc.InputGroup([
                    dbc.InputGroupText("≥"),
                    dbc.Input(
                        id=f"{prefix}-{sid}",
                        type="number",
                        min=0, value=int(fmin),
                        placeholder=f"e.g. {int(fmin)}",
                        debounce=True,
                    ),
                    dbc.InputGroupText("tons"),
                ], className="mb-1"),
                html.Small(f"Data range: {format_tons(fmin)} – {format_tons(fmax)}",
                           className="text-muted"),
            ]

        elif fname == "HHI (USGS)":
            inner = [
                dbc.Label(f"{fname} range (0 – 1)"),
                dcc.RangeSlider(
                    id=f"{prefix}-{sid}",
                    min=0.0, max=1.0, value=[0.0, 1.0],
                    step=0.001, marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]

        elif fname in INTEGER_FILTERS:
            step = max(1, int(round((fmax - fmin) / 1000)))
            inner = [
                dbc.Label(f"{fname} range"),
                dcc.RangeSlider(
                    id=f"{prefix}-{sid}",
                    min=int(fmin), max=int(fmax), value=[int(fmin), int(fmax)],
                    step=step, marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]

        else:
            # step=0.01 prevents the tooltip from rounding typed values:
            # with step=0.05, typing "1.23" snaps to "1.25"; step=0.01 is fine.
            inner = [
                dbc.Label(f"{fname} range"),
                dcc.RangeSlider(
                    id=f"{prefix}-{sid}",
                    min=round(fmin, 2), max=round(fmax, 2),
                    value=[round(fmin, 2), round(fmax, 2)],
                    step=0.01, marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]

        inner.append(html.Div(id=f"{prefix}-{sid}-note",
                               className="text-muted small mb-2"))
        out.append(html.Div(id=f"{prefix}-{sid}-wrap",
                             style={"display": "none"}, children=inner))
    return out


def _make_weight_sliders() -> list:
    """One 0–5 Slider per MCDM criterion.
    Wrapped in 'weight-slider-col' div so the CSS in assets/md_search.css
    can hide the editable input box that Dash 4.x adds beside every slider.
    """
    return [
        dbc.Col(
            html.Div([
                dbc.Label(cname, className="small fw-semibold"),
                dcc.Slider(
                    id=f"weight-{CRIT_SID[cname]}",
                    min=0, max=5, step=1, value=3,
                    marks={i: str(i) for i in range(6)},
                ),
            ], className="weight-slider-col"),
            width=12, lg=4, className="mb-3",
        )
        for cname in CRITERIA_OPTIONS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUTS
# ─────────────────────────────────────────────────────────────────────────────
def layout_home() -> html.Div:
    pmin = DF1["Production (ton)"].min()
    pmax = DF1["Production (ton)"].max()
    return html.Div([
        html.H1("Semiconductor Database", className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.H4("🔍 About This Tool"),
                html.Ul([
                    html.Li([html.Strong("Extensive database"), " on ESG scores, CO₂ footprints, and more"]),
                    html.Li([html.Strong("Visualizations"), " to explore relationships between parameters"]),
                    html.Li([html.Strong("Multi-criteria"), " decision making tools (TOPSIS, PROMETHEE)"]),
                    html.Li([html.Strong("Export capabilities"), " for further analysis"]),
                ]),
            ]),
            dbc.Col([
                html.H4("🚀 Getting Started"),
                html.Ol([
                    html.Li("Select an analysis page from the sidebar"),
                    html.Li("Configure your filters and parameters"),
                    html.Li("Visualize the relationships"),
                    html.Li("Download results for further use"),
                ]),
                dbc.Alert("💡 Pro Tip: Use the MCDM analysis for ranking the most promising semiconductors.",
                          color="info", className="mt-3"),
            ]),
        ]),
        html.Hr(),
        html.H4("📚 Database Information", className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Total Materials", className="text-muted mb-1 small"),
                html.H3(str(len(DF1)), className="text-primary mb-0"),
            ])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Production Range", className="text-muted mb-1 small"),
                html.H5(f"{format_tons(pmin)} – {format_tons(pmax)}", className="mb-0"),
            ])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Bandgap Range", className="text-muted mb-1 small"),
                html.H5(f"{DF1['Bandgap'].min():.1f} – {DF1['Bandgap'].max():.1f} eV", className="mb-0"),
            ])), width=4),
        ]),
    ])


def layout_bandgap() -> html.Div:
    return html.Div([
        html.H1("Bandgap Information"),
        html.P("Most commonly researched semiconductors and their band gap range."),
        html.Hr(),
        html.H4("Element Inclusion"),
        build_periodic_table("bg", mode="include"),
        html.Div(id="bg-filter-info", className="mb-3 mt-2"),
        html.Hr(),
        html.H5("Exploratory Data Analysis of Database"),
        html.Em("The scatterplot provides a broad visual overview of the relative bandgap ranges across the materials."),
        dcc.Graph(id="bg-scatter", className="mt-2"),
        html.Em("Histogram plot shows the frequency distribution of bandgaps."),
        dcc.Graph(id="bg-histogram", className="mt-2"),
        html.Hr(),
        html.Em("The scatter plot visualizes the trends of recently researched materials and their corresponding bandgap values."),
        dcc.Graph(id="bg-temporal-scatter"),
        html.Hr(),
        html.Em("The table displays ten(10) sampled journals relating to the filtered semiconductors."),
        dbc.Button("🔀 Shuffle sample", id="bg-shuffle-btn", color="secondary",
                   size="sm", className="mt-2 mb-2"),
        html.Div(id="bg-sample-table"),
        html.Div(id="bg-download-area"),
    ])


def layout_decision() -> html.Div:
    ifilter_bank   = _make_filter_slider_bank("if")
    efilter_bank   = _make_filter_slider_bank("ef")
    weight_sliders = _make_weight_sliders()

    return html.Div([
        html.H1("Decision-making Assistant"),
        html.P("Facilitate semiconductor selection with advanced filtering and visualization"),

        html.H4("1. Element Exclusion"),
        build_periodic_table("dec", mode="exclude"),
        html.Div(id="dec-excl-info", className="mb-3 mt-2"),

        html.H4("2. Initial Filters"),
        dbc.Row([
            dbc.Col([
                html.H6("Bandgap Selection"),
                dbc.Row([
                    dbc.Col([dbc.Label("Min (eV)"),
                             dbc.Input(id="dec-bg-min", type="number",
                                       min=0, max=35, value=0.0)]),
                    dbc.Col([dbc.Label("Max (eV)"),
                             dbc.Input(id="dec-bg-max", type="number",
                                       min=0, max=35, value=3.0)]),
                ]),
                html.Div(id="dec-bg-error"),
            ], width=6),
            dbc.Col([
                html.H6("Additional Filter"),
                dcc.Dropdown(id="dec-filter-select",
                             options=[{"label": f, "value": f} for f in FILTER_OPTIONS],
                             value=FILTER_OPTIONS[0], clearable=False, className="mb-2"),
                html.Div(ifilter_bank),
            ], width=6),
        ]),

        html.H4("3. Additional Filters (Optional)", className="mt-4"),
        html.P("Pick any number of extra filters:"),
        dcc.Dropdown(id="dec-extra-select",
                     options=[{"label": f, "value": f} for f in FILTER_OPTIONS],
                     multi=True, placeholder="Select additional filters…", className="mb-3"),
        html.Div(efilter_bank),

        dbc.Button("✅ Apply Filters", id="dec-apply-btn",
                   color="primary", size="lg", className="mt-3 mb-2"),
        html.Div(id="dec-apply-status", className="mb-3"),

        html.Hr(),
        html.H4("Filtered Results"),
        html.Div(id="dec-filter-info", className="mb-2"),
        dbc.Row([
            dbc.Col(html.Div(id="dec-axis-info"), width=8),
            dbc.Col(dbc.Checklist(id="dec-log-y",
                                  options=[{"label": " Log Y-axis", "value": "log"}],
                                  value=[]), width=4),
        ]),
        dcc.Graph(id="dec-scatter"),

        html.Hr(),
        html.Div(id="dec-mcdm-wrapper", style={"display": "none"}, children=[
            html.H4("4. Multi-Criteria Decision Making"),
            html.Div(id="dec-mcdm-info", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Method"),
                    dcc.Dropdown(id="mcdm-method",
                                 options=[{"label": "TOPSIS",    "value": "TOPSIS"},
                                          {"label": "PROMETHEE", "value": "PROMETHEE"}],
                                 value="TOPSIS", clearable=False),
                ], width=6),
                dbc.Col([
                    dbc.Label("Weighting Method"),
                    dbc.RadioItems(id="mcdm-weighting",
                                   options=[{"label": "Entropy (automatic)", "value": "Entropy"},
                                            {"label": "Manual Weights",      "value": "Manual"}],
                                   value="Entropy", inline=True),
                ], width=6),
            ], className="mb-3"),

            html.Div(id="mcdm-manual-section", style={"display": "none"}, children=[
                html.H6("📊 Criteria Weights (drag sliders — 0 = ignore, 5 = most important)"),
                dbc.Row([dbc.Col([
                    dbc.Button("Balanced",        id="preset-balanced",  color="outline-secondary", size="sm", className="me-2"),
                    dbc.Button("Long-term goal",  id="preset-longterm",  color="outline-secondary", size="sm", className="me-2"),
                    dbc.Button("Short-term goal", id="preset-shortterm", color="outline-secondary", size="sm"),
                ], className="mb-3")]),
                dbc.Row(weight_sliders),
            ]),

            html.Div(id="mcdm-weights-table", className="mb-3"),
            dbc.Button("🚀 Run MCDM Analysis", id="mcdm-run-btn",
                       color="success", size="lg", className="mb-4"),
            html.Div(id="mcdm-status", className="mb-2"),
            html.Div(id="mcdm-results-area"),
            html.Div(id="mcdm-dl-wrapper", style={"display": "none"}, children=[
                dbc.Button("📥 Download Full MCDM Report", id="mcdm-dl-btn",
                           color="primary", className="mt-3"),
            ]),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Semiconductor Database",
)

server = app.server  # for gunicorn: `gunicorn app:server --bind 0.0.0.0:$PORT`

_SIDEBAR = {
    "backgroundColor": "#f8f9fa",
    "borderRight":     "1px solid #e0e0e0",
    "minHeight":       "100vh",
    "padding":         "1rem",
    "position":        "sticky",
    "top":             0,
    "overflowY":       "auto",
}

app.layout = dbc.Container(fluid=True, children=[
    dcc.Location(id="url"),

    dcc.Store(id="store-excl",           data=[]),
    dcc.Store(id="bg-elem-store",        data=[]),
    dcc.Store(id="store-dec-filters",    data={}),
    dcc.Store(id="store-mcdm-results"),
    dcc.Store(id="store-sample-seed",    data=42),
    dcc.Store(id="store-bg-csv",         data=None),
    dcc.Store(id="store-preset-weights", data={c: 3 for c in CRITERIA_OPTIONS}),

    dcc.Download(id="dl-mcdm"),
    dcc.Download(id="dl-csv"),

    dbc.Row([
        dbc.Col(style=_SIDEBAR, width=2, children=[
            html.H5("Material Analysis", className="fw-bold mb-2"),
            html.Hr(className="my-2"),
            dbc.Nav(vertical=True, pills=True, className="mb-3", children=[
                dbc.NavLink("🏠 Home",                      href="/",         active="exact"),
                dbc.NavLink("📊 Bandgap Information",       href="/bandgap",  active="exact"),
                dbc.NavLink("🧠 Decision-making Assistant", href="/decision", active="exact"),
            ]),
            html.Hr(className="my-2"),
            html.Small("Welcome page",                        className="text-muted d-block"),
            html.Small("Commonly researched semiconductors",  className="text-muted d-block"),
            html.Small("Multi-criteria decision making tool", className="text-muted d-block"),
            html.Hr(className="my-2"),
            html.Small("Semiconductor Database © 2025 | v3.0 | Developed by HERAWS",
                       className="text-muted"),
        ]),
        # All three pages live permanently in the DOM.
        # Routing toggles display:block / display:none — component IDs always exist.
        dbc.Col(width=10, children=[
            html.Div(id="page-home",     className="p-4", children=layout_home()),
            html.Div(id="page-bandgap",  className="p-4", style={"display":"none"}, children=layout_bandgap()),
            html.Div(id="page-decision", className="p-4", style={"display":"none"}, children=layout_decision()),
        ]),
    ]),
])


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING — toggles CSS display; all page IDs always exist in the DOM
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("page-home",     "style"),
    Output("page-bandgap",  "style"),
    Output("page-decision", "style"),
    Input("url", "pathname"),
)
def render_page(pathname):
    show = {"display": "block"}
    hide = {"display": "none"}
    if pathname == "/bandgap":
        return hide, {**show, "padding": "1.5rem"}, hide
    if pathname == "/decision":
        return hide, hide, {**show, "padding": "1.5rem"}
    return {**show, "padding": "1.5rem"}, hide, hide


# ═════════════════════════════════════════════════════════════════════════════
# BANDGAP PAGE CALLBACKS
# OPT: bg_update_charts and bg_update_temporal merged into ONE callback so
#      filter_by_included_df1(DF1, included) is called only once per interaction.
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("bg-filter-info",      "children"),
    Output("bg-scatter",          "figure"),
    Output("bg-histogram",        "figure"),
    Output("bg-temporal-scatter", "figure"),
    Output("bg-sample-table",     "children"),
    Output("store-bg-csv",        "data"),
    Output("bg-download-area",    "children"),
    Input("bg-elem-store",        "data"),
    Input("store-sample-seed",    "data"),
    prevent_initial_call=True,
)
def bg_update_all(included, seed):
    included = included or []

    # ── filter DF1 ONCE — shared by scatter, histogram, and temporal ──────────
    df_f = filter_by_included_df1(DF1, included)

    # Filter info banner
    if not included:
        info = dbc.Alert("Showing all materials. Click elements on the periodic table to filter.",
                         color="secondary")
    elif df_f.empty:
        info = dbc.Alert("⚠️ No materials contain only these elements.", color="warning")
    else:
        info = dbc.Alert(
            f"🔬 Included: {', '.join(sorted(included))} — "
            f"Showing {len(df_f)} of {len(DF1)} materials ({len(df_f)/len(DF1)*100:.1f}%)",
            color="info",
        )

    empty_fig = go.Figure()
    empty_fig.update_layout(
        annotations=[dict(text="⚠️ No materials match these elements.",
                          showarrow=False, font=dict(size=14))],
        template="plotly_white",
    )

    if df_f.empty:
        return info, empty_fig, empty_fig, empty_fig, html.P("No data."), None, html.Div()

    # Detect bandgap column name
    bg_col = next(
        (c for c in ["Bandgap", "bandgap", "Band_gap", "band_gap", "Value", "BandGap"]
         if c in df_f.columns), None
    )

    # Top 9 most-represented material names — used by scatter, histogram, temporal
    df_agg    = (df_f.groupby("Name").size().reset_index(name="Count")
                 .sort_values("Count", ascending=False).head(9))
    top_names = df_agg["Name"].tolist()
    top9      = top_names[:9]

    # ── Bandgap scatter ───────────────────────────────────────────────────────
    if bg_col and top_names:
        fig_sc = px.scatter(
            df_f[df_f["Name"].isin(top_names)], x=bg_col, y="Name", color="Name",
            color_discrete_sequence=pick_palette(len(top_names)),
            title="Bandgap Distribution by Semiconductor",
            labels={bg_col: "Bandgap (eV)", "Name": "Semiconductor"},
            height=500, hover_data={bg_col: ":.2f"},
        )
        fig_sc.update_traces(marker=dict(size=10, opacity=0.9))
        fig_sc.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig_sc.update_yaxes(showgrid=False)
        fig_sc.update_layout(template="plotly_white", hovermode="closest")
    else:
        fig_sc = go.Figure()

    # ── Histogram grid ────────────────────────────────────────────────────────
    if bg_col and top9:
        df_h = df_f[df_f["Name"].isin(top9)].copy()
        df_h = df_h[pd.to_numeric(df_h[bg_col], errors="coerce").notna()]
        df_h[bg_col] = df_h[bg_col].astype(float)
        fig_hist = make_subplots(
            rows=3, cols=3,
            subplot_titles=[f"{n} (n={len(df_h[df_h['Name']==n])})" for n in top9],
            shared_xaxes=True,
        )
        palette = pick_palette(len(top9))
        for i, name in enumerate(top9):
            r, c = divmod(i, 3)
            sub  = df_h.loc[df_h["Name"] == name, bg_col].dropna().values
            if sub.size >= 1:
                fig_hist.add_trace(
                    go.Histogram(x=sub, name=name, marker_color=palette[i],
                                 opacity=0.85, showlegend=False),
                    row=r+1, col=c+1,
                )
                if sub.size >= 2:
                    fig_hist.add_vline(x=float(np.median(sub)), line_dash="dash",
                                       line_color="red", row=r+1, col=c+1)
        fig_hist.update_yaxes(type="log")
        fig_hist.update_layout(
            title="Bandgap Histogram Grid (log y-scale; dashed = median)",
            template="plotly_white", height=700,
        )
    else:
        fig_hist = go.Figure()

    # ── Temporal scatter ──────────────────────────────────────────────────────
    unique_list = df_agg["Name"].dropna().unique().tolist()
    df2_doi = (
        DF2[DF2["Name"].isin(unique_list)]
        .drop(columns=["index", "Composition", "Confidence", "Publisher"], errors="ignore")
    )
    df2_grp = (
        df2_doi.groupby(["Date", "Name", "Value"])
        .size().reset_index(name="Frequency").reset_index()
    )

    if df2_grp.empty or "Value" not in df2_grp.columns:
        fig_t    = empty_fig
        table_el = html.P("No data.")
        csv_data = None
        dl_btn   = html.Div()
    else:
        # OPT: Value is pre-converted to numeric at startup — skip pd.to_numeric here
        df2_grp["Frequency"] = pd.to_numeric(df2_grp["Frequency"], errors="coerce")
        df2_grp  = df2_grp.dropna(subset=["Date", "Value", "Frequency", "Name"])
        df2_plot = df2_grp.sort_values("Date")

        vmax  = float(df2_plot["Value"].max()) if not df2_plot.empty else 4.0
        fig_t = px.scatter(
            df2_plot, x="Date", y="Value", color="Name",
            size="Frequency", size_max=30,
            labels={"Value": "Bandgap Energy (eV)", "Date": "Publication Date"},
            title="Bandgap Trends Over Time",
            template="plotly_white", height=500,
        )
        for lo, hi, col, lbl in [
            (0,    1.6,            "rgba(255,255,0,0.10)", "Infrared (0–1.6 eV)"),
            (1.6,  3.26,           "rgba(0,200,0,0.10)",   "Visible (1.6–3.26 eV)"),
            (3.26, max(vmax, 4.0), "rgba(255,0,0,0.10)",   "Ultraviolet (3.26+ eV)"),
        ]:
            fig_t.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0,
                            annotation_text=lbl, annotation_position="top left",
                            annotation_font_size=10)
        fig_t.update_layout(hovermode="closest")

        # Sample table
        n = min(10, len(df2_doi))
        if n == 0:
            table_el = html.P("No records to display.")
        else:
            sample = df2_doi.sample(n=n, random_state=seed).reset_index(drop=True)
            if "Date" in sample.columns:
                sample["Date"] = pd.to_datetime(sample["Date"], errors="coerce").dt.strftime("%b %Y")
            sample   = sample.drop(columns=[c for c in sample.columns if c.lower() == "year"],
                                   errors="ignore")
            table_el = dbc.Table.from_dataframe(sample, striped=True, bordered=True,
                                                 hover=True, responsive=True, className="small")

        csv_data = df2_doi.to_csv(index=False)
        dl_btn   = html.Div(
            dbc.Button("⬇️ Download filtered data as CSV", id="bg-dl-btn",
                       color="secondary", size="sm", className="mt-2"),
        )

    return info, fig_sc, fig_hist, fig_t, table_el, csv_data, dl_btn


@app.callback(
    Output("store-sample-seed", "data"),
    Input("bg-shuffle-btn", "n_clicks"),
    prevent_initial_call=True,
)
def shuffle_sample(_):
    return random.randint(0, 10**9)


@app.callback(
    Output("dl-csv", "data"),
    Input("bg-dl-btn", "n_clicks"),
    State("store-bg-csv", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, csv_str):
    if not n_clicks or not csv_str:
        return no_update
    if ctx.triggered_id != "bg-dl-btn":
        return no_update
    return dcc.send_string(csv_str, "bandgap-filtered.csv", type="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# DECISION PAGE CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("dec-excl-info", "children"),
    Input("store-excl", "data"),
)
def dec_excl_preview(excluded):
    excluded = excluded or []
    if not excluded:
        return dbc.Alert("Click elements on the table above to exclude them from analysis.",
                         color="secondary", className="py-2")
    preview = filter_by_excluded(DF1, excluded)
    removed = len(DF1) - len(preview)
    return dbc.Alert(
        f"🚫 Excluded: {', '.join(sorted(excluded))} — "
        f"removes {removed} materials ({removed/len(DF1)*100:.1f}%) — "
        f"{len(preview)} remaining.",
        color="warning", className="py-2",
    )


@app.callback(
    [Output(f"if-{FILTER_SID[f]}-wrap", "style") for f in FILTER_OPTIONS],
    Input("dec-filter-select", "value"),
)
def show_initial_filter_slider(selected):
    return [{"display": "block"} if f == selected else {"display": "none"}
            for f in FILTER_OPTIONS]


@app.callback(
    [Output(f"ef-{FILTER_SID[f]}-wrap", "style") for f in FILTER_OPTIONS],
    Input("dec-extra-select", "value"),
)
def show_extra_filter_sliders(selected_list):
    selected_list = selected_list or []
    return [{"display": "block"} if f in selected_list else {"display": "none"}
            for f in FILTER_OPTIONS]


@app.callback(
    Output("dec-extra-select", "options"),
    Output("dec-extra-select", "value"),
    Input("dec-filter-select", "value"),
    State("dec-extra-select",  "value"),
)
def sync_extra_options(initial, extra_vals):
    opts    = [{"label": f, "value": f} for f in FILTER_OPTIONS if f != initial]
    cleaned = [v for v in (extra_vals or []) if v != initial]
    return opts, cleaned


@app.callback(
    Output("store-dec-filters",  "data"),
    Output("dec-apply-status",   "children"),
    Input("dec-apply-btn",       "n_clicks"),
    State("dec-bg-min",          "value"),
    State("dec-bg-max",          "value"),
    State("dec-filter-select",   "value"),
    *[State(f"if-{FILTER_SID[f]}", "value") for f in FILTER_OPTIONS],
    State("dec-extra-select",    "value"),
    *[State(f"ef-{FILTER_SID[f]}", "value") for f in FILTER_OPTIONS],
    prevent_initial_call=True,
)
def apply_filters(_, bg_min, bg_max, init_filter, *args):
    n_filters = len(FILTER_OPTIONS)
    if_vals   = list(args[:n_filters])
    extra_sel = args[n_filters]
    ef_vals   = list(args[n_filters + 1: n_filters * 2 + 1])

    if bg_min is None or bg_max is None:
        return no_update, dbc.Alert("❌ Please fill in bandgap range.", color="danger")
    if bg_min > bg_max:
        return no_update, dbc.Alert("❌ Min bandgap must be ≤ max bandgap.", color="danger")

    def _make_range(fname, val):
        """Convert a slider value to a [lo, hi] pair.
        Single-handle sliders (Reserve, Production) return a float — pair it
        with the data maximum so the filter means 'at least X'.
        RangeSliders return a [lo, hi] list — use it directly.
        """
        if val is None:
            return None
        if fname in SINGLE_MIN_FILTERS:
            return [val, float(DF1[fname].max())]
        return val   # already [lo, hi] from RangeSlider

    filters = {"Bandgap": [bg_min, bg_max]}
    if init_filter:
        val = _make_range(init_filter, if_vals[FILTER_OPTIONS.index(init_filter)])
        if val:
            filters[init_filter] = val
    for fname in (extra_sel or []):
        val = _make_range(fname, ef_vals[FILTER_OPTIONS.index(fname)])
        if val:
            filters[fname] = val

    return filters, dbc.Alert(f"✅ {len(filters)} filter(s) applied successfully!",
                               color="success", className="py-2")


@app.callback(
    Output("dec-scatter",      "figure"),
    Output("dec-filter-info",  "children"),
    Output("dec-axis-info",    "children"),
    Output("dec-mcdm-wrapper", "style"),
    Output("dec-mcdm-info",    "children"),
    Input("store-dec-filters", "data"),
    Input("store-excl",        "data"),
    Input("dec-log-y",         "value"),
)
def dec_update_scatter(filters, excluded, log_y_vals):
    log_y = "log" in (log_y_vals or [])
    df_ex = filter_by_excluded(DF1, excluded or [])

    if not filters:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="Apply filters to see results",
                              showarrow=False, font=dict(size=14))],
            template="plotly_white",
        )
        return (fig, dbc.Alert("📈 No filters applied yet.", color="secondary"),
                html.Div(), {"display": "none"}, html.Div())

    df_f  = filter_df(df_ex, {k: v for k, v in filters.items() if k in DF1.columns})
    x_col = "Bandgap"
    y_col = next((k for k in filters if k != "Bandgap"), x_col)

    if df_f.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="⚠️ No materials match the current filters",
                              showarrow=False)],
            template="plotly_white",
        )
        return (fig, dbc.Alert("⚠️ No materials match.", color="warning"),
                html.Div(), {"display": "none"}, html.Div())

    fig      = professional_scatter(df_f, x_col, y_col, f"{x_col} vs {y_col}",
                                     x_col, y_col, log_y=log_y)
    info     = dbc.Alert(
        f"📊 Showing {len(df_f)} materials | Filters: {', '.join(filters)} | "
        f"Available after exclusion: {len(df_ex)}",
        color="info", className="py-2",
    )
    axis_inf = html.Span([html.Strong("X-axis: "), x_col, "  |  ",
                           html.Strong("Y-axis: "), y_col])
    mcdm_inf = dbc.Alert(
        f"Analyze the {len(df_f)} filtered materials using TOPSIS or PROMETHEE.",
        color="primary", className="py-2",
    )
    return fig, info, axis_inf, {"display": "block"}, mcdm_inf






@app.callback(
    Output("mcdm-manual-section", "style"),
    Input("mcdm-weighting", "value"),
)
def toggle_manual(weighting):
    return {"display": "block"} if weighting == "Manual" else {"display": "none"}


LONG_TERM_HIGH  = {"ESG Score", "Toxicity", "Companionality", "Reserve (ton)"}
SHORT_TERM_HIGH = {"Production (ton)", "HHI (USGS)", "CO2 footprint max (kg/kg)",
                   "Water usage max (l/kg)", "Embodied energy max (MJ/kg)"}

@app.callback(
    [Output(f"weight-{CRIT_SID[c]}", "value") for c in CRITERIA_OPTIONS],
    Input("preset-balanced",  "n_clicks"),
    Input("preset-longterm",  "n_clicks"),
    Input("preset-shortterm", "n_clicks"),
    prevent_initial_call=True,
)
def apply_preset(_, __, ___):
    triggered = ctx.triggered_id
    if triggered == "preset-balanced":
        vals = {c: 3 for c in CRITERIA_OPTIONS}
    elif triggered == "preset-longterm":
        vals = {c: 5 if c in LONG_TERM_HIGH  else 1 for c in CRITERIA_OPTIONS}
    elif triggered == "preset-shortterm":
        vals = {c: 5 if c in SHORT_TERM_HIGH else 1 for c in CRITERIA_OPTIONS}
    else:
        vals = {c: 3 for c in CRITERIA_OPTIONS}
    return [vals[c] for c in CRITERIA_OPTIONS]


@app.callback(
    Output("mcdm-weights-table", "children"),
    Input("mcdm-weighting",      "value"),
    Input("store-dec-filters",   "data"),
    Input("store-excl",          "data"),
    *[Input(f"weight-{CRIT_SID[c]}", "value") for c in CRITERIA_OPTIONS],
)
def show_weights_table(weighting, filters, excluded, *raw_w):
    if not filters:
        return html.Div()
    df_ex = filter_by_excluded(DF1, excluded or [])
    df_f  = filter_df(df_ex, {k: v for k, v in filters.items() if k in DF1.columns})
    avail = {k: v for k, v in CRITERIA_OPTIONS.items() if k in df_f.columns}
    if weighting == "Manual":
        crit_names = list(CRITERIA_OPTIONS.keys())
        w_arr  = np.array([raw_w[crit_names.index(k)] for k in avail], dtype=float)
        s      = w_arr.sum()
        w_norm = w_arr / s if s > 0 else np.ones(len(avail)) / len(avail)
        wdf    = pd.DataFrame({
            "Criterion": list(avail.keys()),
            "Weight":    [f"{w:.2%}" for w in w_norm],
            "Direction": ["Maximize" if d == 1 else "Minimize" for d in avail.values()],
        }).sort_values("Weight", ascending=False).reset_index(drop=True)
        wdf.index = wdf.index + 1
        return html.Div([
            html.H6("Criteria Weights"),
            dbc.Table.from_dataframe(wdf, striped=True, bordered=True,
                                     hover=True, responsive=True, className="small"),
        ])
    return dbc.Alert("✅ Weights computed automatically via Entropy method.", color="info")

@app.callback(
    Output("mcdm-status",        "children"),
    Output("mcdm-results-area",  "children"),
    Output("store-mcdm-results", "data"),
    Output("mcdm-dl-wrapper",    "style"),
    Input("mcdm-run-btn",        "n_clicks"),
    State("store-dec-filters",   "data"),
    State("store-excl",          "data"),
    State("mcdm-method",         "value"),
    State("mcdm-weighting",      "value"),
    *[State(f"weight-{CRIT_SID[c]}", "value") for c in CRITERIA_OPTIONS],
    prevent_initial_call=True,
)
def run_mcdm(_, filters, excluded, method, weighting, *raw_w):
    _hide = {"display": "none"}
    if not filters:
        return dbc.Alert("❌ Apply filters first.", color="danger"), html.Div(), no_update, _hide

    df_ex = filter_by_excluded(DF1, excluded or [])
    df_f  = filter_df(df_ex, {k: v for k, v in filters.items() if k in DF1.columns})
    avail = {k: v for k, v in CRITERIA_OPTIONS.items() if k in df_f.columns}

    if not avail:
        return dbc.Alert("❌ No criteria columns found.", color="danger"), html.Div(), no_update, _hide
    if df_f.empty:
        return dbc.Alert("❌ No materials in filtered set.", color="danger"), html.Div(), no_update, _hide

    crit_cols = list(avail.keys())
    types     = np.array(list(avail.values()))

    # Drop rows that have N/A in ANY selected criterion.
    # Materials with N/A in non-selected columns are NOT affected — they
    # remain visible in the scatter plot and pass through the filters above.
    nan_rows   = df_f[crit_cols].isna().any(axis=1)
    n_dropped  = int(nan_rows.sum())
    df_mcdm    = df_f[~nan_rows].copy()

    if df_mcdm.empty:
        return (dbc.Alert("❌ No materials remain after removing N/A criteria rows.",
                          color="danger"), html.Div(), no_update, _hide)

    matrix = df_mcdm[crit_cols].values

    if np.any(matrix < 0) and weighting == "Entropy":
        return (dbc.Alert("❌ Entropy weighting requires non-negative values.", color="danger"),
                html.Div(), no_update, _hide)

    if weighting == "Entropy":
        try:
            weights = entropy_weights(matrix)
        except Exception:
            weights = np.ones(len(avail)) / len(avail)
    else:
        crit_names = list(CRITERIA_OPTIONS.keys())
        raw     = np.array([raw_w[crit_names.index(k)] for k in avail], dtype=float)
        weights = raw / raw.sum() if raw.sum() > 0 else np.ones(len(avail)) / len(avail)

    if not np.isclose(weights.sum(), 1.0):
        weights /= weights.sum()

    try:
        if method == "TOPSIS":
            scores  = TOPSIS()(matrix, weights, types)
            results = pd.DataFrame({
                "Material":     df_mcdm["Name"].values,
                "Bandgap (eV)": df_mcdm["Bandgap"].values,
                "DOI":          df_mcdm["DOI"].values if "DOI" in df_mcdm.columns else [""] * len(df_mcdm),
                "Score":        scores,
            }).sort_values("Score", ascending=False).reset_index(drop=True)
            score_col = "Score"
        else:
            flows   = PROMETHEE_II("usual")(matrix, weights, types)
            results = pd.DataFrame({
                "Material":     df_mcdm["Name"].values,
                "Bandgap (eV)": df_mcdm["Bandgap"].values,
                "Net Flow":     flows,
            }).sort_values("Net Flow", ascending=False).reset_index(drop=True)
            score_col = "Net Flow"
    except Exception as e:
        return (dbc.Alert(f"❌ Error running {method}: {e}", color="danger"),
                html.Div(), no_update, _hide)

    results.index      = results.index + 1
    results.index.name = "Rank"

    if np.isnan(results[score_col].values).any():
        return (dbc.Alert(f"❌ {method} returned NaN values.", color="danger"),
                html.Div(), no_update, _hide)

    wdf = pd.DataFrame({
        "Criterion": list(avail.keys()), "Weight": weights,
        "Direction": ["Maximize" if d == 1 else "Minimize" for d in avail.values()],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    wdf.index = wdf.index + 1; wdf.index.name = "Rank"

    top3       = results.drop_duplicates(subset=["Material"]).head(3)
    top3_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P(f"Rank #{top3.index[i]}", className="text-muted mb-1 small"),
            html.H5(top3.iloc[i]["Material"], className="mb-0"),
        ])), width=4)
        for i in range(len(top3))
    ])

    display_cols  = ["Material", "Bandgap (eV)"] + (["DOI"] if "DOI" in results.columns else []) + [score_col]
    results_table = dbc.Table.from_dataframe(
        results[display_cols].head(100).reset_index().rename(columns={"index": "Rank"}),
        striped=True, bordered=True, hover=True, responsive=True, className="small",
    )

    # OPT: store only the ranked material names + scores needed to rebuild
    #      the Excel — avoids re-filtering DF1 in download_mcdm.
    results_store = {
        "method":    method,
        "materials": results["Material"].tolist(),
        "scores":    results[score_col].tolist(),
        "bandgaps":  results["Bandgap (eV)"].tolist(),
        "doi":       results["DOI"].tolist() if "DOI" in results.columns else [],
        "filters":   filters,
        "excluded":  excluded or [],
        "criteria":  list(avail.keys()),
        "weights":   weights.tolist(),
        "score_col": score_col,
        # OPT: store full filtered rows as records so download_mcdm skips re-filtering
        "full_data": df_mcdm.to_dict("records"),
    }

    results_ui = html.Div([
        html.H5("MCDM Results"),
        html.P(f"Showing top 100 results (out of {len(results)} total)"),
        results_table,
        html.H5("🏆 Top Materials", className="mt-3"),
        top3_cards,
    ])
    na_note = (dbc.Alert(f"ℹ️ {n_dropped} material(s) with N/A in selected criteria were excluded from MCDM.",
                         color="warning", className="py-2 mt-1")
               if n_dropped > 0 else html.Div())
    return (html.Div([dbc.Alert(f"✅ {method} analysis complete!", color="success", className="py-2"), na_note]),
            results_ui, results_store, {"display": "block"})


@app.callback(
    Output("dl-mcdm", "data"),
    Input("mcdm-dl-btn", "n_clicks"),
    State("store-mcdm-results", "data"),
    prevent_initial_call=True,
)
def download_mcdm(_, store):
    if not store:
        return no_update

    score_col = store["score_col"]

    # OPT: use stored full_data instead of re-filtering DF1
    df_f = pd.DataFrame(store["full_data"])

    rdata = {"Material": store["materials"], "Bandgap (eV)": store["bandgaps"]}
    if store["doi"]:
        rdata["DOI"] = store["doi"]
    rdata[score_col] = store["scores"]
    results_df = pd.DataFrame(rdata)
    results_df.index = results_df.index + 1; results_df.index.name = "Rank"

    wdf = pd.DataFrame({
        "Criterion": store["criteria"],
        "Weight":    store["weights"],
        "Direction": ["Maximize" if CRITERIA_OPTIONS.get(c, -1) == 1 else "Minimize"
                      for c in store["criteria"]],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    wdf.index = wdf.index + 1; wdf.index.name = "Rank"

    return dcc.send_bytes(
        build_excel(df_f, results_df, wdf, store["filters"]),
        f"mcdm_analysis_{store['method']}.xlsx",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PERIODIC TABLE CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("bg-elem-store", "data"),
    Input({"type": "bg-elem", "Z": ALL}, "n_clicks"),
    State("bg-elem-store", "data"),
    prevent_initial_call=True,
)
def bg_toggle_element(_, current):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return no_update
    sym     = PT_BY_Z[triggered["Z"]]["sym"]
    current = current or []
    return [s for s in current if s != sym] if sym in current else current + [sym]


@app.callback(
    [Output({"type": "bg-elem", "Z": e["Z"]}, "style") for e in PT_ELEMENTS],
    Input("bg-elem-store", "data"),
)
def bg_update_pt_styles(active_syms):
    # OPT: convert list to set once for O(1) membership tests across all cells
    active_set = set(active_syms or [])
    return [_pt_cell_style(e, active_set, "include") for e in PT_ELEMENTS]


@app.callback(
    Output("store-excl", "data"),
    Input({"type": "dec-elem", "Z": ALL}, "n_clicks"),
    State("store-excl", "data"),
    prevent_initial_call=True,
)
def dec_toggle_element(_, current):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return no_update
    sym     = PT_BY_Z[triggered["Z"]]["sym"]
    current = current or []
    return [s for s in current if s != sym] if sym in current else current + [sym]


@app.callback(
    [Output({"type": "dec-elem", "Z": e["Z"]}, "style") for e in PT_ELEMENTS],
    Input("store-excl", "data"),
)
def dec_update_pt_styles(active_syms):
    # OPT: convert list to set once for O(1) membership tests across all cells
    active_set = set(active_syms or [])
    return [_pt_cell_style(e, active_set, "exclude") for e in PT_ELEMENTS]


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 8050))
    host  = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("RENDER") is None
    app.run(debug=debug, host=host, port=port)
