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
    """Encode a column name into a valid HTML-id fragment."""
    return re.sub(r"[^a-zA-Z0-9]", "-", s).strip("-")

FILTER_SID  = {f: _sid(f) for f in FILTER_OPTIONS}
CRIT_SID    = {f: _sid(f) for f in CRITERIA_OPTIONS}

# ─────────────────────────────────────────────────────────────────────────────
# PERIODIC TABLE — element data, styles, grid builder
# Cells use {"type": "<prefix>-elem", "Z": z} so bg and dec tables are independent.
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
PT_SYM_SET  = {e["sym"] for e in PT_ELEMENTS}   # valid symbols from CSV

PT_CATS = {
    "alkali metal":    {"bg":"#e74c3c","text":"#fff"},
    "alkaline earth":  {"bg":"#e67e22","text":"#fff"},
    "transition":      {"bg":"#2980b9","text":"#fff"},
    "post-transition": {"bg":"#27ae60","text":"#fff"},
    "metalloid":       {"bg":"#16a085","text":"#fff"},
    "nonmetal":        {"bg":"#f39c12","text":"#1a1a1a"},
    "halogen":         {"bg":"#8e44ad","text":"#fff"},
    "noble gas":       {"bg":"#2c3e50","text":"#ecf0f1"},
    "lanthanide":      {"bg":"#c0392b","text":"#fff"},
    "actinide":        {"bg":"#d35400","text":"#fff"},
    "unknown":         {"bg":"#7f8c8d","text":"#fff"},
}
PT_CAT_LABELS = {
    "alkali metal":"Alkali Metal", "alkaline earth":"Alkaline Earth Metal",
    "transition":"Transition Metal", "post-transition":"Post-Transition Metal",
    "metalloid":"Metalloid", "nonmetal":"Nonmetal", "halogen":"Halogen",
    "noble gas":"Noble Gas", "lanthanide":"Lanthanide",
    "actinide":"Actinide", "unknown":"Unknown",
}
_CW, _CH = 46, 46   # cell px


def _pt_cell_style(elem, active_syms: list, mode: str) -> dict:
    """
    mode='include': active cells highlighted gold, inactive dimmed.
    mode='exclude': active cells highlighted red, inactive normal.
    """
    cat = elem["cat"]
    sym = elem["sym"]
    is_active = sym in active_syms
    s = {
        "gridRow": elem["period"], "gridColumn": elem["group"],
        "width": f"{_CW}px", "height": f"{_CH}px",
        "padding": "2px 2px", "borderRadius": "4px",
        "cursor": "pointer", "userSelect": "none",
        "display": "flex", "flexDirection": "column",
        "justifyContent": "space-between",
        "overflow": "hidden", "boxSizing": "border-box",
        "transition": "transform 0.12s, box-shadow 0.12s, opacity 0.12s",
        "backgroundColor": PT_CATS[cat]["bg"],
        "color": PT_CATS[cat]["text"],
        "border": "2px solid transparent",
        "opacity": "1", "transform": "scale(1)",
        "zIndex": "1", "boxShadow": "none",
    }
    if is_active:
        if mode == "exclude":
            s["border"]    = "2px solid #ff4757"
            s["boxShadow"] = "0 0 0 2px rgba(255,71,87,0.8)"
        else:
            s["border"]    = "2px solid #f9ca24"
            s["boxShadow"] = "0 0 0 2px rgba(249,202,36,0.8)"
        s["transform"] = "scale(1.08)"
        s["zIndex"]    = "15"
    elif active_syms and not is_active:
        s["opacity"] = "0.35"
    return s


def _pt_make_cell(elem, prefix: str) -> html.Div:
    return html.Div(
        id={"type": f"{prefix}-elem", "Z": elem["Z"]},
        n_clicks=0,
        style=_pt_cell_style(elem, [], "include"),
        children=[
            html.Div(str(elem["Z"]),
                     style={"fontSize":"7px","opacity":"0.8","lineHeight":"1"}),
            html.Div(elem["sym"],
                     style={"fontSize":"15px","fontWeight":"700",
                             "textAlign":"center","lineHeight":"1.1"}),
            html.Div(elem["name"],
                     style={"fontSize":"5.5px","textAlign":"center",
                             "overflow":"hidden","whiteSpace":"nowrap",
                             "textOverflow":"ellipsis","lineHeight":"1"}),
        ],
    )


def _pt_make_legend() -> html.Div:
    chips = [
        html.Div([
            html.Div(style={"width":"11px","height":"11px","borderRadius":"2px",
                             "backgroundColor": PT_CATS[k]["bg"],"flexShrink":"0"}),
            html.Span(PT_CAT_LABELS[k], style={"fontSize":"10px","whiteSpace":"nowrap"}),
        ], style={"display":"flex","alignItems":"center","gap":"4px"})
        for k in PT_CATS
    ]
    return html.Div(chips, style={
        "display":"flex","flexWrap":"wrap","gap":"5px 14px","marginTop":"8px",
    })


def build_periodic_table(prefix: str, mode: str = "include") -> html.Div:
    """
    prefix: 'bg'  → id type 'bg-elem'   (bandgap inclusion)
            'dec' → id type 'dec-elem'  (decision exclusion)
    mode:   'include' | 'exclude'  (affects highlight colour)
    """
    verb  = "include" if mode == "include" else "exclude"
    color = "#f9ca24" if mode == "include" else "#ff4757"
    hint  = html.Div([
        html.Span("● ", style={"color": color, "fontWeight":"700"}),
        html.Span(f"Click elements to {verb}. Click again to deselect. "),
        html.Span("Selected elements dimly grey out the rest.",
                  style={"color":"#6c757d"}),
    ], style={"fontSize":"12px","marginBottom":"8px"})

    cells = [_pt_make_cell(e, prefix) for e in PT_ELEMENTS]

    # * / ** placeholders
    for row, label, cat in [(6,"*","lanthanide"),(7,"**","actinide")]:
        cells.append(html.Div(label, style={
            "gridRow":row,"gridColumn":3,
            "width":f"{_CW}px","height":f"{_CH}px",
            "backgroundColor":PT_CATS[cat]["bg"],"color":PT_CATS[cat]["text"],
            "display":"flex","alignItems":"center","justifyContent":"center",
            "borderRadius":"4px","fontSize":"14px","fontWeight":"700",
            "cursor":"default","userSelect":"none",
        }))

    # Period labels
    for p in range(1, 8):
        cells.append(html.Div(str(p), style={
            "gridRow":p,"gridColumn":"1","marginLeft":"-20px",
            "color":"#adb5bd","fontSize":"10px","fontWeight":"600",
            "display":"flex","alignItems":"center","justifyContent":"flex-end",
            "width":"16px","height":f"{_CH}px","paddingRight":"3px",
        }))
    for frow, lbl in [(9,"6"),(10,"7")]:
        cells.append(html.Div(lbl, style={
            "gridRow":frow,"gridColumn":"3","marginLeft":"-20px",
            "color":"#adb5bd","fontSize":"9px",
            "display":"flex","alignItems":"center","justifyContent":"flex-end",
            "width":"16px","height":f"{_CH}px","paddingRight":"3px",
        }))

    # Group labels
    for g in range(1, 19):
        cells.append(html.Div(str(g), style={
            "gridRow":"1","gridColumn":g,
            "color":"#adb5bd","fontSize":"9px","fontWeight":"600",
            "textAlign":"center","marginTop":"-16px","height":"16px",
            "display":"flex","alignItems":"flex-end","justifyContent":"center",
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
        html.Div(grid, style={"overflowX":"auto","paddingBottom":"6px"}),
        _pt_make_legend(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DATA  –  loaded once at server start; no per-request overhead
# ─────────────────────────────────────────────────────────────────────────────
DF1 = pd.read_excel("8_material_properties_cleaned.xlsx")
DF2 = pd.read_excel("materials_high_confidence_cleaned.xlsx").iloc[:, 1:]
DF2["Date"] = pd.to_datetime(DF2["Date"], errors="coerce")

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
    excl = {str(e).strip() for e in excluded}
    valid = [c for c in ELEMENT_COLUMNS if c in df.columns]
    mask = pd.concat(
        [df[c].fillna("").astype(str).str.strip().isin(excl) for c in valid], axis=1
    ).any(axis=1)
    return df[~mask]


def filter_by_included_df1(df: pd.DataFrame, included: list) -> pd.DataFrame:
    if not included:
        return df          # show everything when no elements are selected
    inc = set(included)
    mask = pd.Series(True, index=df.index)
    for c in ELEMENT_COLUMNS:
        if c in df.columns:
            mask &= df[c].isin(inc) | df[c].isna() | (df[c].astype(str).str.strip() == "")
    return df[mask]


def filter_by_included_df2(df: pd.DataFrame, included: list) -> pd.DataFrame:
    if not included:
        return df.iloc[0:0]
    inc = set(included)
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
    dp = df.copy()
    if log_x: dp[x_col] = dp[x_col].clip(lower=1e-10)
    if log_y: dp[y_col] = dp[y_col].clip(lower=1e-10)
    hi = np.random.choice(len(dp), min(10, len(dp)), replace=False)
    dp["_hi"] = False
    dp.iloc[hi, dp.columns.get_loc("_hi")] = True
    reg, hil = dp[~dp["_hi"]], dp[dp["_hi"]]
    fig = go.Figure([
        go.Scatter(
            x=reg[x_col], y=reg[y_col], mode="markers", name="All Materials",
            marker=dict(size=8, color=primary, opacity=0.6),
            text=reg["Name"],
            hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>",
        ),
        go.Scatter(
            x=hil[x_col], y=hil[y_col], mode="markers+text", name="Highlighted",
            marker=dict(size=12, color=accent),
            text=hil["Name"], textposition="top center",
            textfont=dict(color=accent, size=10),
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
    rr = results_df.reset_index()
    fd = filtered_df.copy()
    if "Score" in rr.columns:
        fd["TOPSIS_Score"] = fd["Name"].map(dict(zip(rr["Material"], rr["Score"])))
        fd["TOPSIS_Rank"]  = fd["Name"].map(dict(zip(rr["Material"], rr["Rank"])))
    else:
        fd["PROMETHEE_Net_Flow"] = fd["Name"].map(dict(zip(rr["Material"], rr["Net Flow"])))
        fd["PROMETHEE_Rank"]     = fd["Name"].map(dict(zip(rr["Material"], rr["Rank"])))
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        fd.to_excel(w, sheet_name="Full Data", index=False)
        rr.to_excel(w, sheet_name="Rankings", index=False)
        weights_df.reset_index().to_excel(w, sheet_name="Weights", index=False)
        if filters:
            pd.DataFrame(
                [{"Filter": k, "Min": v[0], "Max": v[1]} for k, v in filters.items()]
            ).to_excel(w, sheet_name="Filter Settings", index=False)
    return out.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# PRE-RENDERED SLIDER BANKS
# Each filter / criterion gets its own slider div at layout-build time.
# Callbacks toggle display; Apply reads State() from every slider.
# ─────────────────────────────────────────────────────────────────────────────
def _make_filter_slider_bank(prefix: str) -> list:
    """One hidden RangeSlider per FILTER_OPTION, keyed by prefix."""
    out = []
    for fname in FILTER_OPTIONS:
        sid   = FILTER_SID[fname]
        fmin  = float(DF1[fname].min())
        fmax  = float(DF1[fname].max())
        step  = 1 if fname == "Toxicity" else None
        out.append(html.Div(
            id=f"{prefix}-{sid}-wrap",
            style={"display": "none"},
            children=[
                dbc.Label(f"{fname} range"),
                dcc.RangeSlider(
                    id=f"{prefix}-{sid}",
                    min=fmin, max=fmax, value=[fmin, fmax],
                    step=step, marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Div(id=f"{prefix}-{sid}-note", className="text-muted small mb-2"),
            ],
        ))
    return out


def _make_weight_sliders() -> list:
    return [
        dbc.Col([
            dbc.Label(cname, className="small fw-semibold"),
            dcc.Slider(
                id=f"weight-{CRIT_SID[cname]}",
                min=0, max=5, step=1, value=3,
                marks={i: str(i) for i in range(6)},
                className="mb-1",
            ),
        ], width=12, lg=4, className="mb-3")
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
                dbc.Alert(
                    "💡 Pro Tip: Use the MCDM analysis for ranking the most promising semiconductors.",
                    color="info", className="mt-3",
                ),
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
        dbc.Button("🔀 Shuffle sample", id="bg-shuffle-btn", color="secondary", size="sm",
                   className="mt-2 mb-2"),
        html.Div(id="bg-sample-table"),
        html.Div(id="bg-download-area"),
    ])


def layout_decision() -> html.Div:
    ifilter_bank = _make_filter_slider_bank("if")   # initial-filter sliders
    efilter_bank = _make_filter_slider_bank("ef")   # extra-filter sliders
    weight_sliders = _make_weight_sliders()

    return html.Div([
        html.H1("Decision-making Assistant"),
        html.P("Facilitate semiconductor selection with advanced filtering and visualization"),

        # ── 1. Element Exclusion ─────────────────────────────────────────────
        html.H4("1. Element Exclusion"),
        build_periodic_table("dec", mode="exclude"),
        html.Div(id="dec-excl-info", className="mb-3 mt-2"),

        # ── 2. Initial Filters ───────────────────────────────────────────────
        html.H4("2. Initial Filters"),
        dbc.Row([
            dbc.Col([
                html.H6("Bandgap Selection"),
                dbc.Row([
                    dbc.Col([dbc.Label("Min (eV)"),
                             dbc.Input(id="dec-bg-min", type="number",
                                       min=0, max=35, step=0.1, value=0.0)]),
                    dbc.Col([dbc.Label("Max (eV)"),
                             dbc.Input(id="dec-bg-max", type="number",
                                       min=0, max=35, step=0.1, value=3.0)]),
                ]),
                html.Div(id="dec-bg-error"),
            ], width=6),
            dbc.Col([
                html.H6("Additional Filter"),
                dcc.Dropdown(
                    id="dec-filter-select",
                    options=[{"label": f, "value": f} for f in FILTER_OPTIONS],
                    value=FILTER_OPTIONS[0], clearable=False, className="mb-2",
                ),
                html.Div(ifilter_bank),  # all pre-rendered, one shown at a time
            ], width=6),
        ]),

        # ── 3. Extra Filters ─────────────────────────────────────────────────
        html.H4("3. Additional Filters (Optional)", className="mt-4"),
        html.P("Pick any number of extra filters:"),
        dcc.Dropdown(
            id="dec-extra-select",
            options=[{"label": f, "value": f} for f in FILTER_OPTIONS],
            multi=True, placeholder="Select additional filters…", className="mb-3",
        ),
        html.Div(efilter_bank),  # all pre-rendered, shown based on selection

        # ── Apply ────────────────────────────────────────────────────────────
        dbc.Button("✅ Apply Filters", id="dec-apply-btn",
                   color="primary", size="lg", className="mt-3 mb-2"),
        html.Div(id="dec-apply-status", className="mb-3"),

        html.Hr(),
        # ── Filtered Results ─────────────────────────────────────────────────
        html.H4("Filtered Results"),
        html.Div(id="dec-filter-info", className="mb-2"),
        dbc.Row([
            dbc.Col(html.Div(id="dec-axis-info"), width=8),
            dbc.Col(dbc.Checklist(
                id="dec-log-y",
                options=[{"label": " Log Y-axis", "value": "log"}],
                value=[],
            ), width=4),
        ]),
        dcc.Graph(id="dec-scatter"),

        html.Hr(),
        # ── MCDM section (hidden until filters applied) ───────────────────────
        html.Div(id="dec-mcdm-wrapper", style={"display": "none"}, children=[
            html.H4("4. Multi-Criteria Decision Making"),
            html.Div(id="dec-mcdm-info", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Method"),
                    dcc.Dropdown(
                        id="mcdm-method",
                        options=[{"label": "TOPSIS",    "value": "TOPSIS"},
                                 {"label": "PROMETHEE", "value": "PROMETHEE"}],
                        value="TOPSIS", clearable=False,
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Weighting Method"),
                    dbc.RadioItems(
                        id="mcdm-weighting",
                        options=[{"label": "Entropy Weighting", "value": "Entropy"},
                                 {"label": "Manual Weights",    "value": "Manual"}],
                        value="Entropy", inline=True,
                    ),
                ], width=6),
            ], className="mb-3"),

            # Manual weights (hidden when Entropy selected)
            html.Div(id="mcdm-manual-section", style={"display": "none"}, children=[
                html.H6("📊 Criteria Weights — Assign importance (0–5 scale)"),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Balanced",        id="preset-balanced",  color="outline-secondary", size="sm", className="me-2"),
                        dbc.Button("Long-term goal",  id="preset-longterm",  color="outline-secondary", size="sm", className="me-2"),
                        dbc.Button("Short-term goal", id="preset-shortterm", color="outline-secondary", size="sm"),
                    ], className="mb-3"),
                ]),
                dbc.Row(weight_sliders),
            ]),

            html.Div(id="mcdm-weights-table", className="mb-3"),
            dbc.Button("🚀 Run MCDM Analysis", id="mcdm-run-btn",
                       color="success", size="lg", className="mb-4"),
            html.Div(id="mcdm-status", className="mb-2"),

            # Results area (always in DOM; populated by callback)
            html.Div(id="mcdm-results-area"),
            # Download button — always in DOM, shown only after analysis runs
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

server = app.server  # for gunicorn; 'app' is the Dash instance, 'server' is the Flask instance

_SIDEBAR = {
    "backgroundColor": "#f8f9fa",
    "borderRight": "1px solid #e0e0e0",
    "minHeight": "100vh",
    "padding": "1rem",
    "position": "sticky",
    "top": 0,
    "overflowY": "auto",
}

app.layout = dbc.Container(fluid=True, children=[
    dcc.Location(id="url"),

    # ── Global stores ─────────────────────────────────────────────────────────
    dcc.Store(id="store-excl",          data=[]),        # excluded elements (decision)
    dcc.Store(id="bg-elem-store",       data=[]),        # included elements (bandgap)
    dcc.Store(id="store-dec-filters",   data={}),        # applied filter dict
    dcc.Store(id="store-mcdm-results"),                  # serialised MCDM result
    dcc.Store(id="store-sample-seed",   data=42),
    dcc.Store(id="store-bg-csv",        data=None),   # CSV data for download
    dcc.Store(id="store-preset-weights", data={c: 3 for c in CRITERIA_OPTIONS}),

    # ── Downloads (global so they persist across navigation) ─────────────────
    dcc.Download(id="dl-mcdm"),
    dcc.Download(id="dl-csv"),

    dbc.Row([
        # ── Sidebar ───────────────────────────────────────────────────────────
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

        # ── Page content ──────────────────────────────────────────────────────
        dbc.Col(width=10, children=[
            html.Div(id="page-content", className="p-4"),
        ]),
    ]),
])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK: routing
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/bandgap":
        return layout_bandgap()
    if pathname == "/decision":
        return layout_decision()
    return layout_home()


# ═════════════════════════════════════════════════════════════════════════════
# BANDGAP PAGE CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("bg-filter-info", "children"),
    Output("bg-scatter",     "figure"),
    Output("bg-histogram",   "figure"),
    Input("bg-elem-store", "data"),
)
def bg_update_charts(included):
    """Update filter banner + scatter + histogram when elements change."""
    included = included or []
    df_f = filter_by_included_df1(DF1, included)

    # ── filter info banner ────────────────────────────────────────────────────
    if not included:
        info = dbc.Alert("Showing all materials. Click elements on the periodic table to filter.", color="secondary")
    elif df_f.empty:
        info = dbc.Alert("⚠️ No materials contain only these elements.", color="warning")
    else:
        info = dbc.Alert(
            f"🔬 Included: {', '.join(sorted(included))} — "
            f"Showing {len(df_f)} of {len(DF1)} materials ({len(df_f)/len(DF1)*100:.1f}%)",
            color="info",
        )

    if df_f.empty:
        empty = go.Figure()
        empty.update_layout(
            annotations=[dict(text="⚠️ No materials match these elements.",
                              showarrow=False, font=dict(size=14))],
            template="plotly_white",
        )
        return info, empty, empty

    # bandgap column
    bg_col = next(
        (c for c in ["Bandgap", "bandgap", "Band_gap", "band_gap", "Value", "BandGap"]
         if c in df_f.columns), None
    )

    df_agg = (
        df_f.groupby("Name").size().reset_index(name="Count")
        .sort_values("Count", ascending=False).head(9)
    )
    top_names = df_agg["Name"].tolist()

    # ── scatter ───────────────────────────────────────────────────────────────
    if bg_col and top_names:
        top_df = df_f[df_f["Name"].isin(top_names)]
        fig_sc = px.scatter(
            top_df, x=bg_col, y="Name", color="Name",
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

    # ── histogram grid (Plotly subplots – replaces matplotlib) ───────────────
    top9 = top_names[:9]
    if bg_col and top9:
        df_h = df_f[df_f["Name"].isin(top9)].copy()
        df_h = df_h[pd.to_numeric(df_h[bg_col], errors="coerce").notna()]
        df_h[bg_col] = df_h[bg_col].astype(float)

        rows, cols_n = 3, 3
        fig_hist = make_subplots(
            rows=rows, cols=cols_n,
            subplot_titles=[f"{n} (n={len(df_h[df_h['Name']==n])})" for n in top9],
            shared_xaxes=True,
        )
        palette = pick_palette(len(top9))
        for i, name in enumerate(top9):
            r, c = divmod(i, cols_n)
            sub = df_h.loc[df_h["Name"] == name, bg_col].dropna().values
            if sub.size >= 1:
                fig_hist.add_trace(
                    go.Histogram(x=sub, name=name, marker_color=palette[i],
                                 opacity=0.85, showlegend=False),
                    row=r + 1, col=c + 1,
                )
                if sub.size >= 2:
                    fig_hist.add_vline(
                        x=float(np.median(sub)), line_dash="dash", line_color="red",
                        row=r + 1, col=c + 1,
                    )
        fig_hist.update_yaxes(type="log")
        fig_hist.update_layout(
            title="Bandgap Histogram Grid (log y-scale; dashed = median)",
            template="plotly_white", height=700,
        )
    else:
        fig_hist = go.Figure()

    return info, fig_sc, fig_hist


@app.callback(
    Output("bg-temporal-scatter", "figure"),
    Output("bg-sample-table",     "children"),
    Output("store-bg-csv",        "data"),
    Output("bg-download-area",    "children"),
    Input("bg-elem-store",        "data"),
    Input("store-sample-seed",    "data"),
    prevent_initial_call=True,
)
def bg_update_temporal(included, seed):
    """Update temporal scatter, sample table, and CSV store."""
    included = included or []
    df_f1 = filter_by_included_df1(DF1, included)

    df_agg = (
        df_f1.groupby("Name").size().reset_index(name="Count")
        .sort_values("Count", ascending=False).head(9)
    )
    unique_list = df_agg["Name"].dropna().unique().tolist()

    # Use full date range automatically (min to max)
    df_d = DF2

    df2_doi = (
        df_d[df_d["Name"].isin(unique_list)]
        .drop(columns=["index", "Composition", "Confidence", "Publisher"], errors="ignore")
    )
    df2_grp = (
        df2_doi.groupby(["Date", "Name", "Value"])
        .size().reset_index(name="Frequency").reset_index()
    )

    empty_fig = go.Figure()
    empty_fig.update_layout(
        annotations=[dict(text="No data for current selection", showarrow=False)],
        template="plotly_white",
    )

    if df2_grp.empty or "Value" not in df2_grp.columns:
        return empty_fig, html.P("No data."), None, html.Div()

    df2_grp["Value"]     = pd.to_numeric(df2_grp["Value"],     errors="coerce")
    df2_grp["Frequency"] = pd.to_numeric(df2_grp["Frequency"], errors="coerce")
    df2_grp = df2_grp.dropna(subset=["Date", "Value", "Frequency", "Name"])
    df2_plot = df2_grp.sort_values("Date")

    # ── temporal scatter (Plotly bubble) ─────────────────────────────────────
    vmax = float(df2_plot["Value"].max()) if not df2_plot.empty else 4.0
    fig_t = px.scatter(
        df2_plot, x="Date", y="Value", color="Name",
        size="Frequency", size_max=30,
        labels={"Value": "Bandgap Energy (eV)", "Date": "Publication Date"},
        title="Bandgap Trends Over Time",
        template="plotly_white", height=500,
    )
    for lo, hi, col, lbl in [
        (0, 1.6, "rgba(255,255,0,0.10)",  "Infrared (0–1.6 eV)"),
        (1.6, 3.26, "rgba(0,200,0,0.10)", "Visible (1.6–3.26 eV)"),
        (3.26, max(vmax, 4.0), "rgba(255,0,0,0.10)", "Ultraviolet (3.26+ eV)"),
    ]:
        fig_t.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0,
                        annotation_text=lbl, annotation_position="top left",
                        annotation_font_size=10)
    fig_t.update_layout(hovermode="closest")

    # ── sample table ──────────────────────────────────────────────────────────
    n = min(10, len(df2_doi))
    if n == 0:
        table_el = html.P("No records to display.")
    else:
        sample = df2_doi.sample(n=n, random_state=seed).reset_index(drop=True)
        # Format Date column as "Mon YYYY" (e.g. "Jan 2023") and drop any Year column
        if "Date" in sample.columns:
            sample["Date"] = pd.to_datetime(sample["Date"], errors="coerce").dt.strftime("%b %Y")
        sample = sample.drop(columns=[c for c in sample.columns if c.lower() == "year"], errors="ignore")
        table_el = dbc.Table.from_dataframe(
            sample, striped=True, bordered=True,
            hover=True, responsive=True, className="small",
        )

    # ── store CSV data + show download button ─────────────────────────────────
    csv_data = df2_doi.to_csv(index=False) if not df2_doi.empty else None
    dl_btn = html.Div(
        dbc.Button("⬇️ Download filtered data as CSV", id="bg-dl-btn",
                   color="secondary", size="sm", className="mt-2"),
    ) if not df2_doi.empty else html.Div()

    return fig_t, table_el, csv_data, dl_btn


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
    # Only proceed if the button was actually clicked and we have data
    if not n_clicks or not csv_str:
        return no_update
    
    # Add a check to ensure this is a real button click
    if not ctx.triggered_id == "bg-dl-btn":
        return no_update
    
    return dcc.send_string(csv_str, "bandgap-filtered.csv", type="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# DECISION PAGE CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

# ── element exclusion preview (live — driven by store-excl) ──────────────────
@app.callback(
    Output("dec-excl-info", "children"),
    Input("store-excl", "data"),
)
def dec_excl_preview(excluded):
    excluded = excluded or []
    if not excluded:
        return dbc.Alert(
            "Click elements on the table above to exclude them from analysis.",
            color="secondary", className="py-2",
        )
    preview = filter_by_excluded(DF1, excluded)
    removed = len(DF1) - len(preview)
    return dbc.Alert(
        f"🚫 Excluded: {', '.join(sorted(excluded))} — "
        f"removes {removed} materials ({removed/len(DF1)*100:.1f}%) — "
        f"{len(preview)} remaining.",
        color="warning", className="py-2",
    )


# ── show/hide initial-filter sliders ─────────────────────────────────────────
@app.callback(
    [Output(f"if-{FILTER_SID[f]}-wrap", "style") for f in FILTER_OPTIONS],
    Input("dec-filter-select", "value"),
)
def show_initial_filter_slider(selected):
    return [
        {"display": "block"} if f == selected else {"display": "none"}
        for f in FILTER_OPTIONS
    ]


# ── show/hide extra-filter sliders ───────────────────────────────────────────
@app.callback(
    [Output(f"ef-{FILTER_SID[f]}-wrap", "style") for f in FILTER_OPTIONS],
    Input("dec-extra-select", "value"),
)
def show_extra_filter_sliders(selected_list):
    selected_list = selected_list or []
    return [
        {"display": "block"} if f in selected_list else {"display": "none"}
        for f in FILTER_OPTIONS
    ]


# ── keep extra-select options consistent with initial select ──────────────────
@app.callback(
    Output("dec-extra-select", "options"),
    Output("dec-extra-select", "value"),
    Input("dec-filter-select", "value"),
    State("dec-extra-select", "value"),
)
def sync_extra_options(initial, extra_vals):
    opts = [{"label": f, "value": f} for f in FILTER_OPTIONS if f != initial]
    cleaned = [v for v in (extra_vals or []) if v != initial]
    return opts, cleaned


# ── Apply Filters ─────────────────────────────────────────────────────────────
@app.callback(
    Output("store-dec-filters",   "data"),
    Output("dec-apply-status",    "children"),
    Input("dec-apply-btn",        "n_clicks"),
    # Bandgap
    State("dec-bg-min",           "value"),
    State("dec-bg-max",           "value"),
    # Initial filter
    State("dec-filter-select",    "value"),
    *[State(f"if-{FILTER_SID[f]}", "value") for f in FILTER_OPTIONS],
    # Extra filters
    State("dec-extra-select",     "value"),
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

    filters = {"Bandgap": [bg_min, bg_max]}

    if init_filter:
        idx = FILTER_OPTIONS.index(init_filter)
        val = if_vals[idx]
        if val:
            filters[init_filter] = val

    for fname in (extra_sel or []):
        idx = FILTER_OPTIONS.index(fname)
        val = ef_vals[idx]
        if val:
            filters[fname] = val

    n_active = len(filters)
    msg = dbc.Alert(f"✅ {n_active} filter(s) applied successfully!", color="success", className="py-2")
    return filters, msg


# ── Scatter plot (filtered results) ──────────────────────────────────────────
@app.callback(
    Output("dec-scatter",     "figure"),
    Output("dec-filter-info", "children"),
    Output("dec-axis-info",   "children"),
    Output("dec-mcdm-wrapper","style"),
    Output("dec-mcdm-info",   "children"),
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
            annotations=[dict(text="Apply filters to see results", showarrow=False, font=dict(size=14))],
            template="plotly_white",
        )
        return fig, dbc.Alert("📈 No filters applied yet.", color="secondary"), \
               html.Div(), {"display": "none"}, html.Div()

    df_f = filter_df(df_ex, {k: v for k, v in filters.items() if k in DF1.columns})

    x_col = "Bandgap"
    filter_keys = [k for k in filters if k != "Bandgap"]
    y_col = filter_keys[0] if filter_keys else (x_col)

    if df_f.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="⚠️ No materials match the current filters", showarrow=False)],
            template="plotly_white",
        )
        info = dbc.Alert("⚠️ No materials match the current filters.", color="warning")
        return fig, info, html.Div(), {"display": "none"}, html.Div()

    fig = professional_scatter(
        df_f, x_col, y_col, f"{x_col} vs {y_col}", x_col, y_col, log_y=log_y
    )

    filter_summary = ", ".join(filters.keys())
    info = dbc.Alert(
        f"📊 Showing {len(df_f)} materials | Filters: {filter_summary} | "
        f"Available after exclusion: {len(df_ex)}",
        color="info", className="py-2",
    )
    axis_info = html.Span([
        html.Strong("X-axis: "), x_col, "  |  ",
        html.Strong("Y-axis: "), y_col,
    ])
    mcdm_info = dbc.Alert(
        f"Analyze the {len(df_f)} filtered materials using TOPSIS or PROMETHEE.",
        color="primary", className="py-2",
    )
    return fig, info, axis_info, {"display": "block"}, mcdm_info


# ── toggle manual weights section ────────────────────────────────────────────
@app.callback(
    Output("mcdm-manual-section", "style"),
    Input("mcdm-weighting", "value"),
)
def toggle_manual(weighting):
    return {"display": "block"} if weighting == "Manual" else {"display": "none"}


# ── preset buttons → update weight sliders ────────────────────────────────────
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


# ── show weights table ────────────────────────────────────────────────────────
@app.callback(
    Output("mcdm-weights-table", "children"),
    Input("mcdm-weighting", "value"),
    Input("store-dec-filters", "data"),
    Input("store-excl", "data"),
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
        w_arr = np.array([raw_w[crit_names.index(k)] for k in avail], dtype=float)
        s = w_arr.sum()
        w_norm = w_arr / s if s > 0 else np.ones(len(avail)) / len(avail)
    else:
        w_norm = None  # computed at run time

    if w_norm is not None:
        wdf = pd.DataFrame({
            "Criterion": list(avail.keys()),
            "Weight":    [f"{w:.2%}" for w in w_norm],
            "Direction": ["Maximize" if d == 1 else "Minimize" for d in avail.values()],
        }).sort_values("Weight", ascending=False).reset_index(drop=True)
        wdf.index = wdf.index + 1
        return html.Div([
            html.H6("Criteria Weights"),
            dbc.Table.from_dataframe(wdf, striped=True, bordered=True, hover=True,
                                     responsive=True, className="small"),
        ])
    return dbc.Alert("✅ Weights will be computed automatically via Entropy method.", color="info")


# ── Run MCDM ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("mcdm-status",        "children"),
    Output("mcdm-results-area",  "children"),
    Output("store-mcdm-results", "data"),
    Output("mcdm-dl-wrapper",    "style"),
    Input("mcdm-run-btn",       "n_clicks"),
    State("store-dec-filters",  "data"),
    State("store-excl",         "data"),
    State("mcdm-method",        "value"),
    State("mcdm-weighting",     "value"),
    *[State(f"weight-{CRIT_SID[c]}", "value") for c in CRITERIA_OPTIONS],
    prevent_initial_call=True,
)
def run_mcdm(_, filters, excluded, method, weighting, *raw_w):
    if not filters:
        return dbc.Alert("❌ Apply filters first.", color="danger"), html.Div(), no_update, {"display": "none"}

    df_ex = filter_by_excluded(DF1, excluded or [])
    df_f  = filter_df(df_ex, {k: v for k, v in filters.items() if k in DF1.columns})
    avail = {k: v for k, v in CRITERIA_OPTIONS.items() if k in df_f.columns}

    if not avail:
        return dbc.Alert("❌ No criteria columns found.", color="danger"), html.Div(), no_update, {"display": "none"}
    if df_f.empty:
        return dbc.Alert("❌ No materials in filtered set.", color="danger"), html.Div(), no_update, {"display": "none"}

    matrix = df_f[list(avail.keys())].values
    types  = np.array(list(avail.values()))

    if np.isnan(matrix).any():
        return dbc.Alert(f"❌ {np.isnan(matrix).sum()} missing values in criteria.", color="danger"), \
               html.Div(), no_update, {"display": "none"}
    if np.any(matrix < 0) and weighting == "Entropy":
        return dbc.Alert("❌ Entropy weighting requires non-negative values.", color="danger"), \
               html.Div(), no_update

    # Weights
    if weighting == "Entropy":
        try:
            weights = entropy_weights(matrix)
        except Exception as e:
            weights = np.ones(len(avail)) / len(avail)
    else:
        crit_names = list(CRITERIA_OPTIONS.keys())
        raw = np.array([raw_w[crit_names.index(k)] for k in avail], dtype=float)
        s   = raw.sum()
        weights = raw / s if s > 0 else np.ones(len(avail)) / len(avail)

    if not np.isclose(weights.sum(), 1.0):
        weights /= weights.sum()

    # Run
    try:
        if method == "TOPSIS":
            scores = TOPSIS()(matrix, weights, types)
            results = pd.DataFrame({
                "Material":     df_f["Name"].values,
                "Bandgap (eV)": df_f["Bandgap"].values,
                "DOI":          df_f["DOI"].values if "DOI" in df_f.columns else [""] * len(df_f),
                "Score":        scores,
            }).sort_values("Score", ascending=False).reset_index(drop=True)
            score_col = "Score"
        else:
            flows = PROMETHEE_II("usual")(matrix, weights, types)
            results = pd.DataFrame({
                "Material":     df_f["Name"].values,
                "Bandgap (eV)": df_f["Bandgap"].values,
                "Net Flow":     flows,
            }).sort_values("Net Flow", ascending=False).reset_index(drop=True)
            score_col = "Net Flow"
    except Exception as e:
        return dbc.Alert(f"❌ Error running {method}: {e}", color="danger"), html.Div(), no_update, {"display": "none"}

    results.index      = results.index + 1
    results.index.name = "Rank"

    if np.isnan(results[score_col].values).any():
        return dbc.Alert(f"❌ {method} returned NaN values.", color="danger"), html.Div(), no_update, {"display": "none"}

    # Weights table
    wdf = pd.DataFrame({
        "Criterion": list(avail.keys()),
        "Weight":    weights,
        "Direction": ["Maximize" if d == 1 else "Minimize" for d in avail.values()],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    wdf.index      = wdf.index + 1
    wdf.index.name = "Rank"

    # Top 3
    top3 = results.drop_duplicates(subset=["Material"]).head(3)
    top3_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P(f"Rank #{top3.index[i]}", className="text-muted mb-1 small"),
            html.H5(top3.iloc[i]["Material"], className="mb-0"),
        ])), width=4)
        for i in range(len(top3))
    ])

    # Results table
    display_cols = ["Material", "Bandgap (eV)"] + (["DOI"] if "DOI" in results.columns else []) + [score_col]
    top100 = results[display_cols].head(100).reset_index()
    results_table = dbc.Table.from_dataframe(
        top100.rename(columns={"index": "Rank"}),
        striped=True, bordered=True, hover=True, responsive=True, className="small",
    )

    # Serialize for download callback
    results_store = {
        "method":    method,
        "materials": results["Material"].tolist(),
        "scores":    results[score_col].tolist(),
        "bandgaps":  results["Bandgap (eV)"].tolist(),
        "filters":   filters,
        "excluded":  excluded or [],
        "criteria":  list(avail.keys()),
        "weights":   weights.tolist(),
        "score_col": score_col,
        "doi":       results["DOI"].tolist() if "DOI" in results.columns else [],
    }

    results_ui = html.Div([
        html.H5("MCDM Results"),
        html.P(f"Showing top 100 results (out of {len(results)} total)"),
        results_table,
        html.H5("🏆 Top Materials", className="mt-3"),
        top3_cards,
    ])

    status = dbc.Alert(f"✅ {method} analysis complete!", color="success", className="py-2")
    return status, results_ui, results_store, {"display": "block"}


# ── MCDM Download ─────────────────────────────────────────────────────────────
@app.callback(
    Output("dl-mcdm", "data"),
    Input("mcdm-dl-btn", "n_clicks"),
    State("store-mcdm-results", "data"),
    prevent_initial_call=True,
)
def download_mcdm(_, store):
    if not store:
        return no_update

    # Reconstruct filtered DataFrame
    filters  = store["filters"]
    excluded = store.get("excluded", [])
    df_ex    = filter_by_excluded(DF1, excluded)
    df_f     = filter_df(df_ex, {k: v for k, v in filters.items() if k in DF1.columns})

    # Reconstruct results DataFrame
    score_col = store["score_col"]
    rdata = {"Material": store["materials"], "Bandgap (eV)": store["bandgaps"]}
    if store["doi"]:
        rdata["DOI"] = store["doi"]
    rdata[score_col] = store["scores"]
    results_df = pd.DataFrame(rdata)
    results_df.index      = results_df.index + 1
    results_df.index.name = "Rank"

    # Weights DataFrame
    criteria  = store["criteria"]
    weights   = store["weights"]
    wdf = pd.DataFrame({
        "Criterion": criteria,
        "Weight":    weights,
        "Direction": ["Maximize" if CRITERIA_OPTIONS.get(c, -1) == 1 else "Minimize" for c in criteria],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    wdf.index      = wdf.index + 1
    wdf.index.name = "Rank"

    excel_bytes = build_excel(df_f, results_df, wdf, filters)
    return dcc.send_bytes(excel_bytes, f"mcdm_analysis_{store['method']}.xlsx")



# ═════════════════════════════════════════════════════════════════════════════
# PERIODIC TABLE CALLBACKS  (shared helpers, prefix distinguishes bg vs dec)
# ═════════════════════════════════════════════════════════════════════════════

# ── Bandgap: toggle element in bg-elem-store ──────────────────────────────────
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
    sym      = PT_BY_Z[triggered["Z"]]["sym"]
    current  = current or []
    if sym in current:
        current = [s for s in current if s != sym]
    else:
        current = current + [sym]
    return current


# ── Bandgap: update cell styles from bg-elem-store ───────────────────────────
@app.callback(
    [Output({"type": "bg-elem", "Z": e["Z"]}, "style") for e in PT_ELEMENTS],
    Input("bg-elem-store", "data"),
)
def bg_update_pt_styles(active_syms):
    active_syms = active_syms or []
    return [_pt_cell_style(e, active_syms, "include") for e in PT_ELEMENTS]


# ── Decision: toggle element in store-excl ────────────────────────────────────
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
    if sym in current:
        current = [s for s in current if s != sym]
    else:
        current = current + [sym]
    return current


# ── Decision: update cell styles from store-excl ─────────────────────────────
@app.callback(
    [Output({"type": "dec-elem", "Z": e["Z"]}, "style") for e in PT_ELEMENTS],
    Input("store-excl", "data"),
)
def dec_update_pt_styles(active_syms):
    active_syms = active_syms or []
    return [_pt_cell_style(e, active_syms, "exclude") for e in PT_ELEMENTS]

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)