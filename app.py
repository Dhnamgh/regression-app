import io
import math
import os
import json
import textwrap
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import streamlit.components.v1 as components

# ===== LOGIN =====
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 Đăng nhập")
    password = st.text_input("Nhập mật khẩu", type="password")

    if st.button("Đăng nhập"):
        if password == st.secrets.get("APP_PASSWORD", "123456"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Sai mật khẩu")

    return False

if not check_password():
    st.stop()

# ===== GOOGLE GEMINI AI CONFIG =====
import google.generativeai as genai

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multicomp import pairwise_tukeyhsd

try:
    from statsmodels.stats.diagnostic import lilliefors, het_breuschpagan
except ImportError:
    lilliefors = None
    het_breuschpagan = None

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="Data Analysis in Health Sciences",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CSS Giao diện: To, rõ, đậm
# =========================================================
st.markdown(
    """
<style>
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0B3A66 0%, #0A2D4E 100%);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div{
  color: #ffffff !important;
}

section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stDownloadButton button{
  width: 100% !important;
  background: rgba(255,255,255,0.10) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.30) !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
  font-size: 17px !important;
  padding: 10px 12px !important;
}
section[data-testid="stSidebar"] .stButton button:hover,
section[data-testid="stSidebar"] .stDownloadButton button:hover{
  background: rgba(255,255,255,0.18) !important;
  border-color: rgba(255,255,255,0.45) !important;
}

section[data-testid="stSidebar"] details summary{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  font-weight: 800 !important;
  font-size: 18px !important;
}

.header-banner{
  background: linear-gradient(90deg, #0B3A66 0%, #0A2D4E 100%);
  border-radius: 18px;
  padding: 18px 22px;
  color: #fff;
}
.header-banner h1{
  margin: 0;
  padding: 0;
  font-size: 34px;
  line-height: 1.1;
  font-weight: 900;
}
.header-banner p{
  margin: 8px 0 0 0;
  opacity: 0.95;
  font-size: 17px;
  font-weight: 600;
}

h2 {
  font-size: 28px !important;
  font-weight: 900 !important;
  color: #0B3A66 !important;
}
h3 {
  font-size: 23px !important;
  font-weight: 800 !important;
  color: #0B3A66 !important;
  margin-top: 15px !important;
}
h4 {
  font-size: 20px !important;
  font-weight: 800 !important;
  color: #0f172a !important;
  margin-top: 12px !important;
}

div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
  font-size: 19px !important;
  font-weight: 800 !important;
  color: #0f172a !important;
  line-height: 1.3 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label,
div[data-testid="stRadio"] div[role="radiogroup"] label p,
div[data-testid="stRadio"] div[role="radiogroup"] span {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] span {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

.stMarkdown p {
  font-size: 17px !important;
  font-weight: 600 !important;
  color: #0f172a !important;
}
.stCaption, [data-testid="stCaptionContainer"] p {
  font-size: 16px !important;
  font-weight: 700 !important;
  color: #334155 !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
  border-width: 1.5px !important;
}

div[data-baseweb="select"] * {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

.analysis-table-wrap{
  width: 100%;
  overflow-x: auto;
  margin: 0.6rem 0 1.3rem 0;
}
.analysis-table{
  width: 100%;
  border-collapse: collapse;
  font-size: 20px;
  color: #0f172a;
  background: #ffffff;
}
.analysis-table th{
  background: #f1f5f9;
  color: #0f172a;
  font-weight: 900;
  border: 1.5px solid #cbd5e1;
  padding: 12px 15px;
  text-align: left;
  line-height: 1.25;
}
.analysis-table td{
  border: 1.2px solid #e2e8f0;
  padding: 12px 15px;
  color: #0f172a;
  font-weight: 500;
  line-height: 1.35;
}
.analysis-table tbody tr:nth-child(even){
  background: #f8fafc;
}
</style>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="header-banner">
  <h1>Data Analysis in Health Sciences</h1>
  <p>Regression, categorical analysis, quantitative tests, diagnostics, sample size, probability & AI reasoning.</p>
</div>
""",
    unsafe_allow_html=True
)
st.write("")

# =========================================================
# Thuật toán 2 bước tính phân vị giáo trình (Q1 = 21 khi n=20)
# =========================================================
def compute_percentile_textbook(arr: np.ndarray, p: float) -> float:
    s = np.sort(np.asarray(arr, dtype=float))
    n = len(s)
    k = n * (p / 100.0)
    if abs(k - round(k)) < 1e-9:
        idx = int(round(k))
        return float((s[idx - 1] + s[idx]) / 2.0)
    else:
        idx = int(math.ceil(k))
        return float(s[idx - 1])

def parse_numeric_text(txt: str) -> np.ndarray:
    if not txt or not txt.strip():
        return np.array([], dtype=float)
    normalized = txt.replace(";", " ").replace(",", " ")
    parts = [p for p in normalized.split() if p.strip()]
    vals = pd.to_numeric(pd.Series(parts), errors="coerce").dropna()
    return vals.values

# =========================================================
# Quy tắc làm tròn số thông minh
# =========================================================
def smart_round_val(val, min_dec: int = 3) -> str:
    if val is None or pd.isna(val) or val == "":
        return ""
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, str):
        return val
    try:
        f = float(val)
    except Exception:
        return str(val)
    if np.isnan(f) or np.isinf(f):
        return ""
    if f == 0.0:
        return f"{0:.{min_dec}f}"

    abs_f = abs(f)
    if abs_f >= 1.0 or round(abs_f, min_dec) > 0:
        return f"{f:.{min_dec}f}"

    dec = min_dec + 1
    while dec <= 8 and round(abs_f, dec) == 0:
        dec += 1
    return f"{f:.{dec}f}"

def format_p_value(p: Optional[float]) -> str:
    if p is None:
        return ""
    try:
        p = float(p)
    except Exception:
        return ""
    if np.isnan(p) or np.isinf(p):
        return ""
    if p < 0.001:
        return "< 0.001"
    return smart_round_val(p, min_dec=3)

def compact_numeric_df(df, decimals: int = 3):
    out = df.copy()
    if isinstance(out, pd.Series):
        return out.map(lambda v: smart_round_val(v, min_dec=decimals))
    for c in out.columns:
        out[c] = out[c].map(lambda v: smart_round_val(v, min_dec=decimals))
    return out

def clean_cell(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        if np.isnan(x):
            return ""
    except Exception:
        pass
    return x

def clean_term_name(s: str) -> str:
    if isinstance(s, str) and s.startswith('Q("') and s.endswith('")'):
        return s[3:-2]
    return s

# =========================================================
# Xuất dữ liệu & Nút chép Excel
# =========================================================
def df_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, _df in sheets.items():
            _df.to_excel(writer, index=False, sheet_name=sheet[:31])
    bio.seek(0)
    return bio.getvalue()

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 200) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
    bio.seek(0)
    return bio.getvalue()

def df_to_png_bytes(df: pd.DataFrame, title: str = "", dpi: int = 200) -> bytes:
    df_plot = df.copy().fillna("-").astype(str)
    n_rows, n_cols = df_plot.shape
    width_limit = 16

    def get_line_count(text, width):
        if not text or text == "-": return 1
        lines = textwrap.wrap(str(text), width=width)
        return len(lines) if len(lines) > 0 else 1

    header_max_lines = max([get_line_count(col, width_limit) for col in df_plot.columns])
    row_line_counts = []
    for _, row in df_plot.iterrows():
        max_lines_in_row = max([get_line_count(val, width_limit) for val in row])
        row_line_counts.append(max_lines_in_row)

    line_unit_height = 0.35
    padding = 0.5
    h_height = (header_max_lines * line_unit_height) + padding
    b_height = sum([(lc * line_unit_height) + padding for lc in row_line_counts])

    fig_width = max(12, n_cols * 2.3)
    fig_height = h_height + b_height + 2.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    wrapped_headers = [textwrap.fill(str(col), width=width_limit) for col in df_plot.columns]
    wrapped_data = []
    for _, row in df_plot.iterrows():
        wrapped_data.append([textwrap.fill(str(val), width=width_limit) for val in row])

    table = ax.table(
        cellText=wrapped_data,
        colLabels=wrapped_headers,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)

    for i in range(n_cols):
        table[0, i].set_height(h_height / fig_height)
        for j in range(n_rows):
            cell_h = (row_line_counts[j] * line_unit_height + padding) / fig_height
            table[j+1, i].set_height(cell_h)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#94a3b8')
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e2e8f0')
        else:
            cell.set_facecolor('white')

    if title:
        plt.title(title, fontsize=18, pad=40, weight='bold')

    bio = io.BytesIO()
    plt.savefig(bio, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    return bio.getvalue()

def render_copy_excel_button(df: pd.DataFrame, base_name: str):
    tsv_data = df.to_csv(sep="\t", index=False)
    json_text = json.dumps(tsv_data)
    btn_id = f"btn_cp_{base_name}_{abs(hash(base_name)) % 100000}"
    html_code = f"""
    <div style="display:flex; justify-content:center; align-items:center; height:100%;">
        <button id="{btn_id}" style="
            width: 100%;
            height: 38px;
            background: #f8fafc;
            color: #0f172a;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: all 0.2s ease;
        ">
            📋 Chép Excel
        </button>
    </div>
    <script>
    document.getElementById("{btn_id}").addEventListener("click", async () => {{
        const text = {json_text};
        try {{
            await navigator.clipboard.writeText(text);
            const b = document.getElementById("{btn_id}");
            b.innerText = "✅ Đã chép!";
            b.style.background = "#dcfce7";
            b.style.borderColor = "#86efac";
            b.style.color = "#166534";
            setTimeout(() => {{
                b.innerText = "📋 Chép Excel";
                b.style.background = "#f8fafc";
                b.style.borderColor = "#cbd5e1";
                b.style.color = "#0f172a";
            }}, 2000);
        }} catch (e) {{
            const ta = document.createElement("textarea");
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            alert("Đã chép vào clipboard! Nhấn Ctrl+V vào Excel.");
        }}
    }});
    </script>
    """
    components.html(html_code, height=45)

def download_table_block(df: pd.DataFrame, base_name: str, title: str = ""):
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.download_button(
            "Download Excel",
            data=df_to_excel_bytes({base_name: df}),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"dl_excel_{base_name}_{abs(hash(base_name)) % 100000}",
            on_click="ignore"
        )
    with c2:
        st.download_button(
            "Download PNG",
            data=df_to_png_bytes(df, title=title),
            file_name=f"{base_name}.png",
            mime="image/png",
            use_container_width=True,
            key=f"dl_png_{base_name}_{abs(hash(base_name)) % 100000}",
            on_click="ignore"
        )
    with c3:
        render_copy_excel_button(df, base_name)

def download_figure_block(fig: plt.Figure, base_name: str):
    st.download_button(
        "Download PNG",
        data=fig_to_png_bytes(fig),
        file_name=f"{base_name}.png",
        mime="image/png",
        use_container_width=False,
        on_click="ignore"
    )

def show_table(df: pd.DataFrame, title: str):
    st.markdown(f"### {title}")
    display_df = df.copy().fillna("")
    html = display_df.to_html(index=False, escape=True, classes="analysis-table")
    st.markdown(f'<div class="analysis-table-wrap">{html}</div>', unsafe_allow_html=True)

# =========================================================
# File loading & Storage
# =========================================================
def load_uploaded_file(uploaded) -> pd.DataFrame:
    if uploaded is None:
        raise ValueError("No file uploaded.")
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError("Unsupported file type. Please upload CSV or XLSX.")

LOGISTIC_KEY = "df_logistic"
LINEAR_KEY = "df_linear"

if LOGISTIC_KEY not in st.session_state:
    st.session_state[LOGISTIC_KEY] = None
if "df_logistic_name" not in st.session_state:
    st.session_state["df_logistic_name"] = ""

if LINEAR_KEY not in st.session_state:
    st.session_state[LINEAR_KEY] = None
if "df_linear_name" not in st.session_state:
    st.session_state["df_linear_name"] = ""

def data_input_panel(template_df: pd.DataFrame, template_name: str, store_key: str, store_name_key: str, help_text: str = ""):
    with st.expander("Data input (template + upload)", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.download_button(
                "Download Excel template",
                data=df_to_excel_bytes({template_name: template_df}),
                file_name=f"{template_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            if help_text:
                st.caption(help_text)

        with c2:
            up = st.file_uploader(
                "Upload CSV/XLSX",
                type=["csv", "xlsx", "xls"],
                key=f"uploader_{template_name}"
            )
            if up is not None:
                df = load_uploaded_file(up)
                st.session_state[store_key] = df
                st.session_state[store_name_key] = up.name
                st.success(f"Loaded: {up.name} • shape={df.shape}")

    df = st.session_state.get(store_key)
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.caption(f"Current dataset: {st.session_state.get(store_name_key,'')} • {df.shape}")
        st.dataframe(df.head(30), use_container_width=True)

def require_df(store_key: str) -> pd.DataFrame:
    df = st.session_state.get(store_key)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("No dataset loaded on this page yet. Use the Data input panel above.")
        raise RuntimeError("No dataset")
    return df

# =========================================================
# Bảng liên định r x c & Đo lường 2x2
# =========================================================
def contingency_editor(key: str, default_rows: List[str], default_cols: List[str], default_counts: np.ndarray):
    ss_key = f"ct_{key}"
    if ss_key not in st.session_state:
        init = np.asarray(default_counts, dtype=int)
        init = np.clip(init, 0, None)
        df0 = pd.DataFrame(init, columns=default_cols)
        df0.insert(0, "Group", default_rows)
        st.session_state[ss_key] = df0

    df = st.session_state[ss_key].copy()

    st.markdown("#### Labels")
    with st.expander("Rename row/column labels", expanded=False):
        new_rows = []
        for i, old in enumerate(df["Group"].astype(str).tolist()):
            new_rows.append(st.text_input(f"Row {i+1} label", value=old, key=f"{key}_rowlbl_{i}"))
        df["Group"] = new_rows

        cat_cols = [c for c in df.columns if c != "Group"]
        new_cols = []
        for j, old in enumerate(cat_cols):
            new_cols.append(st.text_input(f"Column {j+1} label", value=str(old), key=f"{key}_collbl_{j}"))

        cleaned, seen = [], set()
        for name in new_cols:
            nm = name.strip() if name.strip() else "Category"
            base = nm
            kdup = 1
            while nm in seen:
                kdup += 1
                nm = f"{base}_{kdup}"
            seen.add(nm)
            cleaned.append(nm)

        rename_map = {old: new for old, new in zip(cat_cols, cleaned)}
        df = df.rename(columns=rename_map)

    cat_cols = [c for c in df.columns if c != "Group"]
    for c in cat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(0).astype(int).clip(lower=0)

    st.markdown("#### Observed counts (edit cells)")
    edited = st.data_editor(
        df,
        key=f"{key}_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={"Group": st.column_config.TextColumn("Group")} | {
            c: st.column_config.NumberColumn(c, min_value=0, step=1, format="%d") for c in cat_cols
        }
    )
    edited = edited.copy()
    edited["Group"] = edited["Group"].astype(str)
    for c in cat_cols:
        edited[c] = pd.to_numeric(edited[c], errors="coerce").fillna(0).round(0).astype(int).clip(lower=0)

    st.session_state[ss_key] = edited
    observed_df = edited[cat_cols].copy()

    counts_df = edited.copy()
    counts_df["Total"] = observed_df.sum(axis=1).astype(int)

    total_row = {"Group": "Total"}
    for c in cat_cols:
        total_row[c] = int(observed_df[c].sum())
    total_row["Total"] = int(observed_df.values.sum())
    counts_df = pd.concat([counts_df, pd.DataFrame([total_row])], ignore_index=True)

    return counts_df, observed_df

def rc_contingency_ui(key: str, default_r: int = 2, default_c: int = 2):
    st.markdown("### Table size")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        r = st.number_input("Rows (r)", min_value=2, max_value=20, value=default_r, step=1, key=f"{key}_r")
    with c2:
        c = st.number_input("Columns (c)", min_value=2, max_value=20, value=default_c, step=1, key=f"{key}_c")
    with c3:
        if st.button("Apply size (reset table)", key=f"{key}_apply_rc", use_container_width=True):
            rows = [f"Group {i+1}" for i in range(int(r))]
            cols = [f"Category {j+1}" for j in range(int(c))]
            init = np.ones((int(r), int(c)), dtype=int)
            df0 = pd.DataFrame(init, columns=cols)
            df0.insert(0, "Group", rows)
            st.session_state[f"ct_{key}"] = df0
            st.rerun()

    return contingency_editor(
        key=key,
        default_rows=[f"Group {i+1}" for i in range(default_r)],
        default_cols=[f"Category {j+1}" for j in range(default_c)],
        default_counts=np.ones((default_r, default_c), dtype=int),
    )

def get_observed_matrix(observed_df: pd.DataFrame) -> np.ndarray:
    if observed_df is None or observed_df.empty:
        raise ValueError("Observed table is empty.")
    df = observed_df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0).round(0).astype(int)
    df[df < 0] = 0
    mat = df.to_numpy(dtype=int)
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
        raise ValueError("Contingency table must be r×c with r>=2 and c>=2.")
    if mat.sum() <= 0:
        raise ValueError("Total count must be > 0.")
    return mat

def require_2x2(observed_df: pd.DataFrame) -> np.ndarray:
    mat = get_observed_matrix(observed_df)
    if mat.shape != (2, 2):
        raise ValueError("Fisher's Exact Test requires a 2×2 table.")
    return mat

def _safe_div(a, b):
    return np.nan if b == 0 else a / b

def wilson_ci(x, n, alpha=0.05) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha/2)
    phat = x / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def log_ci_ratio(est, se, alpha=0.05) -> Tuple[float, float]:
    z = stats.norm.ppf(1 - alpha/2)
    lo = math.exp(math.log(est) - z*se)
    hi = math.exp(math.log(est) + z*se)
    return lo, hi

def two_by_two_measures(obs2x2: np.ndarray, alpha=0.05) -> pd.DataFrame:
    a, b, c, d = int(obs2x2[0,0]), int(obs2x2[0,1]), int(obs2x2[1,0]), int(obs2x2[1,1])
    cc = 0.5 if min(a,b,c,d) == 0 else 0.0
    a2, b2, c2, d2 = a+cc, b+cc, c+cc, d+cc

    OR = (a2*d2) / (b2*c2)
    se_log_or = math.sqrt(1/a2 + 1/b2 + 1/c2 + 1/d2)
    or_lo, or_hi = log_ci_ratio(OR, se_log_or, alpha)

    risk_e = _safe_div(a2, (a2+b2))
    risk_u = _safe_div(c2, (c2+d2))
    RR = _safe_div(risk_e, risk_u)
    se_log_rr = math.sqrt((1/a2) - (1/(a2+b2)) + (1/c2) - (1/(c2+d2)))
    rr_lo, rr_hi = log_ci_ratio(RR, se_log_rr, alpha)

    VE = 1 - RR
    ve_lo, ve_hi = 1 - rr_hi, 1 - rr_lo

    TP, FP, FN, TN = a, b, c, d
    sens = _safe_div(TP, TP+FN)
    spec = _safe_div(TN, TN+FP)
    fpr  = _safe_div(FP, FP+TN)
    fnr  = _safe_div(FN, FN+TP)
    ppv  = _safe_div(TP, TP+FP)
    npv  = _safe_div(TN, TN+FN)
    lr_p = _safe_div(sens, 1-spec) if (1-spec) not in [0, np.nan] else np.nan
    lr_n = _safe_div(1-sens, spec) if spec not in [0, np.nan] else np.nan

    sens_lo, sens_hi = wilson_ci(TP, TP+FN, alpha) if (TP+FN)>0 else (np.nan, np.nan)
    spec_lo, spec_hi = wilson_ci(TN, TN+FP, alpha) if (TN+FP)>0 else (np.nan, np.nan)
    ppv_lo, ppv_hi   = wilson_ci(TP, TP+FP, alpha) if (TP+FP)>0 else (np.nan, np.nan)
    npv_lo, npv_hi   = wilson_ci(TN, TN+FN, alpha) if (TN+FN)>0 else (np.nan, np.nan)

    rows = [
        ("Odds Ratio (OR)", OR, or_lo, or_hi),
        ("Risk Ratio (RR)", RR, rr_lo, rr_hi),
        ("Vaccine Effectiveness (VE = 1−RR)", VE, ve_lo, ve_hi),
        ("Sensitivity", sens, sens_lo, sens_hi),
        ("Specificity", spec, spec_lo, spec_hi),
        ("False Positive Rate", fpr, np.nan, np.nan),
        ("False Negative Rate", fnr, np.nan, np.nan),
        ("PPV", ppv, ppv_lo, ppv_hi),
        ("NPV", npv, npv_lo, npv_hi),
        ("LR+", lr_p, np.nan, np.nan),
        ("LR-", lr_n, np.nan, np.nan),
    ]
    df = pd.DataFrame(rows, columns=["Measure", "Estimate", "CI 2.5%", "CI 97.5%"])
    return compact_numeric_df(df, decimals=3)

# =========================================================
# Thuật toán Hồi quy Logistic
# =========================================================
def hosmer_lemeshow_table(y_true, y_prob, g=10):
    tmp = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_prob, dtype=float)}).dropna()
    if tmp.empty or tmp["p"].nunique() < 2:
        raise ValueError("Hosmer-Lemeshow cannot be computed due to insufficient variation.")

    q = min(int(g), int(tmp["p"].nunique()), len(tmp))
    tmp["group"] = pd.qcut(tmp["p"], q=q, duplicates="drop")
    rows, chi2 = [], 0.0

    for i, (_, d) in enumerate(tmp.groupby("group", observed=False), start=1):
        n = int(len(d))
        obs1 = float(d["y"].sum())
        obs0 = float(n - obs1)
        exp1 = float(d["p"].sum())
        exp0 = float(n - exp1)
        if exp1 > 0: chi2 += (obs1 - exp1) ** 2 / exp1
        if exp0 > 0: chi2 += (obs0 - exp0) ** 2 / exp0
        rows.append([i, obs0, exp0, obs1, exp1, n])

    detail = pd.DataFrame(rows, columns=["Step", "Observed 0", "Expected 0", "Observed 1", "Expected 1", "Total"])
    df_hl = max(len(detail) - 2, 1)
    pval = float(stats.chi2.sf(chi2, df_hl))

    summary = pd.DataFrame([[
        "Hosmer and Lemeshow Test", chi2, df_hl, format_p_value(pval),
        "Yes" if pval < 0.05 else "No"
    ]], columns=["Test", "Chi-square", "df", "Sig.", "Significant (p<0.05)"])

    return summary, detail

def run_logistic_statsmodels(df: pd.DataFrame, target: str, features: List[str], cutoff: float = 0.5):
    if not features:
        raise ValueError("Please select at least one predictor.")

    raw_n = int(len(df))
    data = df[[target] + features].copy()
    for col in data.columns:
        data[col] = data[col].replace(r"^\s*$", np.nan, regex=True)
        data[col] = pd.to_numeric(data[col], errors="coerce")

    missing_any = data.isna().any(axis=1)
    excluded_n = int(missing_any.sum())
    data = data.dropna().copy()
    included_n = int(len(data))

    if included_n < 10:
        raise ValueError("Not enough valid cases after removing missing rows.")

    unique_y = sorted(data[target].unique())
    if len(unique_y) != 2:
        raise ValueError("Target must have exactly 2 numeric values (e.g. 0 and 1).")

    if set(unique_y) == {0, 1}:
        y = data[target].astype(int)
        encoding_rows = [["0", 0], ["1", 1]]
    else:
        mapping = {unique_y[0]: 0, unique_y[1]: 1}
        y = data[target].map(mapping).astype(int)
        encoding_rows = [[str(unique_y[0]), 0], [str(unique_y[1]), 1]]

    X = data[features].astype(float)
    case_summary = pd.DataFrame([
        ["Selected Cases", "Included in Analysis", included_n, included_n / raw_n * 100],
        ["Selected Cases", "Missing Cases", excluded_n, excluded_n / raw_n * 100],
        ["Selected Cases", "Total", raw_n, 100.0],
    ], columns=["Case Type", "Status", "N", "Percent"])

    encoding_tbl = pd.DataFrame(encoding_rows, columns=["Original Value", "Internal Value"])

    X0 = sm.add_constant(pd.DataFrame(index=X.index), has_constant="add")
    null_model = sm.Logit(y, X0).fit(disp=False)
    X_sm = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X_sm).fit(disp=False)

    ll_null = float(null_model.llf)
    ll_model = float(model.llf)
    minus2ll = -2 * ll_model
    chi2_model = -2 * (ll_null - ll_model)
    df_model = int(len(features))
    p_model = float(stats.chi2.sf(chi2_model, df_model))

    omnibus_tbl = pd.DataFrame([
        ["Step 1", "Step", chi2_model, df_model, format_p_value(p_model)],
        ["Step 1", "Block", chi2_model, df_model, format_p_value(p_model)],
        ["Step 1", "Model", chi2_model, df_model, format_p_value(p_model)],
    ], columns=["Step", "", "Chi-square", "df", "Sig."])

    n = int(len(y))
    cox_snell = 1 - np.exp((2 / n) * (ll_null - ll_model))
    nagelkerke_den = 1 - np.exp((2 / n) * ll_null)
    nagelkerke = cox_snell / nagelkerke_den if nagelkerke_den != 0 else np.nan

    model_summary = pd.DataFrame([[1, minus2ll, cox_snell, nagelkerke]],
                                 columns=["Step", "-2 Log likelihood", "Cox & Snell R Square", "Nagelkerke R Square"])

    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues

    variables_tbl = pd.DataFrame({
        "Variable": params.index,
        "B": params.values,
        "S.E.": model.bse.values,
        "Wald": (params.values / model.bse.values) ** 2,
        "df": 1,
        "Sig.": [format_p_value(p) for p in pvals.values],
        "Exp(B)": np.exp(params.values),
        "95% C.I.for EXP(B) Lower": np.exp(conf[0].values),
        "95% C.I.for EXP(B) Upper": np.exp(conf[1].values),
        "_praw": pvals.values
    })
    variables_tbl["Variable"] = variables_tbl["Variable"].replace({"const": "Constant", "Intercept": "Constant"})
    is_const = variables_tbl["Variable"].eq("Constant")
    variables_tbl.loc[is_const, ["Exp(B)", "95% C.I.for EXP(B) Lower", "95% C.I.for EXP(B) Upper"]] = np.nan
    variables_tbl["Significant (p<0.05)"] = variables_tbl["_praw"].apply(lambda p: "Yes" if float(p) < 0.05 else "No")
    variables_tbl = variables_tbl.drop(columns=["_praw"])

    probs = model.predict(X_sm)
    pred = (probs >= cutoff).astype(int)

    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    r0, r1 = tn + fp, fn + tp
    tot = r0 + r1

    classification_tbl = pd.DataFrame([
        ["Step 1", "0", tn, fp, tn / r0 * 100 if r0 else np.nan],
        ["Step 1", "1", fn, tp, tp / r1 * 100 if r1 else np.nan],
        ["Step 1", "Overall Percentage", "", "", (tn + tp) / tot * 100 if tot else np.nan],
    ], columns=["Step", "Observed", "Predicted 0", "Predicted 1", "Percentage Correct"])

    classification_cutoff = pd.DataFrame([[f"The cut value is {cutoff:.2f}"]], columns=["Classification cutoff"])
    hl_tbl, hl_detail = hosmer_lemeshow_table(y, probs, g=10)

    return {
        "model": model, "y": y, "X": X, "prob": probs,
        "case_summary": compact_numeric_df(case_summary, 3),
        "encoding": encoding_tbl,
        "omnibus": compact_numeric_df(omnibus_tbl, 3),
        "model_summary": compact_numeric_df(model_summary, 3),
        "hosmer": compact_numeric_df(hl_tbl, 3),
        "hosmer_detail": compact_numeric_df(hl_detail, 3),
        "classification_cutoff": classification_cutoff,
        "classification": compact_numeric_df(classification_tbl, 3),
        "table": compact_numeric_df(variables_tbl, 3),
    }

def statsmodels_roc(y_true: np.ndarray, y_prob: np.ndarray):
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    thresholds = np.r_[np.inf, np.unique(y_prob)]
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    tpr, fpr, thr_list = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        TP = int(((y_pred == 1) & (y_true == 1)).sum())
        FP = int(((y_pred == 1) & (y_true == 0)).sum())
        tpr.append(TP / P if P > 0 else 0.0)
        fpr.append(FP / N if N > 0 else 0.0)
        thr_list.append(t if np.isfinite(t) else 1.0)
    return np.array(fpr), np.array(tpr), np.array(thr_list)

def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))

def roc_outputs(model, y: pd.Series, X: pd.DataFrame):
    X_sm = sm.add_constant(X)
    p = model.predict(X_sm)
    fpr, tpr, thr = statsmodels_roc(y.values, p.values)
    auc_val = auc_from_roc(fpr, tpr)
    specificity = 1 - fpr
    youden = tpr + specificity - 1

    finite_mask = np.isfinite(thr)
    if finite_mask.any():
        idx_candidates = np.where(finite_mask)[0]
        best_idx = idx_candidates[np.nanargmax(youden[idx_candidates])]
    else:
        best_idx = int(np.nanargmax(youden))

    auc_tbl = compact_numeric_df(pd.DataFrame([["Area Under the Curve", auc_val]], columns=["Measure", "Value"]), 3)
    cutoff_tbl = compact_numeric_df(pd.DataFrame([[
        thr[best_idx], tpr[best_idx], specificity[best_idx], youden[best_idx]
    ]], columns=["Optimal cutoff", "Sensitivity", "Specificity", "Youden index"]), 3)

    roc_tbl = compact_numeric_df(pd.DataFrame({
        "Threshold": thr,
        "Sensitivity (TPR)": tpr,
        "Specificity": specificity,
        "1 - Specificity (FPR)": fpr,
        "Youden index": youden
    }), 3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"Logistic (AUC={auc_val:.3f})")
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], label=f"Optimal cutoff={thr[best_idx]:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    return auc_tbl, roc_tbl, cutoff_tbl, fig

def fit_linear_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]):
    data = df[[y_col] + x_cols].dropna().copy()
    if len(data) < 3:
        raise ValueError("Not enough rows after dropping missing values.")
    def q(name: str) -> str: return f'Q("{name}")'
    formula = q(y_col) + " ~ " + " + ".join(q(c) for c in x_cols)
    model = smf.ols(formula=formula, data=data).fit()
    return model, data

# =========================================================
# Thuật toán Ước lượng Khoảng tin cậy (CI)
# =========================================================
def ci_combined_estimates(x: np.ndarray, alpha: float = 0.05, use_bootstrap: bool = False, n_boot: int = 5000, seed: int = 123) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean_val = float(np.mean(x))
    s_val = float(np.std(x, ddof=1)) if n > 1 else np.nan
    s2_val = float(np.var(x, ddof=1)) if n > 1 else np.nan

    lo_pct = alpha / 2 * 100
    hi_pct = (1 - alpha / 2) * 100
    lo_label = f"CI {lo_pct:.1f}%" if lo_pct % 1 != 0 else f"CI {lo_pct:.0f}%"
    hi_label = f"CI {hi_pct:.1f}%" if hi_pct % 1 != 0 else f"CI {hi_pct:.0f}%"

    if not use_bootstrap:
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        se = s_val / math.sqrt(n)
        mean_lo = mean_val - tcrit * se
        mean_hi = mean_val + tcrit * se

        chi2_lo = float(stats.chi2.ppf(alpha / 2, df=n - 1))
        chi2_hi = float(stats.chi2.ppf(1 - alpha / 2, df=n - 1))
        var_lo = (n - 1) * s2_val / chi2_hi
        var_hi = (n - 1) * s2_val / chi2_lo

        sd_lo = math.sqrt(max(0.0, var_lo))
        sd_hi = math.sqrt(max(0.0, var_hi))
        method_suffix = ""
    else:
        rng = np.random.default_rng(seed)
        boot_means, boot_sds, boot_vars = [], [], []
        for _ in range(int(n_boot)):
            samp = rng.choice(x, size=n, replace=True)
            boot_means.append(float(np.mean(samp)))
            boot_sds.append(float(np.std(samp, ddof=1)))
            boot_vars.append(float(np.var(samp, ddof=1)))

        mean_lo = float(np.percentile(boot_means, lo_pct))
        mean_hi = float(np.percentile(boot_means, hi_pct))
        sd_lo = float(np.percentile(boot_sds, lo_pct))
        sd_hi = float(np.percentile(boot_sds, hi_pct))
        var_lo = float(np.percentile(boot_vars, lo_pct))
        var_hi = float(np.percentile(boot_vars, hi_pct))
        method_suffix = " (Bootstrap)"

    df_res = pd.DataFrame([
        [f"Mean{method_suffix}", mean_val, mean_lo, mean_hi],
        [f"Std. Deviation{method_suffix}", s_val, sd_lo, sd_hi],
        [f"Variance{method_suffix}", s2_val, var_lo, var_hi],
    ], columns=["Parameter", "Estimate", lo_label, hi_label])

    return compact_numeric_df(df_res, decimals=3)

def ci_from_summary_stats(n: int, mean_val: float, s_val: float, s2_val: float, alpha: float = 0.05) -> pd.DataFrame:
    lo_pct = alpha / 2 * 100
    hi_pct = (1 - alpha / 2) * 100
    lo_label = f"CI {lo_pct:.1f}%" if lo_pct % 1 != 0 else f"CI {lo_pct:.0f}%"
    hi_label = f"CI {hi_pct:.1f}%" if hi_pct % 1 != 0 else f"CI {hi_pct:.0f}%"

    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    se = s_val / math.sqrt(n)
    mean_lo = mean_val - tcrit * se
    mean_hi = mean_val + tcrit * se

    chi2_lo = float(stats.chi2.ppf(alpha / 2, df=n - 1))
    chi2_hi = float(stats.chi2.ppf(1 - alpha / 2, df=n - 1))
    var_lo = (n - 1) * s2_val / chi2_hi
    var_hi = (n - 1) * s2_val / chi2_lo

    sd_lo = math.sqrt(max(0.0, var_lo))
    sd_hi = math.sqrt(max(0.0, var_hi))

    df_res = pd.DataFrame([
        ["Mean", mean_val, mean_lo, mean_hi],
        ["Std. Deviation", s_val, sd_lo, sd_hi],
        ["Variance", s2_val, var_lo, var_hi],
    ], columns=["Parameter", "Estimate", lo_label, hi_label])

    return compact_numeric_df(df_res, decimals=3)

def ci_two_means_diff(n1: int, m1: float, s1: float, n2: int, m2: float, s2: float, alpha: float = 0.05) -> pd.DataFrame:
    diff = m1 - m2
    s1_sq = s1 ** 2
    s2_sq = s2 ** 2

    # 1. Equal variances assumed (Student's t pooled)
    df_pool = n1 + n2 - 2
    sp_sq = ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / df_pool
    se_pool = math.sqrt(sp_sq * (1.0 / n1 + 1.0 / n2))
    tcrit_pool = float(stats.t.ppf(1 - alpha / 2, df=df_pool))
    lo_pool = diff - tcrit_pool * se_pool
    hi_pool = diff + tcrit_pool * se_pool

    # 2. Equal variances not assumed (Welch's t)
    v1 = s1_sq / n1
    v2 = s2_sq / n2
    se_welch = math.sqrt(v1 + v2)
    df_welch = ((v1 + v2) ** 2) / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1))
    tcrit_welch = float(stats.t.ppf(1 - alpha / 2, df=df_welch))
    lo_welch = diff - tcrit_welch * se_welch
    hi_welch = diff + tcrit_welch * se_welch

    lo_pct = alpha / 2 * 100
    hi_pct = (1 - alpha / 2) * 100
    lo_lbl = f"CI {lo_pct:.1f}%" if lo_pct % 1 != 0 else f"CI {lo_pct:.0f}%"
    hi_lbl = f"CI {hi_pct:.1f}%" if hi_pct % 1 != 0 else f"CI {hi_pct:.0f}%"

    res = pd.DataFrame([
        ["Phương sai đồng nhất (Student's t)", diff, se_pool, df_pool, lo_pool, hi_pool],
        ["Phương sai không đồng nhất (Welch's t)", diff, se_welch, df_welch, lo_welch, hi_welch]
    ], columns=["Giả định phương sai", "Hiệu TB (x̄₁ - x̄₂)", "Sai số chuẩn (SE)", "df", lo_lbl, hi_lbl])

    return compact_numeric_df(res, decimals=3)

def ci_two_proportions_diff(x1: int, n1: int, x2: int, n2: int, conf_level: float = 0.95) -> pd.DataFrame:
    alpha = 1.0 - conf_level
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    # 1. Wald method
    se_wald = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    w_lo = diff - z * se_wald
    w_hi = diff + z * se_wald

    # 2. Agresti-Caffo (Adjusted Wald)
    n1_ac = n1 + 2
    x1_ac = x1 + 1
    p1_ac = x1_ac / n1_ac
    n2_ac = n2 + 2
    x2_ac = x2 + 1
    p2_ac = x2_ac / n2_ac
    diff_ac = p1_ac - p2_ac
    se_ac = math.sqrt(p1_ac * (1.0 - p1_ac) / n1_ac + p2_ac * (1.0 - p2_ac) / n2_ac)
    ac_lo = diff_ac - z * se_ac
    ac_hi = diff_ac + z * se_ac

    # 3. Newcombe-Wilson
    w1_lo, w1_hi = wilson_ci(x1, n1, alpha)
    w2_lo, w2_hi = wilson_ci(x2, n2, alpha)
    nw_lo = diff - z * math.sqrt(w1_lo * (1.0 - w1_lo) / n1 + w2_hi * (1.0 - w2_hi) / n2)
    nw_hi = diff + z * math.sqrt(w1_hi * (1.0 - w1_hi) / n1 + w2_lo * (1.0 - w2_lo) / n2)

    rows = [
        ["Wald (Truyền thống)", diff * 100, w_lo * 100, w_hi * 100, (w_hi - w_lo) * 100, "Cỡ mẫu lớn (np >= 5)"],
        ["Newcombe-Wilson (Khuyên dùng)", diff * 100, nw_lo * 100, nw_hi * 100, (nw_hi - nw_lo) * 100, "Tối ưu cho cả mẫu nhỏ & tỷ lệ gần 0 hoặc 1"],
        ["Agresti-Caffo (Hiệu chỉnh)", diff * 100, ac_lo * 100, ac_hi * 100, (ac_hi - ac_lo) * 100, "Hiệu chỉnh cộng 2 thành công/thất bại"]
    ]

    out = pd.DataFrame(rows, columns=[
        "Phương pháp (Method)", "Hiệu tỷ lệ (%)", "Cận dưới (%)", "Cận trên (%)", "Độ rộng KTC (%)", "Khuyến nghị sử dụng"
    ])
    return compact_numeric_df(out, decimals=3)

# =========================================================
# Thuật toán Kiểm định Định lượng
# =========================================================
def numeric_series_from_df(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").dropna()

def selectbox_default(label, options, default=None, key=None):
    options = list(options)
    if not options:
        raise ValueError(f"No available options for {label}.")
    idx = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=idx, key=key)

def numeric_candidate_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

def categorical_candidate_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = exclude or []
    cols = []
    n = max(len(df), 1)
    for c in df.columns:
        if c in exclude: continue
        x = df[c].dropna()
        if x.empty: continue
        is_num = pd.to_numeric(df[c], errors="coerce").notna().sum() == df[c].notna().sum()
        nunique = x.astype(str).nunique()
        if (not is_num) or nunique <= max(10, int(0.4 * n)):
            cols.append(c)
    return cols

def first_existing(cols: List[str], preferred: List[str], fallback=None):
    lower_map = {str(c).lower(): c for c in cols}
    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return fallback if fallback is not None else (cols[0] if cols else None)

def default_numeric_col(df: pd.DataFrame):
    nums = numeric_candidate_cols(df)
    return first_existing(nums, ["value", "score", "measurement", "Y_outcome", "outcome", "after", "before"], nums[0] if nums else None)

def default_group_col(df: pd.DataFrame, exclude: Optional[List[str]] = None):
    cats = categorical_candidate_cols(df, exclude=exclude)
    return first_existing(cats, ["group", "grouping", "factor", "factor_a", "treatment", "arm"], cats[0] if cats else None)

def default_subject_col(df: pd.DataFrame):
    cols = list(df.columns)
    return first_existing(cols, ["subject", "subject_id", "id", "patient", "patient_id"], cols[0] if cols else None)

def default_within_col(df: pd.DataFrame, exclude: Optional[List[str]] = None):
    cats = categorical_candidate_cols(df, exclude=exclude)
    valid = [c for c in cats if df[c].dropna().astype(str).nunique() >= 2]
    preferred = ["time", "factor_b", "condition", "visit", "period", "within", "occasion", "measurement"]
    picked = first_existing(valid, preferred, None)
    if picked is not None:
        return picked
    non_group = [c for c in valid if str(c).lower() not in {"group", "grouping", "factor_a", "treatment", "arm"}]
    return non_group[0] if non_group else (valid[0] if valid else None)

def level_selector(df: pd.DataFrame, group_col: str, prefix: str):
    levels = sorted([str(x) for x in df[group_col].dropna().astype(str).unique()])
    if len(levels) < 2:
        raise ValueError("Grouping variable must have at least 2 groups.")
    g1 = selectbox_default("Group 1", levels, levels[0], key=f"{prefix}_g1")
    remaining = [g for g in levels if g != g1]
    g2 = selectbox_default("Group 2", remaining, remaining[0], key=f"{prefix}_g2")
    return g1, g2

def paired_numeric_data(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    d = df[[col1, col2]].copy()
    d[col1] = pd.to_numeric(d[col1], errors="coerce")
    d[col2] = pd.to_numeric(d[col2], errors="coerce")
    d = d.dropna()
    if len(d) < 2:
        raise ValueError("Not enough paired observations after removing missing values.")
    return d

def long_numeric_group_data(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    d = df[[value_col, group_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d[group_col] = d[group_col].astype(str)
    d = d.dropna()
    if d.empty:
        raise ValueError("No valid numeric observations after removing missing values.")
    return d

def descriptives_for_groups(groups: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, arr in groups.items():
        x = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().astype(float).values
        rows.append([name, len(x), np.mean(x) if len(x) else np.nan, np.std(x, ddof=1) if len(x)>1 else np.nan, np.median(x) if len(x) else np.nan, np.min(x) if len(x) else np.nan, np.max(x) if len(x) else np.nan])
    return compact_numeric_df(pd.DataFrame(rows, columns=["Group", "N", "Mean", "Std. Deviation", "Median", "Minimum", "Maximum"]), 3)

def conclusion_text(pval: float, alpha: float = 0.05, effect_label: str = "difference") -> str:
    try:
        p = float(pval)
    except Exception:
        return "Unable to determine statistical significance."
    if np.isnan(p):
        return "Unable to determine statistical significance."
    if p < alpha:
        return f"Statistically significant {effect_label} (p < {alpha:.2f})."
    return f"No statistically significant {effect_label} (p >= {alpha:.2f})."

def nonparam_result_table(test_name: str, statistic: float, pval: float) -> pd.DataFrame:
    out = pd.DataFrame([[test_name, statistic, format_p_value(pval), "Yes" if pval < 0.05 else "No", conclusion_text(pval)]], columns=["Test", "Statistic", "Sig.", "Significant (p<0.05)", "Conclusion"])
    return compact_numeric_df(out, 3)

def ttest_result_table(test_name: str, statistic: float, dfree, pval: float, mean_diff: float = np.nan, ci=None) -> pd.DataFrame:
    if ci is None:
        ci = (np.nan, np.nan)
    out = pd.DataFrame([[test_name, statistic, dfree, format_p_value(pval), mean_diff, ci[0], ci[1], "Yes" if pval < 0.05 else "No", conclusion_text(pval)]], columns=["Test", "t", "df", "Sig. (2-tailed)", "Mean Difference", "CI 2.5%", "CI 97.5%", "Significant (p<0.05)", "Conclusion"])
    return compact_numeric_df(out, 3)

def one_sample_ttest_table(x: np.ndarray, mu: float, alpha: float = 0.05) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    stat, pval = stats.ttest_1samp(x, popmean=mu, nan_policy="omit")
    n = len(x)
    md = float(np.mean(x) - mu)
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha/2, n-1))
    return ttest_result_table("One-Sample t Test", float(stat), n-1, float(pval), md, (md - tcrit*se, md + tcrit*se))

def independent_ttest_tables(d: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05):
    levels = list(pd.unique(d[group_col]))
    if len(levels) != 2:
        raise ValueError("Independent-samples t test requires exactly 2 groups.")
    x1 = d.loc[d[group_col] == levels[0], value_col].astype(float).values
    x2 = d.loc[d[group_col] == levels[1], value_col].astype(float).values
    if len(x1) < 2 or len(x2) < 2:
        raise ValueError("Each group must have at least 2 valid observations.")
    lev_stat, lev_p = stats.levene(x1, x2, center="mean")
    lev_tbl = compact_numeric_df(pd.DataFrame([["Levene's Test for Equality of Variances", lev_stat, format_p_value(lev_p), "Equal variances assumed" if lev_p >= 0.05 else "Equal variances not assumed"]], columns=["Test", "F", "Sig.", "Decision"]), 3)
    rows = []
    for label, equal_var in [("Equal variances assumed", True), ("Equal variances not assumed (Welch)", False)]:
        res = stats.ttest_ind(x1, x2, equal_var=equal_var, nan_policy="omit")
        md = float(np.mean(x1) - np.mean(x2))
        if equal_var:
            dfree = len(x1) + len(x2) - 2
            sp2 = ((len(x1)-1)*np.var(x1, ddof=1) + (len(x2)-1)*np.var(x2, ddof=1)) / dfree
            se = math.sqrt(sp2*(1/len(x1)+1/len(x2)))
        else:
            v1 = np.var(x1, ddof=1)/len(x1)
            v2 = np.var(x2, ddof=1)/len(x2)
            se = math.sqrt(v1+v2)
            dfree = (v1+v2)**2 / ((v1**2)/(len(x1)-1) + (v2**2)/(len(x2)-1))
        tcrit = float(stats.t.ppf(1-alpha/2, dfree))
        rows.append([label, float(res.statistic), float(dfree), format_p_value(float(res.pvalue)), md, md-tcrit*se, md+tcrit*se, "Yes" if float(res.pvalue)<0.05 else "No"])
    t_tbl = compact_numeric_df(pd.DataFrame(rows, columns=["Assumption", "t", "df", "Sig. (2-tailed)", "Mean Difference", "CI 2.5%", "CI 97.5%", "Significant (p<0.05)"]), 3)
    return {str(levels[0]): x1, str(levels[1]): x2}, lev_tbl, t_tbl

def paired_ttest_table(d: pd.DataFrame, before_col: str, after_col: str, alpha: float = 0.05):
    diff = (d[before_col].astype(float) - d[after_col].astype(float)).values
    stat, pval = stats.ttest_rel(d[before_col].astype(float).values, d[after_col].astype(float).values, nan_policy="omit")
    n = len(diff)
    md = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(1-alpha/2, n-1))
    return {"Paired Difference": diff}, ttest_result_table("Paired-Samples t Test", float(stat), n-1, float(pval), md, (md-tcrit*se, md+tcrit*se))

def anova_summary_table(model, typ=2) -> pd.DataFrame:
    a = anova_lm(model, typ=typ).reset_index().rename(columns={"index": "Source"})
    a["Source"] = a["Source"].apply(clean_term_name)
    a = a.rename(columns={"df": "df", "sum_sq": "Sum Sq", "mean_sq": "Mean Sq", "F": "F", "PR(>F)": "Sig."})
    if "Sig." in a.columns:
        a["Sig."] = a["Sig."].apply(format_p_value)
    for col in ["Sum Sq", "Mean Sq", "F"]:
        if col in a.columns:
            a[col] = pd.to_numeric(a[col], errors="coerce").round(3)
    return a.apply(lambda col: col.map(clean_cell))

def tukey_posthoc_table(d, value_col, group_col, alpha=0.05):
    dd = d[[value_col, group_col]].dropna().copy()
    dd[value_col] = pd.to_numeric(dd[value_col], errors="coerce")
    dd = dd.dropna()
    if dd[group_col].nunique() < 2: return pd.DataFrame()
    res = pairwise_tukeyhsd(endog=dd[value_col].astype(float), groups=dd[group_col].astype(str), alpha=alpha)
    tbl = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return compact_numeric_df(tbl, 3)

# =========================================================
# Khoảng tin cậy Tỷ lệ & Xác suất Chẩn đoán (PPV, NPV)
# =========================================================
def proportion_ci_methods(x, n, conf_level=0.95):
    alpha = 1 - conf_level
    rows = []
    methods = [("Wald", "normal"), ("Wilson", "wilson"), ("Exact (Clopper-Pearson)", "beta"), ("Agresti-Coull", "agresti_coull"), ("Jeffreys", "jeffreys")]
    p_hat = x/n if n else np.nan
    wald_ok = (n*p_hat >= 5 and n*(1-p_hat) >= 5) if n else False
    for label, method in methods:
        try:
            lo, hi = proportion_confint(count=x, nobs=n, alpha=alpha, method=method)
        except Exception:
            lo, hi = np.nan, np.nan
        rows.append([label, p_hat, lo, hi, "Primary" if label == "Wald" and wald_ok else ("Recommended" if label == "Wilson" and not wald_ok else "")])
    out = pd.DataFrame(rows, columns=["Method", "Proportion", "Lower CI", "Upper CI", "Use"])
    out[["Proportion", "Lower CI", "Upper CI"]] = out[["Proportion", "Lower CI", "Upper CI"]] * 100
    return compact_numeric_df(out, 3), wald_ok

def diagnostic_probability_tables(sens_pct, spec_pct, prev_pct, population=1000):
    sens = sens_pct/100
    spec = spec_pct/100
    prev = prev_pct/100
    disease = population * prev
    no_disease = population - disease
    tp = disease * sens
    fn = disease * (1-sens)
    tn = no_disease * spec
    fp = no_disease * (1-spec)
    ppv = tp/(tp+fp) if (tp+fp) else np.nan
    npv = tn/(tn+fn) if (tn+fn) else np.nan

    summary = compact_numeric_df(pd.DataFrame([
        ["Positive Predictive Value (PPV)", ppv*100],
        ["Negative Predictive Value (NPV)", npv*100],
        ["False positive probability after positive test", (1-ppv)*100],
        ["False negative probability after negative test", (1-npv)*100],
    ], columns=["Measure", "Percent"]), 3)

    table = compact_numeric_df(pd.DataFrame([
        ["Test Positive", tp, fp, tp+fp],
        ["Test Negative", fn, tn, fn+tn],
        ["Total", disease, no_disease, population],
    ], columns=["Result", "Disease Present", "Disease Absent", "Total"]), 3)

    prior_odds = prev/(1-prev) if prev < 1 else np.inf
    lr_pos = sens/(1-spec) if spec < 1 else np.inf
    lr_neg = (1-sens)/spec if spec > 0 else np.inf
    post_odds_pos = prior_odds * lr_pos
    post_odds_neg = prior_odds * lr_neg

    calc = compact_numeric_df(pd.DataFrame([
        ["Prior odds", prior_odds],
        ["Likelihood ratio positive (LR+)", lr_pos],
        ["Posterior odds after positive test", post_odds_pos],
        ["Posterior probability after positive test", post_odds_pos/(1+post_odds_pos) if np.isfinite(post_odds_pos) else np.nan],
        ["Likelihood ratio negative (LR-)", lr_neg],
        ["Posterior odds after negative test", post_odds_neg],
        ["Posterior probability after negative test", post_odds_neg/(1+post_odds_neg) if np.isfinite(post_odds_neg) else np.nan],
    ], columns=["Calculation", "Value"]), 3)

    return summary, table, calc

# =========================================================
# Sidebar & Điều hướng TOÀN BỘ
# =========================================================
if "section" not in st.session_state:
    st.session_state.section = "Home"
if "sub" not in st.session_state:
    st.session_state.sub = "Overview"

def set_nav(section: str, sub: str):
    st.session_state.section = section
    st.session_state.sub = sub

with st.sidebar:
    st.markdown("## Navigation")

    if st.button("Home", use_container_width=True):
        set_nav("Home", "Overview")

    with st.expander("Logistic Regression", expanded=(st.session_state.section == "Logistic Regression")):
        if st.button("Data (Upload & Template)", key="log_data", use_container_width=True):
            set_nav("Logistic Regression", "Data")
        if st.button("EDA", key="log_eda", use_container_width=True):
            set_nav("Logistic Regression", "EDA")
        if st.button("Modeling (OR, ROC)", key="log_model", use_container_width=True):
            set_nav("Logistic Regression", "Modeling")
        if st.button("Export", key="log_export", use_container_width=True):
            set_nav("Logistic Regression", "Export")

    with st.expander("Linear Regression (Multivariable)", expanded=(st.session_state.section == "Linear Regression")):
        if st.button("Data (Upload & Template)", key="lin_data", use_container_width=True):
            set_nav("Linear Regression", "Data")
        if st.button("Assumptions & Diagnostics", key="lin_diag", use_container_width=True):
            set_nav("Linear Regression", "Diagnostics")
        if st.button("Modeling (Coefficients, ANOVA)", key="lin_model", use_container_width=True):
            set_nav("Linear Regression", "Modeling")

    with st.expander("Categorical Tests", expanded=(st.session_state.section == "Categorical Tests")):
        if st.button("Contingency Table (r×c) / Chi-square", key="c_1", use_container_width=True):
            set_nav("Categorical Tests", "Chi-square r×c")
        if st.button("Fisher's Exact (2×2)", key="c_2", use_container_width=True):
            set_nav("Categorical Tests", "Fisher 2×2")
        if st.button("Goodness-of-fit", key="c_3", use_container_width=True):
            set_nav("Categorical Tests", "Goodness-of-fit")
        if st.button("Mantel–Haenszel (Stratified 2×2)", key="c_4", use_container_width=True):
            set_nav("Categorical Tests", "Mantel–Haenszel")

    with st.expander("Quantitative Tests", expanded=(st.session_state.section == "Quantitative Tests")):
        if st.button("t Tests", key="qt_ttests", use_container_width=True):
            set_nav("Quantitative Tests", "t Tests")
        if st.button("Nonparametric Tests", key="qt_nonparam", use_container_width=True):
            set_nav("Quantitative Tests", "Nonparametric Tests")
        if st.button("ANOVA", key="qt_anova", use_container_width=True):
            set_nav("Quantitative Tests", "ANOVA")

    with st.expander("Confidence Intervals", expanded=(st.session_state.section == "Confidence Intervals")):
        if st.button("Mean, SD & Variance CI", key="ci_1", use_container_width=True):
            set_nav("Confidence Intervals", "Mean & Variance")
        if st.button("Proportion CI", key="ci_prop", use_container_width=True):
            set_nav("Confidence Intervals", "Proportion")

    with st.expander("Diagnostic Probability", expanded=(st.session_state.section == "Diagnostic Probability")):
        if st.button("Predictive Values (PPV, NPV)", key="diag_prob", use_container_width=True):
            set_nav("Diagnostic Probability", "Predictive Values")

    with st.expander("Tính cỡ mẫu", expanded=(st.session_state.section == "Tính cỡ mẫu")):
        if st.button("Ước lượng KTC", key="ss_ci", use_container_width=True):
            set_nav("Tính cỡ mẫu", "Ước lượng KTC")
        if st.button("Kiểm định giả thuyết", key="ss_test", use_container_width=True):
            set_nav("Tính cỡ mẫu", "Kiểm định giả thuyết")
        if st.button("Hồi quy (Linear & Logistic)", key="ss_reg", use_container_width=True):
            set_nav("Tính cỡ mẫu", "Hồi quy")
        if st.button("ANOVA (k nhóm)", key="ss_anova", use_container_width=True):
            set_nav("Tính cỡ mẫu", "ANOVA")
        if st.button("Phân tích sống sót", key="ss_surv", use_container_width=True):
            set_nav("Tính cỡ mẫu", "Phân tích sống sót")

    with st.expander("Tính xác suất", expanded=(st.session_state.section == "Tính xác suất")):
        if st.button("Công thức xác suất & Bayes", key="prob_formulas", use_container_width=True):
            set_nav("Tính xác suất", "Công thức xác suất & Bayes")
        if st.button("Phân phối Nhị thức B(n, p)", key="prob_binom", use_container_width=True):
            set_nav("Tính xác suất", "Phân phối Nhị thức B(n, p)")
        if st.button("Phân phối Chuẩn N(μ, σ)", key="prob_norm", use_container_width=True):
            set_nav("Tính xác suất", "Phân phối Chuẩn N(μ, σ)")

    # MỤC MỚI: AI TRỢ LÝ
    with st.expander("AI Trợ lý Thông minh", expanded=(st.session_state.section == "AI Trợ lý")):
        if st.button("Giải toán & Tạo trắc nghiệm", key="ai_solver", use_container_width=True):
            set_nav("AI Trợ lý", "Giải toán & Trắc nghiệm")

# =========================================================
# Nội dung từng trang
# =========================================================
section = st.session_state.section
sub = st.session_state.sub

# -----------------------------
# HOME
# -----------------------------
if section == "Home":
    st.markdown("## Overview")
    st.info("Select a module from the sidebar. Each analysis page includes its own template download and file upload panel.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Logistic dataset")
        df = st.session_state.get(LOGISTIC_KEY)
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.success(f"{st.session_state.get('df_logistic_name','')} • {df.shape}")
            st.dataframe(df.head(15), use_container_width=True)
        else:
            st.caption("No logistic dataset loaded.")
    with c2:
        st.markdown("### Linear dataset")
        df = st.session_state.get(LINEAR_KEY)
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.success(f"{st.session_state.get('df_linear_name','')} • {df.shape}")
            st.dataframe(df.head(15), use_container_width=True)
        else:
            st.caption("No linear dataset loaded.")

# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------
elif section == "Logistic Regression":
    logistic_template = pd.DataFrame({
        "target_binary": [0, 1, 0, 1],
        "CRP": [10.2, 35.1, 8.7, 22.4],
        "WBC": [7.1, 12.5, 6.8, 10.3],
    })

    if sub == "Data":
        st.markdown("## Logistic Regression — Data")
        data_input_panel(logistic_template, "logistic_template", LOGISTIC_KEY, "df_logistic_name")
    elif sub == "EDA":
        st.markdown("## Logistic Regression — EDA")
        data_input_panel(logistic_template, "logistic_template", LOGISTIC_KEY, "df_logistic_name")
        df = require_df(LOGISTIC_KEY)
        miss = df.isna().sum().reset_index()
        miss.columns = ["Variable", "Missing"]
        show_table(miss, "Missing Values")
        download_table_block(miss, "logistic_missing_values", "Missing Values")
    elif sub == "Modeling":
        st.markdown("## Logistic Regression — Modeling")
        data_input_panel(logistic_template, "logistic_template", LOGISTIC_KEY, "df_logistic_name")
        df = require_df(LOGISTIC_KEY)
        cols = list(df.columns)
        target = st.selectbox("Target column (binary 0/1)", options=cols)
        features = st.multiselect("Predictors (numeric)", options=[c for c in cols if c != target])
        cutoff = st.slider("Classification cutoff", 0.10, 0.90, 0.50, 0.05)
        if st.button("Run Logistic Regression", type="primary", use_container_width=True):
            try:
                res = run_logistic_statsmodels(df, target, features, cutoff=cutoff)
                show_table(res["case_summary"], "Case Processing Summary")
                download_table_block(res["case_summary"], "logistic_case_processing", "Case Processing Summary")
                show_table(res["encoding"], "Dependent Variable Encoding")
                download_table_block(res["encoding"], "logistic_encoding", "Encoding")
                show_table(res["omnibus"], "Omnibus Tests of Model Coefficients")
                download_table_block(res["omnibus"], "logistic_omnibus", "Omnibus Tests")
                show_table(res["model_summary"], "Model Summary")
                download_table_block(res["model_summary"], "logistic_model_summary", "Model Summary")
                show_table(res["classification"], "Classification Table")
                download_table_block(res["classification"], "logistic_classification", "Classification Table")
                show_table(res["table"], "Variables in the Equation")
                download_table_block(res["table"], "logistic_variables", "Variables in the Equation")

                auc_tbl, roc_tbl, cutoff_tbl, fig = roc_outputs(res["model"], res["y"], res["X"])
                show_table(auc_tbl, "ROC — Area Under the Curve")
                download_table_block(auc_tbl, "logistic_auc", "AUC")
                st.pyplot(fig)
                download_figure_block(fig, "logistic_roc_curve")
                plt.close(fig)
            except Exception as e:
                st.error(f"Modeling failed: {e}")
    elif sub == "Export":
        st.markdown("## Logistic Regression — Export")
        st.info("Each table/figure includes download and copy buttons.")

# -----------------------------
# LINEAR REGRESSION
# -----------------------------
elif section == "Linear Regression":
    linear_template = pd.DataFrame({
        "Y_outcome": [10.2, 12.1, 9.8, 14.0],
        "CRP": [10.2, 35.1, 8.7, 22.4],
        "WBC": [7.1, 12.5, 6.8, 10.3],
    })
    if sub == "Data":
        st.markdown("## Linear Regression — Data")
        data_input_panel(linear_template, "linear_template", LINEAR_KEY, "df_linear_name")
    elif sub == "Diagnostics":
        st.markdown("## Linear Regression — Assumptions & Diagnostics")
        data_input_panel(linear_template, "linear_template", LINEAR_KEY, "df_linear_name")
        df = require_df(LINEAR_KEY)
        cols = list(df.columns)
        y_col = st.selectbox("Dependent variable (Y)", options=cols)
        x_cols = st.multiselect("Independent variables (X)", options=[c for c in cols if c != y_col])
        if st.button("Run diagnostics", type="primary", use_container_width=True):
            try:
                model, data_used = fit_linear_ols(df, y_col, x_cols)
                resid = np.asarray(model.resid, dtype=float)
                sh_stat, sh_p = stats.shapiro(resid) if 3 <= len(resid) <= 5000 else (np.nan, np.nan)
                norm_tbl = compact_numeric_df(pd.DataFrame([["Shapiro-Wilk", sh_stat, format_p_value(sh_p)]], columns=["Test", "Statistic", "Sig."]), 3)
                show_table(norm_tbl, "Residual Normality")
                download_table_block(norm_tbl, "linear_residual_normality", "Residual Normality")

                if het_breuschpagan is not None:
                    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(model.resid, model.model.exog)
                    bp_tbl = compact_numeric_df(pd.DataFrame([["Breusch-Pagan", bp_lm, format_p_value(bp_p)]], columns=["Test", "LM", "Sig."]), 3)
                    show_table(bp_tbl, "Homoscedasticity")
                    download_table_block(bp_tbl, "linear_homoscedasticity", "Homoscedasticity")

                X_exog = model.model.exog
                names = model.model.exog_names
                vif_rows = []
                for i in range(len(names)):
                    if names[i] == "Intercept": continue
                    vif_rows.append([clean_term_name(names[i]), variance_inflation_factor(X_exog, i)])
                vif_df = compact_numeric_df(pd.DataFrame(vif_rows, columns=["Predictor", "VIF"]), 3)
                show_table(vif_df, "Collinearity Statistics (VIF)")
                download_table_block(vif_df, "linear_vif", "VIF")
            except Exception as e:
                st.error(f"Diagnostics failed: {e}")
    elif sub == "Modeling":
        st.markdown("## Linear Regression — Modeling")
        data_input_panel(linear_template, "linear_template", LINEAR_KEY, "df_linear_name")
        df = require_df(LINEAR_KEY)
        cols = list(df.columns)
        y_col = st.selectbox("Dependent variable (Y)", options=cols)
        x_cols = st.multiselect("Independent variables (X)", options=[c for c in cols if c != y_col])
        if st.button("Run linear regression", type="primary", use_container_width=True):
            try:
                model, data_used = fit_linear_ols(df, y_col, x_cols)
                a = anova_summary_table(model, typ=1)
                show_table(a, "ANOVA")
                download_table_block(a, "linear_anova", "ANOVA")

                b = model.summary2().tables[1].reset_index().rename(columns={"index": "Term"})
                b["Term"] = b["Term"].apply(clean_term_name)
                b = b.rename(columns={"Coef.": "B", "Std.Err.": "S.E.", "t": "t", "P>|t|": "Sig.", "[0.025": "CI 2.5%", "0.975]": "CI 97.5%"})
                b["Sig."] = b["Sig."].apply(format_p_value)
                b["Significant (p<0.05)"] = b["Sig."].apply(lambda s: "Yes" if s != "" and s != "< 0.001" and float(s) < 0.05 else ("Yes" if s == "< 0.001" else "No"))
                b = compact_numeric_df(b, 3)
                show_table(b, "Coefficients")
                download_table_block(b, "linear_coefficients", "Coefficients")
            except Exception as e:
                st.error(f"Modeling failed: {e}")

# -----------------------------
# CATEGORICAL TESTS
# -----------------------------
elif section == "Categorical Tests":
    if sub == "Chi-square r×c":
        st.markdown("## Categorical Tests — Contingency Table (r×c)")
        counts_df, observed_df = rc_contingency_ui(key="chisq", default_r=2, default_c=2)
        show_table(counts_df, "Observed Frequencies (with Totals)")
        download_table_block(counts_df, "observed_frequencies_rc", "Observed Frequencies")

        if st.button("Run Chi-square", type="primary", use_container_width=True):
            try:
                obs = get_observed_matrix(observed_df)
                chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)

                chi_tbl = pd.DataFrame([["Pearson Chi-Square", chi2, dof, format_p_value(p), "Yes" if p < 0.05 else "No", conclusion_text(p)]],
                                       columns=["Test", "Value", "df", "Asymp. Sig. (2-sided)", "Significant (p<0.05)", "Conclusion"])
                chi_tbl = compact_numeric_df(chi_tbl, decimals=3)
                show_table(chi_tbl, "Chi-Square Tests")
                download_table_block(chi_tbl, "chisq_tests", "Chi-Square Tests")

                group_labels = st.session_state.get("ct_chisq", pd.DataFrame()).get("Group", pd.Series([""]*obs.shape[0])).tolist()
                exp_df = pd.DataFrame(expected, columns=observed_df.columns)
                exp_df.insert(0, "Group", group_labels[:exp_df.shape[0]])
                exp_df["Total"] = exp_df[observed_df.columns].sum(axis=1)
                total_row = {"Group": "Total"}
                for c in observed_df.columns: total_row[c] = float(exp_df[c].sum())
                total_row["Total"] = float(exp_df["Total"].sum())
                exp_df = pd.concat([exp_df, pd.DataFrame([total_row])], ignore_index=True)
                exp_df = compact_numeric_df(exp_df, decimals=3)
                show_table(exp_df, "Expected Frequencies")
                download_table_block(exp_df, "chisq_expected", "Expected Frequencies")

                if obs.shape == (2, 2):
                    meas = two_by_two_measures(obs, alpha=0.05)
                    show_table(meas, "2×2 Measures (OR, RR, VE, Diagnostic Accuracy)")
                    download_table_block(meas, "chisq_2x2_measures", "2×2 Measures")
            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Fisher 2×2":
        st.markdown("## Categorical Tests — Fisher's Exact Test (2×2)")
        counts_df, observed_df = contingency_editor("fisher", ["Group 1", "Group 2"], ["Outcome +", "Outcome -"], np.array([[10, 30], [20, 15]]))
        show_table(counts_df, "Observed Frequencies (with Totals)")
        download_table_block(counts_df, "observed_frequencies_fisher", "Observed Frequencies")

        if st.button("Run Fisher's Exact", type="primary", use_container_width=True):
            try:
                obs = require_2x2(observed_df)
                oddsratio, p = stats.fisher_exact(obs, alternative="two-sided")
                tbl = compact_numeric_df(pd.DataFrame([["Fisher's Exact Test", oddsratio, format_p_value(p), "Yes" if p < 0.05 else "No"]],
                                                      columns=["Test", "Odds Ratio", "Exact Sig. (2-sided)", "Significant (p<0.05)"]), decimals=3)
                show_table(tbl, "Fisher's Exact Test")
                download_table_block(tbl, "fisher_exact", "Fisher's Exact")
                meas = two_by_two_measures(obs, alpha=0.05)
                show_table(meas, "2×2 Measures (OR, RR, VE, Diagnostic Accuracy)")
                download_table_block(meas, "fisher_2x2_measures", "2×2 Measures")
            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Goodness-of-fit":
        st.markdown("## Categorical Tests — Goodness-of-fit (Chi-square)")
        template_gof = pd.DataFrame({"Category": ["A", "B", "C"], "Observed": [30, 50, 20]})
        st.download_button("Download Excel template", data=df_to_excel_bytes({"gof_template": template_gof}), file_name="gof_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        up = st.file_uploader("Upload GOF template (XLSX/CSV)", type=["xlsx", "csv"], key="gof_upload")
        if up is not None:
            df_gof = load_uploaded_file(up)
            st.dataframe(df_gof, use_container_width=True)
            if "Observed" in df_gof.columns:
                obs = pd.to_numeric(df_gof["Observed"], errors="coerce").dropna().values
                if st.button("Run Goodness-of-fit", type="primary", use_container_width=True):
                    exp = np.ones(len(obs)) * (np.sum(obs) / len(obs))
                    stat, p = stats.chisquare(f_obs=obs, f_exp=exp)
                    gof_res = compact_numeric_df(pd.DataFrame([["Chi-square Goodness-of-fit", stat, len(obs)-1, format_p_value(p), "Yes" if p < 0.05 else "No"]], columns=["Test", "Chi-square", "df", "Sig.", "Significant (p<0.05)"]), 3)
                    show_table(gof_res, "Chi-Square Goodness-of-Fit Test")
                    download_table_block(gof_res, "gof_results", "Goodness-of-Fit")

    elif sub == "Mantel–Haenszel":
        st.markdown("## Categorical Tests — Mantel–Haenszel (Stratified 2×2)")
        template_mh = pd.DataFrame({"Stratum": ["S1", "S2"], "a": [5, 8], "b": [10, 12], "c": [7, 6], "d": [20, 18]})
        st.download_button("Download Excel template", data=df_to_excel_bytes({"mh_template": template_mh}), file_name="mh_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        up = st.file_uploader("Upload MH template (XLSX/CSV)", type=["xlsx", "csv"], key="mh_upload")
        if up is not None:
            df_mh = load_uploaded_file(up)
            st.dataframe(df_mh, use_container_width=True)
            if st.button("Run Mantel–Haenszel", type="primary", use_container_width=True):
                tables = []
                for _, r in df_mh.iterrows():
                    tables.append(np.array([[r["a"], r["b"]], [r["c"], r["d"]]], dtype=int))
                stbl = StratifiedTable(tables)
                mh_or = float(stbl.oddsratio_pooled)
                mh_ci = stbl.oddsratio_pooled_confint()
                mh_p = float(stbl.test_null_odds().pvalue)
                out = compact_numeric_df(pd.DataFrame([["Mantel-Haenszel Pooled OR", mh_or, mh_ci[0], mh_ci[1], format_p_value(mh_p), "Yes" if mh_p < 0.05 else "No"]], columns=["Test", "Common OR", "CI 2.5%", "CI 97.5%", "Sig.", "Significant (p<0.05)"]), 3)
                show_table(out, "Mantel-Haenszel Test")
                download_table_block(out, "mh_results", "Mantel–Haenszel")

# -----------------------------
# QUANTITATIVE TESTS
# -----------------------------
elif section == "Quantitative Tests":
    def _show_template_downloads(kind: str):
        if kind == "ttest":
            c1, c2, c3 = st.columns(3)
            with c1:
                tpl = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [10.2, 11.1, 13.0, 12.4]})
                st.download_button("Template: Independent / 1-sample", data=df_to_excel_bytes({"ttest_ind": tpl}), file_name="ttest_independent_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c2:
                tpl = pd.DataFrame({"subject": [1, 2, 3, 4], "before": [10.0, 11.2, 9.8, 12.0], "after": [11.0, 12.1, 10.3, 12.9]})
                st.download_button("Template: Paired-samples", data=df_to_excel_bytes({"ttest_pair": tpl}), file_name="ttest_paired_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c3:
                tpl = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [10.2, 11.1, 13.0, 12.4], "before": [10.0, 11.2, 9.8, 12.0], "after": [11.0, 12.1, 10.3, 12.9]})
                st.download_button("Template: Combined format", data=df_to_excel_bytes({"ttest_comb": tpl}), file_name="ttest_combined_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        elif kind == "nonparam":
            c1, c2, c3 = st.columns(3)
            with c1:
                tpl = pd.DataFrame({"group": ["A", "A", "B", "B", "C", "C"], "value": [10.2, 11.1, 13.0, 12.4, 9.5, 9.9]})
                st.download_button("Template: Independent groups", data=df_to_excel_bytes({"nonparam_ind": tpl}), file_name="nonparam_independent_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c2:
                tpl = pd.DataFrame({"subject": [1, 2, 3, 4], "before": [10.0, 11.2, 9.8, 12.0], "after": [11.0, 12.1, 10.3, 12.9]})
                st.download_button("Template: Wilcoxon paired", data=df_to_excel_bytes({"wilcox_pair": tpl}), file_name="wilcoxon_paired_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c3:
                tpl = pd.DataFrame({"subject": [1,1,1,2,2,2,3,3,3], "time": ["T1","T2","T3","T1","T2","T3","T1","T2","T3"], "value": [10,12,11,9,10,8,13,15,14]})
                st.download_button("Template: Friedman", data=df_to_excel_bytes({"friedman": tpl}), file_name="friedman_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        elif kind == "anova":
            c1, c2, c3 = st.columns(3)
            with c1:
                tpl = pd.DataFrame({"group": ["A","A","B","B","C","C"], "value": [10.2,11.1,13.0,12.4,9.5,9.9]})
                st.download_button("Template: One-way ANOVA", data=df_to_excel_bytes({"anova_1w": tpl}), file_name="anova_oneway_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c2:
                tpl = pd.DataFrame({"subject": [1,1,1,2,2,2,3,3,3], "time": ["T1","T2","T3","T1","T2","T3","T1","T2","T3"], "value": [10,12,11,9,10,8,13,15,14]})
                st.download_button("Template: Repeated-measures", data=df_to_excel_bytes({"anova_rm": tpl}), file_name="anova_repeated_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c3:
                tpl = pd.DataFrame({"factor_a": ["A","A","B","B","A","A","B","B"], "factor_b": ["T1","T1","T1","T1","T2","T2","T2","T2"], "value": [10,11,13,14,12,13,15,16]})
                st.download_button("Template: Two-way ANOVA", data=df_to_excel_bytes({"anova_2w": tpl}), file_name="anova_twoway_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    if sub == "t Tests":
        st.markdown("## Quantitative Tests — t Tests")
        _show_template_downloads("ttest")
        up = st.file_uploader("Upload t-test data (XLSX/CSV)", type=["xlsx", "csv"], key="ttest_upload")
        if up is not None:
            df = load_uploaded_file(up)
            st.dataframe(df.head(30), use_container_width=True)
            numeric_cols = numeric_candidate_cols(df)
            test_type = st.radio("Test type", ["One-sample t test", "Independent-samples t test", "Paired-samples t test"])
            if test_type == "One-sample t test":
                v_col = st.selectbox("Test variable", numeric_cols)
                mu_val = st.number_input("Test value (mu)", value=0.0)
                if st.button("Run One-Sample t Test", type="primary", use_container_width=True):
                    arr = numeric_series_from_df(df, v_col).values
                    res_t = one_sample_ttest_table(arr, mu_val)
                    show_table(res_t, "One-Sample Test")
                    download_table_block(res_t, "one_sample_ttest", "One-Sample t Test")
            elif test_type == "Independent-samples t test":
                v_col = st.selectbox("Test variable", numeric_cols)
                g_col = st.selectbox("Grouping variable", categorical_candidate_cols(df, exclude=[v_col]))
                g1, g2 = level_selector(df, g_col, "ind_ttest")
                if st.button("Run Independent-Samples t Test", type="primary", use_container_width=True):
                    d = df[df[g_col].astype(str).isin([g1, g2])].copy()
                    groups, lev_tbl, t_tbl = independent_ttest_tables(d, v_col, g_col)
                    show_table(descriptives_for_groups(groups), "Group Statistics")
                    download_table_block(descriptives_for_groups(groups), "ttest_ind_desc", "Group Statistics")
                    show_table(lev_tbl, "Test of Homogeneity of Variances")
                    download_table_block(lev_tbl, "ttest_ind_levene", "Homogeneity of Variances")
                    show_table(t_tbl, "Independent Samples Test")
                    download_table_block(t_tbl, "ttest_ind_results", "Independent Samples Test")
            else:
                c1, c2 = st.columns(2)
                with c1: col1 = st.selectbox("Variable 1 (Before)", numeric_cols)
                with c2: col2 = st.selectbox("Variable 2 (After)", [c for c in numeric_cols if c != col1])
                if st.button("Run Paired-Samples t Test", type="primary", use_container_width=True):
                    d = paired_numeric_data(df, col1, col2)
                    groups, t_tbl = paired_ttest_table(d, col1, col2)
                    show_table(t_tbl, "Paired Samples Test")
                    download_table_block(t_tbl, "ttest_paired_results", "Paired Samples Test")

    elif sub == "Nonparametric Tests":
        st.markdown("## Quantitative Tests — Nonparametric Tests")
        _show_template_downloads("nonparam")
        up = st.file_uploader("Upload nonparametric data (XLSX/CSV)", type=["xlsx", "csv"], key="nonparam_upload")
        if up is not None:
            df = load_uploaded_file(up)
            st.dataframe(df.head(30), use_container_width=True)
            numeric_cols = numeric_candidate_cols(df)
            test_type = st.radio("Test", ["Mann-Whitney U", "Wilcoxon signed-rank (paired)", "Kruskal-Wallis", "Friedman"])
            if test_type == "Mann-Whitney U":
                v_col = st.selectbox("Test variable", numeric_cols)
                g_col = st.selectbox("Grouping variable", categorical_candidate_cols(df, exclude=[v_col]))
                g1, g2 = level_selector(df, g_col, "mw_test")
                if st.button("Run Mann-Whitney U", type="primary", use_container_width=True):
                    x1 = pd.to_numeric(df.loc[df[g_col].astype(str) == g1, v_col], errors="coerce").dropna().values
                    x2 = pd.to_numeric(df.loc[df[g_col].astype(str) == g2, v_col], errors="coerce").dropna().values
                    stat, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
                    out = nonparam_result_table("Mann-Whitney U Test", stat, p)
                    show_table(out, "Test Statistics")
                    download_table_block(out, "mann_whitney_u", "Mann-Whitney U Test")
            elif test_type == "Wilcoxon signed-rank (paired)":
                c1, c2 = st.columns(2)
                with c1: col1 = st.selectbox("Variable 1", numeric_cols)
                with c2: col2 = st.selectbox("Variable 2", [c for c in numeric_cols if c != col1])
                if st.button("Run Wilcoxon Test", type="primary", use_container_width=True):
                    d = paired_numeric_data(df, col1, col2)
                    stat, p = stats.wilcoxon(d[col1].values, d[col2].values, zero_method="wilcox", alternative="two-sided")
                    out = nonparam_result_table("Wilcoxon Signed-Rank Test", stat, p)
                    show_table(out, "Test Statistics")
                    download_table_block(out, "wilcoxon_paired", "Wilcoxon Test")
            elif test_type == "Kruskal-Wallis":
                v_col = st.selectbox("Test variable", numeric_cols)
                g_col = st.selectbox("Grouping variable", categorical_candidate_cols(df, exclude=[v_col]))
                if st.button("Run Kruskal-Wallis", type="primary", use_container_width=True):
                    d = long_numeric_group_data(df, v_col, g_col)
                    groups = [v[v_col].astype(float).values for _, v in d.groupby(g_col)]
                    stat, p = stats.kruskal(*groups)
                    out = nonparam_result_table("Kruskal-Wallis Test", stat, p)
                    show_table(out, "Test Statistics")
                    download_table_block(out, "kruskal_wallis", "Kruskal-Wallis")
            elif test_type == "Friedman":
                sub_col = selectbox_default("Subject ID", list(df.columns), default_subject_col(df))
                val_col = selectbox_default("Test variable", numeric_cols, default_numeric_col(df))
                win_col = selectbox_default("Within-subject factor", categorical_candidate_cols(df, exclude=[sub_col, val_col]))
                if st.button("Run Friedman Test", type="primary", use_container_width=True):
                    d = df[[sub_col, win_col, val_col]].dropna()
                    wide = d.pivot_table(index=sub_col, columns=win_col, values=val_col, aggfunc="mean").dropna()
                    stat, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
                    out = nonparam_result_table("Friedman Test", stat, p)
                    show_table(out, "Test Statistics")
                    download_table_block(out, "friedman_test", "Friedman Test")

    elif sub == "ANOVA":
        st.markdown("## Quantitative Tests — ANOVA")
        _show_template_downloads("anova")
        up = st.file_uploader("Upload ANOVA data (XLSX/CSV)", type=["xlsx", "csv"], key="anova_upload")
        if up is not None:
            df = load_uploaded_file(up)
            st.dataframe(df.head(30), use_container_width=True)
            numeric_cols = numeric_candidate_cols(df)
            anova_type = st.radio("ANOVA Type", ["One-way ANOVA", "One-way repeated-measures ANOVA", "Two-way ANOVA"])
            if anova_type == "One-way ANOVA":
                v_col = st.selectbox("Dependent variable", numeric_cols)
                f_col = st.selectbox("Factor", categorical_candidate_cols(df, exclude=[v_col]))
                if st.button("Run One-way ANOVA", type="primary", use_container_width=True):
                    d = long_numeric_group_data(df, v_col, f_col)
                    model = smf.ols(f'Q("{v_col}") ~ C(Q("{f_col}"))', data=d).fit()
                    a_tbl = anova_summary_table(model, typ=2)
                    show_table(a_tbl, "ANOVA Table")
                    download_table_block(a_tbl, "anova_oneway", "ANOVA")
                    if d[f_col].nunique() >= 3:
                        tukey = tukey_posthoc_table(d, v_col, f_col)
                        show_table(tukey, "Tukey HSD Post-hoc")
                        download_table_block(tukey, "anova_tukey", "Tukey HSD")
            elif anova_type == "One-way repeated-measures ANOVA":
                sub_col = selectbox_default("Subject ID", list(df.columns), default_subject_col(df))
                val_col = selectbox_default("Dependent variable", numeric_cols, default_numeric_col(df))
                win_col = selectbox_default("Within-subject factor", categorical_candidate_cols(df, exclude=[sub_col, val_col]))
                if st.button("Run Repeated-Measures ANOVA", type="primary", use_container_width=True):
                    d = df[[sub_col, win_col, val_col]].dropna()
                    rm = sm.stats.AnovaRM(d, depvar=val_col, subject=sub_col, within=[win_col]).fit()
                    out = rm.anova_table.reset_index().rename(columns={"index": "Source", "F Value": "F", "Num DF": "df1", "Den DF": "df2", "Pr > F": "Sig."})
                    out["Sig."] = out["Sig."].apply(format_p_value)
                    out = compact_numeric_df(out, 3)
                    show_table(out, "Tests of Within-Subjects Effects")
                    download_table_block(out, "anova_rm", "Repeated-Measures ANOVA")
            else:
                v_col = st.selectbox("Dependent variable", numeric_cols)
                f_a = st.selectbox("Factor A", categorical_candidate_cols(df, exclude=[v_col]))
                f_b = st.selectbox("Factor B", categorical_candidate_cols(df, exclude=[v_col, f_a]))
                if st.button("Run Two-way ANOVA", type="primary", use_container_width=True):
                    d = df[[v_col, f_a, f_b]].dropna()
                    formula = f'Q("{v_col}") ~ C(Q("{f_a}")) * C(Q("{f_b}"))'
                    model = smf.ols(formula=formula, data=d).fit()
                    a_tbl = anova_summary_table(model, typ=2)
                    show_table(a_tbl, "Tests of Between-Subjects Effects")
                    download_table_block(a_tbl, "anova_twoway", "Two-way ANOVA")

# -----------------------------
# CONFIDENCE INTERVALS — PROPORTION
# -----------------------------
elif section == "Confidence Intervals" and sub == "Proportion":
    st.markdown("## Confidence Intervals — Proportion")

    ci_prop_target = st.radio(
        "Chọn mục tiêu ước lượng tỷ lệ:",
        ["Ước lượng một tỷ lệ (Single Proportion)", "Ước lượng hiệu hai tỷ lệ (Difference between Two Proportions)"],
        horizontal=True
    )

    if ci_prop_target.startswith("Ước lượng một tỷ lệ"):
        c1, c2, c3 = st.columns(3)
        with c1:
            x_evt = st.number_input("Số biến cố (x / events)", min_value=0, value=50, step=1)
        with c2:
            n_tot = st.number_input("Cỡ mẫu (n / total)", min_value=1, value=100, step=1)
        with c3:
            conf_prop_choice = st.radio("Độ tin cậy", ["95%", "99%", "Khác..."], horizontal=True, key="prop_conf_choice")
            if conf_prop_choice == "95%":
                conf_l = 0.95
            elif conf_prop_choice == "99%":
                conf_l = 0.99
            else:
                conf_l = st.slider("Độ tin cậy tùy chỉnh", 0.80, 0.999, 0.90, 0.005, format="%.3f", key="prop_conf_custom")

        if int(x_evt) > int(n_tot):
            st.error("Số biến cố không thể lớn hơn cỡ mẫu.")
        else:
            if st.button("Compute Proportion CI", type="primary", use_container_width=True):
                prop_tbl, wald_ok = proportion_ci_methods(int(x_evt), int(n_tot), float(conf_l))
                show_table(prop_tbl, "Proportion Confidence Intervals (%)")
                download_table_block(prop_tbl, "proportion_ci", "Proportion CI")

    else:
        st.markdown("### Ước lượng khoảng tin cậy cho hiệu hai tỷ lệ ($p_1 - p_2$)")
        method_p2 = st.radio(
            "Phương thức nhập dữ liệu:",
            ["Upload file (template)", "Paste values", "Enter summary statistics (x1, n1 & x2, n2)"],
            horizontal=True,
            key="prop2_method"
        )

        p2_tpl = pd.DataFrame({
            "Sample_1": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "Sample_2": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        })
        st.download_button(
            "Download Excel template (2 proportions)",
            data=df_to_excel_bytes({"ci_prop2_template": p2_tpl}),
            file_name="ci_diff_proportions_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )

        x1_val, n1_val, x2_val, n2_val = None, None, None, None

        if method_p2 == "Upload file (template)":
            up_p2 = st.file_uploader("Upload file chứa 2 cột tỷ lệ nhị phân 0/1 (XLSX/CSV)", type=["xlsx", "csv"], key="ci_prop2_up")
            if up_p2 is not None:
                df_p2 = load_uploaded_file(up_p2)
                st.dataframe(df_p2.head(30), use_container_width=True)
                cols_p2 = list(df_p2.columns)
                if len(cols_p2) >= 2:
                    col_p1 = st.selectbox("Cột nhóm 1", cols_p2, index=0)
                    col_p2 = st.selectbox("Cột nhóm 2", [c for c in cols_p2 if c != col_p1], index=0)
                    s1_arr = pd.to_numeric(df_p2[col_p1], errors="coerce").dropna().values
                    s2_arr = pd.to_numeric(df_p2[col_p2], errors="coerce").dropna().values
                    n1_val = len(s1_arr)
                    x1_val = int(np.sum(s1_arr == 1))
                    n2_val = len(s2_arr)
                    x2_val = int(np.sum(s2_arr == 1))
                else:
                    st.error("File tải lên cần ít nhất 2 cột dữ liệu.")

        elif method_p2 == "Paste values":
            st.caption("Dán các giá trị 0 và 1 của từng nhóm (cách nhau bởi dấu cách, phẩy, chấm phẩy hoặc xuống dòng):")
            c_txt1, c_txt2 = st.columns(2)
            with c_txt1:
                txt_p1 = st.text_area("Dãy số nhóm 1 (0 và 1)", value="1; 1; 0; 1; 0; 1; 1; 0; 1; 0; 1; 1", height=100)
            with c_txt2:
                txt_p2 = st.text_area("Dãy số nhóm 2 (0 và 1)", value="0; 1; 0; 0; 1; 0; 0; 1; 0; 0; 0; 1", height=100)

            arr_p1 = parse_numeric_text(txt_p1)
            arr_p2 = parse_numeric_text(txt_p2)
            if len(arr_p1) > 0 and len(arr_p2) > 0:
                n1_val = len(arr_p1)
                x1_val = int(np.sum(arr_p1 == 1))
                n2_val = len(arr_p2)
                x2_val = int(np.sum(arr_p2 == 1))
        else:
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown("**Nhóm 1 (Sample 1)**")
                n1_in = st.number_input("Cỡ mẫu nhóm 1 (n1)", min_value=1, value=100, step=1, key="prop2_n1")
                type_p1 = st.radio("Cách nhập nhóm 1:", ["Số biến cố (x1)", "Tỷ lệ phần trăm (p1 %)"], horizontal=True, key="prop2_type1")
                if type_p1.startswith("Số biến cố"):
                    x1_in = st.number_input("Số biến cố (x1)", min_value=0, max_value=int(n1_in), value=45, step=1, key="prop2_x1")
                else:
                    pct1_in = st.number_input("Tỷ lệ nhóm 1 (p1 %)", min_value=0.0, max_value=100.0, value=45.0, step=1.0, key="prop2_pct1")
                    x1_in = int(round(pct1_in * n1_in / 100.0))
                    st.caption(f"Số biến cố x1 tương ứng: **{x1_in}**")

            with c_s2:
                st.markdown("**Nhóm 2 (Sample 2)**")
                n2_in = st.number_input("Cỡ mẫu nhóm 2 (n2)", min_value=1, value=120, step=1, key="prop2_n2")
                type_p2 = st.radio("Cách nhập nhóm 2:", ["Số biến cố (x2)", "Tỷ lệ phần trăm (p2 %)"], horizontal=True, key="prop2_type2")
                if type_p2.startswith("Số biến cố"):
                    x2_in = st.number_input("Số biến cố (x2)", min_value=0, max_value=int(n2_in), value=30, step=1, key="prop2_x2")
                else:
                    pct2_in = st.number_input("Tỷ lệ nhóm 2 (p2 %)", min_value=0.0, max_value=100.0, value=25.0, step=1.0, key="prop2_pct2")
                    x2_in = int(round(pct2_in * n2_in / 100.0))
                    st.caption(f"Số biến cố x2 tương ứng: **{x2_in}**")

            x1_val, n1_val, x2_val, n2_val = int(x1_in), int(n1_in), int(x2_in), int(n2_in)

        st.markdown("#### Độ tin cậy (Confidence level)")
        c_pconf1, c_pconf2 = st.columns([1, 1])
        with c_pconf1:
            conf_diff_choice = st.radio("Chọn mức tin cậy:", ["95%", "99%", "Khác..."], horizontal=True, key="prop2_conf_choice")
        with c_pconf2:
            if conf_diff_choice == "95%":
                conf_level_diff = 0.95
            elif conf_diff_choice == "99%":
                conf_level_diff = 0.99
            else:
                conf_level_diff = st.slider("Nhập mức tin cậy tùy chỉnh", 0.80, 0.999, 0.90, 0.005, format="%.3f", key="prop2_conf_custom")

        if st.button("Compute Difference in Proportions CI", type="primary", use_container_width=True):
            if None in [x1_val, n1_val, x2_val, n2_val] or n1_val <= 0 or n2_val <= 0:
                st.warning("Vui lòng cung cấp đầy đủ dữ liệu hợp lệ cho cả 2 nhóm.")
            else:
                try:
                    p1_hat = x1_val / n1_val
                    p2_hat = x2_val / n2_val
                    se1 = math.sqrt(p1_hat * (1.0 - p1_hat) / n1_val) * 100
                    se2 = math.sqrt(p2_hat * (1.0 - p2_hat) / n2_val) * 100

                    desc_prop2 = pd.DataFrame([
                        ["Nhóm 1 (Sample 1)", x1_val, n1_val, p1_hat * 100, se1],
                        ["Nhóm 2 (Sample 2)", x2_val, n2_val, p2_hat * 100, se2]
                    ], columns=["Nhóm", "Số biến cố (x)", "Cỡ mẫu (n)", "Tỷ lệ (%)", "Sai số chuẩn SE (%)"])
                    desc_prop2 = compact_numeric_df(desc_prop2, decimals=3)
                    show_table(desc_prop2, "Thống kê mô tả hai nhóm tỷ lệ")
                    download_table_block(desc_prop2, "desc_two_proportions", "Mô tả 2 tỷ lệ")

                    diff_ci_tbl = ci_two_proportions_diff(x1_val, n1_val, x2_val, n2_val, conf_level=conf_level_diff)
                    show_table(diff_ci_tbl, f"Khoảng tin cậy cho hiệu hai tỷ lệ (p₁ - p₂) — Mức tin cậy {conf_level_diff*100:.1f}%")
                    download_table_block(diff_ci_tbl, "ci_difference_two_proportions", "KTC hiệu 2 tỷ lệ")
                except Exception as e:
                    st.error(f"Tính toán thất bại: {e}")

# -----------------------------
# CONFIDENCE INTERVALS — MEAN, SD & VARIANCE
# -----------------------------
elif section == "Confidence Intervals" and sub == "Mean & Variance":
    st.markdown("## Confidence Intervals — Mean, SD & Variance")

    ci_mean_target = st.radio(
        "Chọn mục tiêu ước lượng:",
        ["Ước lượng một số trung bình (Mean, SD, Variance)", "Ước lượng hiệu hai số trung bình (Difference between Two Means)"],
        horizontal=True
    )

    if ci_mean_target.startswith("Ước lượng một số trung bình"):
        template = pd.DataFrame({"X": [1.2, 2.0, 1.8, 2.2, 1.6]})
        st.download_button(
            "Download Excel template",
            data=df_to_excel_bytes({"ci_template": template}),
            file_name="ci_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )

        method = st.radio(
            "Input method",
            ["Upload file (template)", "Paste values", "Enter summary statistics (n, Mean, s/s²)"],
            horizontal=True
        )

        x = None
        summary_params = None

        if method == "Upload file (template)":
            up = st.file_uploader("Upload CI template (XLSX/CSV)", type=["xlsx", "csv"], key="ci_upload")
            if up is not None:
                df = load_uploaded_file(up)
                st.dataframe(df.head(50), use_container_width=True)
                if "X" not in df.columns:
                    st.error("Template must have a column named 'X'.")
                else:
                    x = pd.to_numeric(df["X"], errors="coerce").dropna().values
        elif method == "Paste values":
            txt = st.text_area(
                "Paste numeric values (separated by ; , space or newline)",
                value="12; 14; 16; 18; 20; 22; 24; 26; 28; 30; 32; 34; 36; 38; 40; 42; 44; 46; 48; 50",
                height=100
            )
            if txt.strip():
                x = parse_numeric_text(txt)
        else:
            st.markdown("#### Nhập tham số thống kê mẫu")
            c1, c2 = st.columns(2)
            with c1:
                n_input = st.number_input("Cỡ mẫu (n)", min_value=2, value=20, step=1, key="ci_sum_n")
                mean_input = st.number_input("Trung bình mẫu (Mean, x̄)", value=31.000, format="%.4f", key="ci_sum_mean")
            with c2:
                disp_choice = st.radio(
                    "Chọn tham số độ phân tán để nhập:",
                    ["Độ lệch chuẩn (s)", "Phương sai (s²)"],
                    horizontal=True,
                    key="ci_sum_disp_choice"
                )
                if disp_choice == "Độ lệch chuẩn (s)":
                    s_input = st.number_input("Độ lệch chuẩn mẫu (s)", min_value=0.0001, value=11.832, format="%.4f", key="ci_sum_s")
                    var_input = s_input ** 2
                    st.caption(f"Phương sai tương ứng ($s^2$): **{var_input:.4f}**")
                else:
                    var_input = st.number_input("Phương sai mẫu (s²)", min_value=0.0001, value=140.000, format="%.4f", key="ci_sum_var")
                    s_input = math.sqrt(var_input)
                    st.caption(f"Độ lệch chuẩn tương ứng ($s$): **{s_input:.4f}**")

            summary_params = {
                "n": int(n_input),
                "mean": float(mean_input),
                "s": float(s_input),
                "s2": float(var_input)
            }

        st.markdown("#### Độ tin cậy (Confidence level)")
        c_conf1, c_conf2 = st.columns([1, 1])
        with c_conf1:
            conf_choice = st.radio(
                "Chọn mức tin cậy:",
                ["95%", "99%", "Khác..."],
                index=0,
                horizontal=True,
                key="ci_conf_choice"
            )
        with c_conf2:
            if conf_choice == "95%":
                conf_level = 0.95
            elif conf_choice == "99%":
                conf_level = 0.99
            else:
                conf_level = st.slider(
                    "Nhập mức tin cậy tùy chỉnh",
                    min_value=0.80,
                    max_value=0.999,
                    value=0.90,
                    step=0.005,
                    format="%.3f",
                    key="ci_conf_custom"
                )

        alpha_tail = 1.0 - conf_level

        if method in ["Upload file (template)", "Paste values"]:
            c_boot1, c_boot2 = st.columns(2)
            with c_boot1:
                force_boot = st.checkbox("Force bootstrap (recommended if non-normal)", value=False, key="ci_force_boot")
            with c_boot2:
                n_boot = st.number_input("Bootstrap resamples", min_value=1000, max_value=20000, value=5000, step=500, key="ci_n_boot")
        else:
            force_boot = False
            n_boot = 5000

        st.markdown("### Confidence Interval Results")
        if st.button("Compute CI", type="primary", use_container_width=True):
            if method == "Enter summary statistics (n, Mean, s/s²)":
                try:
                    n_v = summary_params["n"]
                    m_v = summary_params["mean"]
                    s_v = summary_params["s"]
                    s2_v = summary_params["s2"]
                    se_v = s_v / math.sqrt(n_v)

                    desc_summary_df = pd.DataFrame([{
                        "n": n_v,
                        "Mean": m_v,
                        "s": s_v,
                        "s²": s2_v,
                        "Std. Error (SE)": se_v
                    }])
                    desc_summary_df = compact_numeric_df(desc_summary_df, decimals=3)
                    show_table(desc_summary_df, "Sample Summary Statistics")
                    download_table_block(desc_summary_df, "ci_summary_statistics", "Sample Summary Statistics")

                    ci_table = ci_from_summary_stats(
                        n=n_v,
                        mean_val=m_v,
                        s_val=s_v,
                        s2_val=s2_v,
                        alpha=alpha_tail
                    )
                    show_table(ci_table, "Confidence Interval Estimates (Parametric: Student-t & Chi-square)")
                    download_table_block(ci_table, "ci_estimates_combined", "Confidence Interval Estimates")
                except Exception as e:
                    st.error(f"Tính toán thất bại: {e}")
            else:
                if x is None or len(x) < 2:
                    st.warning("Vui lòng nhập ít nhất 2 giá trị số hợp lệ.")
                else:
                    try:
                        n = int(len(x))
                        mean_v = float(np.mean(x))
                        median_v = float(np.median(x))
                        s_v = float(np.std(x, ddof=1))
                        s2_v = float(np.var(x, ddof=1))
                        min_v = float(np.min(x))
                        max_v = float(np.max(x))
                        rng_v = max_v - min_v

                        q1 = compute_percentile_textbook(x, 25)
                        q3 = compute_percentile_textbook(x, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        has_outliers = "Có" if np.any((x < lower_bound) | (x > upper_bound)) else "Không"

                        mode_res = stats.mode(x, keepdims=True)
                        mode_v = mode_res.mode[0] if len(mode_res.mode) > 0 else np.nan

                        desc_df = pd.DataFrame([{
                            "n": n,
                            "Mean": mean_v,
                            "Mode": mode_v,
                            "Median": median_v,
                            "s": s_v,
                            "s²": s2_v,
                            "Min": min_v,
                            "Max": max_v,
                            "Range": rng_v,
                            "Q1": q1,
                            "Q3": q3,
                            "IQR": iqr
                        }])
                        desc_df = compact_numeric_df(desc_df, decimals=3)
                        show_table(desc_df, "Descriptive Statistics")
                        download_table_block(desc_df, "ci_descriptive_statistics", "Descriptive Statistics")

                        if 3 <= n <= 5000:
                            sw_stat, sw_p = stats.shapiro(x)
                        else:
                            sw_stat, sw_p = np.nan, np.nan

                        if lilliefors is not None and n >= 4:
                            ks_stat, ks_p = lilliefors(x, dist='norm')
                        else:
                            ks_res = stats.kstest(x, 'norm', args=(mean_v, s_v))
                            ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)

                        is_normal = (sw_p >= 0.05) if not np.isnan(sw_p) else ((ks_p >= 0.05) if not np.isnan(ks_p) else True)
                        norm_status = "Có" if is_normal else "Không"

                        normality_diag_df = pd.DataFrame([{
                            "[Q1-1.5IQR; Q3+1.5IQR]": f"[{smart_round_val(lower_bound, 3)}; {smart_round_val(upper_bound, 3)}]",
                            "Outliers": has_outliers,
                            "Statistic (Shapiro-Wilk)": sw_stat,
                            "Sig. (Shapiro-Wilk)": format_p_value(sw_p),
                            "Statistic (Kolmogorov-Smirnov)": ks_stat,
                            "Sig. (Kolmogorov-Smirnov)": format_p_value(ks_p),
                            "Phân phối chuẩn": norm_status
                        }])
                        normality_diag_df = compact_numeric_df(normality_diag_df, decimals=3)
                        show_table(normality_diag_df, "Normality & Outlier Diagnostics")
                        download_table_block(normality_diag_df, "ci_normality_diagnostics", "Normality & Outlier Diagnostics")

                        use_boot = force_boot or (not is_normal)
                        method_title = "Bootstrap" if use_boot else "Parametric"

                        ci_table = ci_combined_estimates(
                            x=x,
                            alpha=alpha_tail,
                            use_bootstrap=use_boot,
                            n_boot=int(n_boot)
                        )

                        show_table(ci_table, f"Confidence Interval Estimates ({method_title})")
                        download_table_block(ci_table, "ci_estimates_combined", f"Confidence Interval Estimates ({method_title})")

                    except Exception as e:
                        st.error(f"Tính toán thất bại: {e}")

    else:
        st.markdown("### Ước lượng khoảng tin cậy cho hiệu hai số trung bình ($\mu_1 - \mu_2$)")
        method_m2 = st.radio(
            "Phương thức nhập dữ liệu:",
            ["Upload file (template)", "Paste values", "Enter summary statistics (n, Mean, s/s²)"],
            horizontal=True,
            key="mean2_method"
        )

        template_diff_m = pd.DataFrame({
            "Sample_1": [12.5, 14.2, 11.8, 15.0, 13.6, 14.8, 12.9],
            "Sample_2": [10.2, 11.5, 9.8, 12.0, 10.9, 11.2, 9.5]
        })
        st.download_button(
            "Download Excel template (2 samples)",
            data=df_to_excel_bytes({"ci_diff_mean_template": template_diff_m}),
            file_name="ci_diff_mean_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )

        s1_arr, s2_arr = None, None
        m2_summary = None

        if method_m2 == "Upload file (template)":
            up_m2 = st.file_uploader("Upload file chứa 2 cột dữ liệu độc lập (XLSX/CSV)", type=["xlsx", "csv"], key="ci_mean2_up")
            if up_m2 is not None:
                df_m2 = load_uploaded_file(up_m2)
                st.dataframe(df_m2.head(30), use_container_width=True)
                cols_m2 = list(df_m2.columns)
                if len(cols_m2) >= 2:
                    c_col1 = st.selectbox("Cột mẫu 1", cols_m2, index=0)
                    c_col2 = st.selectbox("Cột mẫu 2", [c for c in cols_m2 if c != c_col1], index=0)
                    s1_arr = pd.to_numeric(df_m2[c_col1], errors="coerce").dropna().values
                    s2_arr = pd.to_numeric(df_m2[c_col2], errors="coerce").dropna().values
                else:
                    st.error("File tải lên cần ít nhất 2 cột dữ liệu số.")

        elif method_m2 == "Paste values":
            st.caption("Dán các giá trị của từng nhóm (cách nhau bởi dấu cách, phẩy, chấm phẩy hoặc xuống dòng):")
            c_txt1, c_txt2 = st.columns(2)
            with c_txt1:
                txt_m1 = st.text_area("Dãy số nhóm 1 (Sample 1)", value="12.5; 14.2; 11.8; 15.0; 13.6; 14.8; 12.9", height=100)
            with c_txt2:
                txt_m2 = st.text_area("Dãy số nhóm 2 (Sample 2)", value="10.2; 11.5; 9.8; 12.0; 10.9; 11.2; 9.5", height=100)

            s1_arr = parse_numeric_text(txt_m1)
            s2_arr = parse_numeric_text(txt_m2)
        else:
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown("**Nhóm 1 (Sample 1)**")
                n1_m = st.number_input("Cỡ mẫu (n1)", min_value=2, value=30, step=1, key="m2_n1")
                m1_m = st.number_input("Trung bình (x̄1)", value=14.500, format="%.4f", key="m2_m1")
                disp1_choice = st.radio("Độ phân tán nhóm 1:", ["Độ lệch chuẩn (s1)", "Phương sai (s1²)"], horizontal=True, key="m2_disp1")
                if disp1_choice.startswith("Độ lệch"):
                    s1_m = st.number_input("s1", min_value=0.0001, value=2.500, format="%.4f", key="m2_s1")
                else:
                    var1_m = st.number_input("s1²", min_value=0.0001, value=6.250, format="%.4f", key="m2_var1")
                    s1_m = math.sqrt(var1_m)

            with c_s2:
                st.markdown("**Nhóm 2 (Sample 2)**")
                n2_m = st.number_input("Cỡ mẫu (n2)", min_value=2, value=35, step=1, key="m2_n2")
                m2_m = st.number_input("Trung bình (x̄2)", value=11.200, format="%.4f", key="m2_m2")
                disp2_choice = st.radio("Độ phân tán nhóm 2:", ["Độ lệch chuẩn (s2)", "Phương sai (s2²)"], horizontal=True, key="m2_disp2")
                if disp2_choice.startswith("Độ lệch"):
                    s2_m = st.number_input("s2", min_value=0.0001, value=2.800, format="%.4f", key="m2_s2")
                else:
                    var2_m = st.number_input("s2²", min_value=0.0001, value=7.840, format="%.4f", key="m2_var2")
                    s2_m = math.sqrt(var2_m)

            m2_summary = {
                "n1": int(n1_m), "m1": float(m1_m), "s1": float(s1_m),
                "n2": int(n2_m), "m2": float(m2_m), "s2": float(s2_m)
            }

        st.markdown("#### Độ tin cậy (Confidence level)")
        c_mconf1, c_mconf2 = st.columns([1, 1])
        with c_mconf1:
            conf_m2_choice = st.radio("Chọn mức tin cậy:", ["95%", "99%", "Khác..."], horizontal=True, key="m2_conf_choice")
        with c_mconf2:
            if conf_m2_choice == "95%":
                conf_level_m2 = 0.95
            elif conf_m2_choice == "99%":
                conf_level_m2 = 0.99
            else:
                conf_level_m2 = st.slider("Nhập mức tin cậy tùy chỉnh", 0.80, 0.999, 0.90, 0.005, format="%.3f", key="m2_conf_custom")

        alpha_m2 = 1.0 - conf_level_m2

        if st.button("Compute Difference in Means CI", type="primary", use_container_width=True):
            if method_m2 == "Enter summary statistics (n, Mean, s/s²)":
                try:
                    n1 = m2_summary["n1"]
                    m1 = m2_summary["m1"]
                    s1 = m2_summary["s1"]
                    n2 = m2_summary["n2"]
                    m2 = m2_summary["m2"]
                    s2 = m2_summary["s2"]

                    desc_two_df = pd.DataFrame([
                        ["Nhóm 1 (Sample 1)", n1, m1, s1, s1**2, s1/math.sqrt(n1)],
                        ["Nhóm 2 (Sample 2)", n2, m2, s2, s2**2, s2/math.sqrt(n2)]
                    ], columns=["Nhóm", "Cỡ mẫu (n)", "Trung bình (x̄)", "Độ lệch chuẩn (s)", "Phương sai (s²)", "Sai số chuẩn (SE)"])
                    desc_two_df = compact_numeric_df(desc_two_df, decimals=3)
                    show_table(desc_two_df, "Thống kê mô tả hai nhóm")
                    download_table_block(desc_two_df, "desc_two_means", "Mô tả 2 nhóm")

                    diff_tbl = ci_two_means_diff(n1, m1, s1, n2, m2, s2, alpha=alpha_m2)
                    show_table(diff_tbl, f"Khoảng tin cậy cho hiệu hai số trung bình (μ₁ - μ₂) — Mức tin cậy {conf_level_m2*100:.1f}%")
                    download_table_block(diff_tbl, "ci_difference_two_means", "KTC hiệu 2 trung bình")
                except Exception as e:
                    st.error(f"Tính toán thất bại: {e}")
            else:
                if s1_arr is None or s2_arr is None or len(s1_arr) < 2 or len(s2_arr) < 2:
                    st.warning("Vui lòng cung cấp ít nhất 2 quan sát hợp lệ cho mỗi nhóm.")
                else:
                    try:
                        n1, m1, s1 = len(s1_arr), float(np.mean(s1_arr)), float(np.std(s1_arr, ddof=1))
                        n2, m2, s2 = len(s2_arr), float(np.mean(s2_arr)), float(np.std(s2_arr, ddof=1))

                        desc_two_df = pd.DataFrame([
                            ["Nhóm 1 (Sample 1)", n1, m1, s1, s1**2, s1/math.sqrt(n1)],
                            ["Nhóm 2 (Sample 2)", n2, m2, s2, s2**2, s2/math.sqrt(n2)]
                        ], columns=["Nhóm", "Cỡ mẫu (n)", "Trung bình (x̄)", "Độ lệch chuẩn (s)", "Phương sai (s²)", "Sai số chuẩn (SE)"])
                        desc_two_df = compact_numeric_df(desc_two_df, decimals=3)
                        show_table(desc_two_df, "Thống kê mô tả hai nhóm")
                        download_table_block(desc_two_df, "desc_two_means", "Mô tả 2 nhóm")

                        lev_s, lev_p = stats.levene(s1_arr, s2_arr, center="mean")
                        lev_df = compact_numeric_df(pd.DataFrame([[
                            "Levene's Test for Equality of Variances", lev_s, format_p_value(lev_p),
                            "Phương sai đồng nhất (p >= 0.05)" if lev_p >= 0.05 else "Phương sai không đồng nhất (p < 0.05)"
                        ]], columns=["Kiểm định", "Thống kê F", "Sig.", "Kết luận"]), decimals=3)
                        show_table(lev_df, "Kiểm định tính đồng nhất của phương sai (Levene)")
                        download_table_block(lev_df, "levene_homogeneity", "Kiểm định Levene")

                        diff_tbl = ci_two_means_diff(n1, m1, s1, n2, m2, s2, alpha=alpha_m2)
                        diff_tbl["Khuyến nghị"] = ["Ưu tiên sử dụng" if lev_p >= 0.05 else "", "Ưu tiên sử dụng" if lev_p < 0.05 else ""]
                        show_table(diff_tbl, f"Khoảng tin cậy cho hiệu hai số trung bình (μ₁ - μ₂) — Mức tin cậy {conf_level_m2*100:.1f}%")
                        download_table_block(diff_tbl, "ci_difference_two_means", "KTC hiệu 2 trung bình")
                    except Exception as e:
                        st.error(f"Tính toán thất bại: {e}")

# -----------------------------
# DIAGNOSTIC PROBABILITY (PPV, NPV)
# -----------------------------
elif section == "Diagnostic Probability" and sub == "Predictive Values":
    st.markdown("## Diagnostic Probability — Predictive Values")
    st.write("Compute positive and negative predictive values from sensitivity, specificity, and prevalence.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sens = st.number_input("Sensitivity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    with c2:
        spec = st.number_input("Specificity (%)", min_value=0.0, max_value=100.0, value=78.0, step=0.1)
    with c3:
        prev = st.number_input("Prevalence (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    with c4:
        population = st.number_input("Population size", min_value=1, value=1000, step=100)

    if st.button("Compute predictive values", type="primary", use_container_width=True):
        summary, table, calc = diagnostic_probability_tables(float(sens), float(spec), float(prev), int(population))
        show_table(summary, "Predictive Values (PPV, NPV)")
        download_table_block(summary, "diagnostic_predictive_values", "Predictive Values")
        show_table(table, f"Expected Results per {int(population)} People")
        download_table_block(table, "diagnostic_expected_results", "Expected Results")
        show_table(calc, "Calculations (Likelihood Ratios & Odds)")
        download_table_block(calc, "diagnostic_calculations", "Calculations")

# -----------------------------
# TÍNH CỠ MẪU (SAMPLE SIZE)
# -----------------------------
elif section == "Tính cỡ mẫu":
    st.markdown(f"## Tính cỡ mẫu tối thiểu — {sub}")

    if sub == "Ước lượng KTC":
        type_ci = st.radio("Mục tiêu ước lượng", ["Ước lượng một tỷ lệ (Proportion)", "Ước lượng một số trung bình (Mean)"], horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            conf_level = st.selectbox("Độ tin cậy (1 - α)", [0.95, 0.99, 0.90], index=0)
            alpha = 1.0 - conf_level
            z_val = stats.norm.ppf(1.0 - alpha / 2.0)
        with c2:
            finite_pop = st.checkbox("Hiệu chỉnh cho quần thể hữu hạn (Finite Population)?", value=False)
            pop_size = st.number_input("Quy mô quần thể (N)", min_value=10, value=5000, step=100) if finite_pop else None

        if type_ci == "Ước lượng một tỷ lệ (Proportion)":
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                p_est = st.number_input("Tỷ lệ ước tính từ y văn hoặc NC thử nghiệm (p)", min_value=0.01, max_value=0.99, value=0.50, step=0.05)
            with c_p2:
                d_val = st.number_input("Khoảng sai số tuyệt đối mong muốn (d hoặc Precision)", min_value=0.005, max_value=0.30, value=0.05, step=0.01, format="%.4f")

            if st.button("Tính cỡ mẫu ước lượng tỷ lệ", type="primary", use_container_width=True):
                n0 = (z_val ** 2) * p_est * (1.0 - p_est) / (d_val ** 2)
                if finite_pop and pop_size is not None:
                    n_final = n0 / (1.0 + (n0 - 1.0) / pop_size)
                else:
                    n_final = n0
                n_rec = int(math.ceil(n_final))

                res_df = pd.DataFrame([{
                    "Tỷ lệ dự đoán (p)": p_est,
                    "Sai số mong muốn (d)": d_val,
                    "Mức tin cậy (1-α)": f"{conf_level*100:.1f}%",
                    "Z(1-α/2)": z_val,
                    "Cỡ mẫu tính toán": n_final,
                    "Cỡ mẫu làm tròn lên (n)": n_rec,
                    "Dự phòng mất mẫu 10%": int(math.ceil(n_rec * 1.10))
                }])
                res_df = compact_numeric_df(res_df, decimals=3)
                show_table(res_df, "Kết quả tính cỡ mẫu ước lượng tỷ lệ")
                download_table_block(res_df, "sample_size_ci_proportion", "Cỡ mẫu ước lượng tỷ lệ")

        else:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                sd_est = st.number_input("Độ lệch chuẩn ước tính từ nghiên cứu trước (s hoặc σ)", min_value=0.001, value=5.000, step=0.5, format="%.4f")
            with c_m2:
                d_mean = st.number_input("Khoảng sai số biên độ mong muốn (d hoặc Precision)", min_value=0.01, value=1.000, step=0.1, format="%.4f")

            if st.button("Tính cỡ mẫu ước lượng trung bình", type="primary", use_container_width=True):
                n0 = (z_val ** 2) * (sd_est ** 2) / (d_mean ** 2)
                if finite_pop and pop_size is not None:
                    n_final = n0 / (1.0 + (n0 - 1.0) / pop_size)
                else:
                    n_final = n0
                n_rec = int(math.ceil(n_final))

                res_df = pd.DataFrame([{
                    "Độ lệch chuẩn (s)": sd_est,
                    "Sai số biên độ (d)": d_mean,
                    "Mức tin cậy (1-α)": f"{conf_level*100:.1f}%",
                    "Z(1-α/2)": z_val,
                    "Cỡ mẫu tính toán": n_final,
                    "Cỡ mẫu làm tròn lên (n)": n_rec,
                    "Dự phòng mất mẫu 10%": int(math.ceil(n_rec * 1.10))
                }])
                res_df = compact_numeric_df(res_df, decimals=3)
                show_table(res_df, "Kết quả tính cỡ mẫu ước lượng trung bình")
                download_table_block(res_df, "sample_size_ci_mean", "Cỡ mẫu ước lượng trung bình")

    elif sub == "Kiểm định giả thuyết":
        test_category = st.radio("Chọn phép kiểm định hai phía", ["So sánh 2 tỷ lệ độc lập", "So sánh 2 số trung bình độc lập", "So sánh 2 số trung bình bắt cặp"], horizontal=True)
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            alpha_test = st.selectbox("Mức ý nghĩa thống kê (α)", [0.05, 0.01, 0.10], index=0, key="hyp_alpha")
            z_a = stats.norm.ppf(1.0 - alpha_test / 2.0)
        with c_a2:
            power_test = st.selectbox("Lực lượng mẫu / Power (1 - β)", [0.80, 0.90, 0.85, 0.95], index=0, key="hyp_power")
            z_b = stats.norm.ppf(power_test)

        if test_category == "So sánh 2 tỷ lệ độc lập":
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                p1_val = st.number_input("Tỷ lệ nhóm 1 (p1)", min_value=0.01, max_value=0.99, value=0.40, step=0.05)
            with c_p2:
                p2_val = st.number_input("Tỷ lệ nhóm 2 (p2)", min_value=0.01, max_value=0.99, value=0.20, step=0.05)

            if st.button("Tính cỡ mẫu kiểm định 2 tỷ lệ", type="primary", use_container_width=True):
                p_bar = (p1_val + p2_val) / 2.0
                num = (z_a * math.sqrt(2.0 * p_bar * (1.0 - p_bar)) + z_b * math.sqrt(p1_val * (1.0 - p1_val) + p2_val * (1.0 - p2_val))) ** 2
                den = (p1_val - p2_val) ** 2
                n_group = num / den if den > 0 else np.nan
                n_rec_grp = int(math.ceil(n_group))

                res_df = pd.DataFrame([{
                    "Tỷ lệ nhóm 1 (p1)": p1_val,
                    "Tỷ lệ nhóm 2 (p2)": p2_val,
                    "Chênh lệch (p1 - p2)": abs(p1_val - p2_val),
                    "Alpha (α)": alpha_test,
                    "Power (1-β)": f"{power_test*100:.0f}%",
                    "Cỡ mẫu mỗi nhóm (n)": n_rec_grp,
                    "Tổng cỡ mẫu 2 nhóm (2n)": n_rec_grp * 2,
                    "Tổng tính cả dự phòng 10%": int(math.ceil(n_rec_grp * 2 * 1.10))
                }])
                res_df = compact_numeric_df(res_df, decimals=3)
                show_table(res_df, "Kết quả cỡ mẫu so sánh 2 tỷ lệ")
                download_table_block(res_df, "sample_size_two_proportions", "Cỡ mẫu 2 tỷ lệ")

        elif test_category == "So sánh 2 số trung bình độc lập":
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                mu_diff = st.number_input("Chênh lệch trung bình có ý nghĩa lâm sàng (|μ1 - μ2|)", min_value=0.01, value=2.0, step=0.2)
            with c_m2:
                sd_pool = st.number_input("Độ lệch chuẩn gộp ước tính (σ)", min_value=0.01, value=3.0, step=0.2)
            with c_m3:
                ratio_k = st.number_input("Tỷ lệ phân bổ mẫu (Nhóm 2 / Nhóm 1)", min_value=0.2, max_value=5.0, value=1.0, step=0.1)

            if st.button("Tính cỡ mẫu so sánh 2 trung bình", type="primary", use_container_width=True):
                n1_calc = (1.0 + 1.0 / ratio_k) * ((z_a + z_b) ** 2) * (sd_pool ** 2) / (mu_diff ** 2)
                n1_rec = int(math.ceil(n1_calc))
                n2_rec = int(math.ceil(n1_rec * ratio_k))

                res_df = pd.DataFrame([{
                    "Chênh lệch (|μ1 - μ2|)": mu_diff,
                    "Độ lệch chuẩn (σ)": sd_pool,
                    "Tỷ lệ n2/n1": ratio_k,
                    "Nhóm 1 (n1)": n1_rec,
                    "Nhóm 2 (n2)": n2_rec,
                    "Tổng cỡ mẫu (n1+n2)": n1_rec + n2_rec,
                    "Tổng tính dự phòng 10%": int(math.ceil((n1_rec + n2_rec) * 1.10))
                }])
                res_df = compact_numeric_df(res_df, decimals=3)
                show_table(res_df, "Kết quả cỡ mẫu so sánh 2 trung bình độc lập")
                download_table_block(res_df, "sample_size_two_means", "Cỡ mẫu 2 trung bình")

        else:
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                mean_d = st.number_input("Hiệu số trung bình trước - sau mong muốn (|μ_d|)", min_value=0.01, value=1.5, step=0.1)
            with c_p2:
                sd_d = st.number_input("Độ lệch chuẩn của hiệu số trước - sau (σ_d)", min_value=0.01, value=2.5, step=0.2)

            if st.button("Tính cỡ mẫu kiểm định bắt cặp", type="primary", use_container_width=True):
                n_pair = ((z_a + z_b) ** 2) * (sd_d ** 2) / (mean_d ** 2)
                n_pair_rec = int(math.ceil(n_pair))

                res_df = pd.DataFrame([{
                    "Hiệu số trung bình (|μ_d|)": mean_d,
                    "Độ lệch chuẩn hiệu số (σ_d)": sd_d,
                    "Alpha (α)": alpha_test,
                    "Power (1-β)": f"{power_test*100:.0f}%",
                    "Số cặp đối tượng (n)": n_pair_rec,
                    "Dự phòng mất dấu 10%": int(math.ceil(n_pair_rec * 1.10))
                }])
                res_df = compact_numeric_df(res_df, decimals=3)
                show_table(res_df, "Kết quả cỡ mẫu kiểm định bắt cặp")
                download_table_block(res_df, "sample_size_paired_means", "Cỡ mẫu bắt cặp")

    elif sub == "Hồi quy":
        reg_type = st.radio("Kiểu mô hình hồi quy", ["Hồi quy Tuyến tính đa biến (Multivariable Linear)", "Hồi quy Logistic đa biến (Multivariable Logistic)"], horizontal=True)

        if reg_type == "Hồi quy Tuyến tính đa biến (Multivariable Linear)":
            c1, c2, c3 = st.columns(3)
            with c1:
                k_preds = st.number_input("Số biến độc lập / tham số dự báo (k)", min_value=1, max_value=50, value=4, step=1)
            with c2:
                r2_target = st.number_input("Hệ số xác định kỳ vọng (R²)", min_value=0.02, max_value=0.80, value=0.15, step=0.02)
            with c3:
                power_reg = st.selectbox("Lực lượng kiểm định (Power)", [0.80, 0.90, 0.85], index=0)

            if st.button("Tính cỡ mẫu hồi quy tuyến tính", type="primary", use_container_width=True):
                f2 = r2_target / (1.0 - r2_target)
                n_green_overall = 50 + 8 * int(k_preds)
                n_green_individual = 104 + int(k_preds)
                z_a = stats.norm.ppf(0.975)
                z_b = stats.norm.ppf(power_reg)
                n_cohen_approx = int(math.ceil(((z_a + z_b) ** 2) / f2 + int(k_preds) + 1))

                res_df = pd.DataFrame([{
                    "Số biến độc lập (k)": int(k_preds),
                    "Hệ số R² kỳ vọng": r2_target,
                    "Cỡ hiệu ứng Cohen's f²": f2,
                    "Quy tắc Green (kiểm định toàn bộ R²)": n_green_overall,
                    "Quy tắc Green (kiểm định từng hệ số β)": n_green_individual,
                    "Cỡ mẫu khuyến nghị (Cohen f²)": max(n_cohen_approx, n_green_overall),
                    "Dự phòng mất mẫu 10%": int(math.ceil(max(n_cohen_approx, n_green_overall) * 1.10))
                }])
                res_df = compact_numeric_df(res_df, decimals=3)
                show_table(res_df, "Cỡ mẫu hồi quy tuyến tính đa biến")
                download_table_block(res_df, "sample_size_linear_regression", "Cỡ mẫu hồi quy tuyến tính")

        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                k_vars = st.number_input("Số biến độc lập đưa vào mô hình (k)", min_value=1, max_value=50, value=5, step=1)
            with c2:
                prev_event = st.number_input("Tỷ lệ xảy ra biến cố trong quần thể (P)", min_value=0.01, max_value=0.50, value=0.15, step=0.01, format="%.3f")
            with c3:
                epv_rule = st.selectbox("Nguyên tắc EPV (Events Per Variable)", [10, 20, 15], index=0)

            if st.button("Tính cỡ mẫu hồi quy Logistic", type="primary", use_container_width=True):
                events_needed = int(k_vars) * int(epv_rule)
                n_logistic = int(math.ceil(events_needed / prev_event))

                res_df = pd.DataFrame([{
                    "Số biến độc lập (k)": int(k_vars),
                    "Tiêu chuẩn EPV": int(epv_rule),
                    "Số biến cố tối thiểu cần có": events_needed,
                    "Tỷ lệ biến cố (P)": prev_event,
                    "Cỡ mẫu tối thiểu cần tuyển (N)": n_logistic,
                    "Dự phòng mất mẫu 10%": int(math.ceil(n_logistic * 1.10))
                }])
                res_df = compact_numeric_df(res_df, decimals=3)
                show_table(res_df, "Cỡ mẫu hồi quy Logistic đa biến (Quy tắc EPV)")
                download_table_block(res_df, "sample_size_logistic_regression", "Cỡ mẫu hồi quy logistic")

    elif sub == "ANOVA":
        c1, c2, c3 = st.columns(3)
        with c1:
            k_groups = st.number_input("Số nhóm so sánh (k)", min_value=3, max_value=12, value=3, step=1)
        with c2:
            f_effect = st.number_input("Cỡ hiệu ứng Cohen's f", min_value=0.05, max_value=0.80, value=0.25, step=0.05)
        with c3:
            power_a = st.selectbox("Lực lượng (Power)", [0.80, 0.90, 0.85, 0.95], index=0)

        alpha_a = 0.05
        if st.button("Tính cỡ mẫu One-way ANOVA", type="primary", use_container_width=True):
            k = int(k_groups)
            df1 = k - 1
            found_n = 5
            for n_trial in range(5, 5000):
                df2 = k * (n_trial - 1)
                nc = n_trial * k * (f_effect ** 2)
                crit_f = stats.f.ppf(1.0 - alpha_a, df1, df2)
                p_calc = 1.0 - stats.ncf.cdf(crit_f, df1, df2, nc)
                if p_calc >= power_a:
                    found_n = n_trial
                    break

            total_n = found_n * k
            res_df = pd.DataFrame([{
                "Số nhóm so sánh (k)": k,
                "Cỡ hiệu ứng Cohen's f": f_effect,
                "Alpha (α)": alpha_a,
                "Lực lượng kỳ vọng": f"{power_a*100:.0f}%",
                "Cỡ mẫu mỗi nhóm (n)": found_n,
                "Tổng cỡ mẫu nghiên cứu (N)": total_n,
                "Tổng tính dự phòng 10%": int(math.ceil(total_n * 1.10))
            }])
            res_df = compact_numeric_df(res_df, decimals=3)
            show_table(res_df, "Kết quả cỡ mẫu One-way ANOVA")
            download_table_block(res_df, "sample_size_anova", "Cỡ mẫu ANOVA")

    elif sub == "Phân tích sống sót":
        st.markdown("#### Phép kiểm Log-rank / Mô hình Cox (Schoenfeld formula)")
        c1, c2, c3 = st.columns(3)
        with c1:
            hr_val = st.number_input("Tỷ số nguy cơ kỳ vọng (Hazard Ratio, HR)", min_value=0.10, max_value=10.0, value=1.75, step=0.05)
        with c2:
            alpha_s = st.selectbox("Mức ý nghĩa (α)", [0.05, 0.01, 0.10], index=0)
        with c3:
            power_s = st.selectbox("Lực lượng (1 - β)", [0.80, 0.90, 0.85], index=0)

        c4, c5 = st.columns(2)
        with c4:
            p_event_total = st.number_input("Tỷ lệ bệnh nhân dự kiến xảy ra biến cố (P_event)", min_value=0.05, max_value=1.0, value=0.50, step=0.05)
        with c5:
            alloc_ratio = st.selectbox("Tỷ lệ phân bổ 2 nhóm", ["1:1 (Đều nhau)"], index=0)

        if st.button("Tính cỡ mẫu phân tích sống sót", type="primary", use_container_width=True):
            if hr_val == 1.0:
                st.error("HR không thể bằng 1.0 (không có hiệu ứng).")
            else:
                z_a = stats.norm.ppf(1.0 - alpha_s / 2.0)
                z_b = stats.norm.ppf(power_s)
                events_needed = ((z_a + z_b) ** 2) * 4.0 / ((math.log(hr_val)) ** 2)
                e_rec = int(math.ceil(events_needed))
                n_total_surv = int(math.ceil(e_rec / p_event_total))
                n_each_group = int(math.ceil(n_total_surv / 2.0))

                res_df = pd.DataFrame([{
                    "Tỷ số nguy cơ (HR)": hr_val,
                    "Alpha (α)": alpha_s,
                    "Power (1-β)": f"{power_s*100:.0f}%",
                    "Số biến cố tối thiểu cần quan sát (E)": e_rec,
                    "Tỷ lệ biến cố ước tính": f"{p_event_total*100:.1f}%",
                    "Mỗi nhóm cần tuyển": n_each_group,
                    "Tổng số đối tượng cần tuyển (N)": n_each_group * 2,
                    "Dự phòng mất dấu 10%": int(math.ceil(n_each_group * 2 * 1.10))
                }])
                res_df = compact_numeric_df(res_df, decimals=3)
                show_table(res_df, "Kết quả cỡ mẫu phân tích sống sót (Log-rank / Cox)")
                download_table_block(res_df, "sample_size_survival", "Cỡ mẫu phân tích sống sót")

# -----------------------------
# TÍNH XÁC SUẤT (PROBABILITY)
# -----------------------------
elif section == "Tính xác suất":
    st.markdown(f"## Tính toán lý thuyết xác suất — {sub}")

    if sub == "Công thức xác suất & Bayes":
        mode_prob = st.radio("Chọn dạng bài toán xác suất", [
            "Công thức Cộng & Nhân (Hai biến cố A và B)",
            "Xác suất Toàn phần & Công thức Bayes"
        ], horizontal=True)

        if mode_prob == "Công thức Cộng & Nhân (Hai biến cố A và B)":
            c1, c2 = st.columns(2)
            with c1:
                p_a = st.number_input("Xác suất P(A)", min_value=0.0, max_value=1.0, value=0.40, step=0.05, format="%.4f")
            with c2:
                p_b = st.number_input("Xác suất P(B)", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.4f")

            rel_type = st.radio("Mối quan hệ giữa A và B", [
                "Độc lập (Independent): P(A ∩ B) = P(A) × P(B)",
                "Xung khắc (Mutually exclusive): P(A ∩ B) = 0",
                "Tùy biến (Nhập trực tiếp giao P(A ∩ B) hoặc P(B|A))"
            ])

            if rel_type.startswith("Độc lập"):
                p_ab = p_a * p_b
            elif rel_type.startswith("Xung khắc"):
                p_ab = 0.0
            else:
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    input_mode_custom = st.selectbox("Nhập theo:", ["P(A ∩ B) - Đồng thời xảy ra", "P(B|A) - Có điều kiện"])
                with c_c2:
                    if input_mode_custom.startswith("P(A ∩ B)"):
                        p_ab = st.number_input("P(A ∩ B)", min_value=0.0, max_value=min(p_a, p_b), value=min(p_a, p_b)*0.5, step=0.02, format="%.4f")
                    else:
                        p_b_given_a = st.number_input("P(B|A)", min_value=0.0, max_value=1.0, value=0.50, step=0.05, format="%.4f")
                        p_ab = p_b_given_a * p_a

            if st.button("Tính toán các xác suất", type="primary", use_container_width=True):
                p_a_or_b = p_a + p_b - p_ab
                p_b_given_a = p_ab / p_a if p_a > 0 else np.nan
                p_a_given_b = p_ab / p_b if p_b > 0 else np.nan
                p_not_a = 1.0 - p_a
                p_not_b = 1.0 - p_b
                p_neither = 1.0 - p_a_or_b

                df_prob = pd.DataFrame([
                    ["P(A)", p_a, "Xác suất của biến cố A"],
                    ["P(B)", p_b, "Xác suất của biến cố B"],
                    ["P(A ∩ B)", p_ab, "Xác suất A và B đồng thời xảy ra (Tích)"],
                    ["P(A ∪ B)", p_a_or_b, "Xác suất ít nhất A hoặc B xảy ra (Cộng)"],
                    ["P(B|A)", p_b_given_a, "Xác suất B xảy ra khi biết A đã xảy ra"],
                    ["P(A|B)", p_a_given_b, "Xác suất A xảy ra khi biết B đã xảy ra"],
                    ["P(A')", p_not_a, "Xác suất biến cố đối của A (không xảy ra A)"],
                    ["P(B')", p_not_b, "Xác suất biến cố đối của B (không xảy ra B)"],
                    ["P(A' ∩ B')", p_neither, "Xác suất cả A và B đều không xảy ra"]
                ], columns=["Phép tính", "Giá trị xác suất", "Ý nghĩa"])

                df_prob = compact_numeric_df(df_prob, decimals=3)
                show_table(df_prob, "Kết quả công thức cộng & nhân xác suất")
                download_table_block(df_prob, "probability_addition_multiplication", "Công thức cộng nhân xác suất")

        else:
            st.markdown("#### Hệ đầy đủ các biến cố $A_1, A_2, ..., A_k$ và biến cố $B$")
            num_hyp = st.number_input("Số biến cố phân hoạch (k)", min_value=2, max_value=5, value=3, step=1)
            k = int(num_hyp)

            st.caption("Nhập xác suất tiên nghiệm P(Ai) [tổng phải bằng 1] và xác suất có điều kiện P(B|Ai):")
            default_priors = [0.5, 0.3, 0.2, 0.0, 0.0][:k]
            if sum(default_priors) != 1.0:
                default_priors = [1.0/k]*k
            default_conds = [0.08, 0.05, 0.02, 0.01, 0.01][:k]

            cols_input = st.columns(k)
            p_prior_list, p_cond_list = [], []
            for i in range(k):
                with cols_input[i]:
                    st.markdown(f"**Nhóm A{i+1}**")
                    p_prior = st.number_input(f"P(A{i+1})", min_value=0.0, max_value=1.0, value=float(default_priors[i]), step=0.05, key=f"prior_{i}", format="%.4f")
                    p_cond = st.number_input(f"P(B|A{i+1})", min_value=0.0, max_value=1.0, value=float(default_conds[i]), step=0.01, key=f"cond_{i}", format="%.4f")
                    p_prior_list.append(p_prior)
                    p_cond_list.append(p_cond)

            sum_priors = sum(p_prior_list)
            if abs(sum_priors - 1.0) > 1e-4:
                st.warning(f"⚠️ Tổng các xác suất tiên nghiệm P(Ai) = {sum_priors:.4f} (phải bằng 1.0). Vui lòng điều chỉnh lại.")

            if st.button("Tính xác suất toàn phần & Công thức Bayes", type="primary", use_container_width=True):
                joint_probs = [p_prior_list[i] * p_cond_list[i] for i in range(k)]
                p_b_total = sum(joint_probs)

                if p_b_total <= 0:
                    st.error("Tổng P(B) = 0, không thể chia để tính công thức Bayes.")
                else:
                    posterior_probs = [j / p_b_total for j in joint_probs]

                    table_rows = []
                    for i in range(k):
                        table_rows.append([
                            f"A{i+1}",
                            p_prior_list[i],
                            p_cond_list[i],
                            joint_probs[i],
                            posterior_probs[i]
                        ])

                    table_rows.append([
                        "Tổng (B)",
                        sum(p_prior_list),
                        "-",
                        p_b_total,
                        sum(posterior_probs)
                    ])

                    bayes_df = pd.DataFrame(table_rows, columns=[
                        "Biến cố (Ai)",
                        "Tiên nghiệm P(Ai)",
                        "Khả năng P(B|Ai)",
                        "Đồng thời P(Ai ∩ B)",
                        "Hậu nghiệm Bayes P(Ai|B)"
                    ])
                    bayes_df = compact_numeric_df(bayes_df, decimals=4)
                    show_table(bayes_df, f"Bảng tính chi tiết định lý Bayes (Xác suất toàn phần P(B) = {smart_round_val(p_b_total, 4)})")
                    download_table_block(bayes_df, "bayes_theorem_results", "Định lý Bayes")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    labels = [f"A{i+1}" for i in range(k)]
                    x_idx = np.arange(k)
                    width = 0.35

                    rects1 = ax.bar(x_idx - width/2, p_prior_list, width, label='Tiên nghiệm P(Ai)', color='#0B3A66')
                    rects2 = ax.bar(x_idx + width/2, posterior_probs, width, label='Hậu nghiệm P(Ai|B)', color='#E63946')

                    ax.set_ylabel('Xác suất', fontsize=12, fontweight='bold')
                    ax.set_title('So sánh xác suất Tiên nghiệm và Hậu nghiệm (Bayes)', fontsize=14, fontweight='bold')
                    ax.set_xticks(x_idx)
                    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)

                    st.pyplot(fig)
                    download_figure_block(fig, "bayes_comparison_chart")
                    plt.close(fig)

    elif sub == "Phân phối Nhị thức B(n, p)":
        st.markdown("#### Biến ngẫu nhiên rời rạc: $X \sim B(n, p)$")
        c1, c2 = st.columns(2)
        with c1:
            n_binom = st.number_input("Số phép thử độc lập (n)", min_value=1, max_value=500, value=20, step=1)
        with c2:
            p_binom = st.number_input("Xác suất thành công trong mỗi phép thử (p)", min_value=0.001, max_value=0.999, value=0.250, step=0.05, format="%.4f")

        n_val = int(n_binom)
        p_val = float(p_binom)
        e_val = n_val * p_val
        var_val = n_val * p_val * (1.0 - p_val)
        sd_val = math.sqrt(var_val)

        desc_binom_df = pd.DataFrame([{
            "Số phép thử (n)": n_val,
            "Xác suất (p)": p_val,
            "Kỳ vọng E(X) = np": e_val,
            "Phương sai Var(X)": var_val,
            "Độ lệch chuẩn σ": sd_val
        }])
        desc_binom_df = compact_numeric_df(desc_binom_df, decimals=3)
        show_table(desc_binom_df, "Đặc trưng của phân phối nhị thức")

        st.markdown("#### Tính toán xác suất biến cố")
        c_k1, c_k2, c_k3 = st.columns(3)
        with c_k1:
            k_target = st.number_input("Giá trị k", min_value=0, max_value=n_val, value=min(5, n_val), step=1)
        with c_k2:
            m_low = st.number_input("Cận dưới m", min_value=0, max_value=n_val, value=min(3, n_val), step=1)
        with c_k3:
            m_high = st.number_input("Cận trên n*", min_value=0, max_value=n_val, value=min(8, n_val), step=1)

        k_val = int(k_target)
        low_val = int(min(m_low, m_high))
        high_val = int(max(m_low, m_high))

        if st.button("Tính xác suất nhị thức & Vẽ đồ thị", type="primary", use_container_width=True):
            p_eq = stats.binom.pmf(k_val, n_val, p_val)
            p_le = stats.binom.cdf(k_val, n_val, p_val)
            p_ge = stats.binom.sf(k_val - 1, n_val, p_val) if k_val > 0 else 1.0
            p_between = stats.binom.cdf(high_val, n_val, p_val) - (stats.binom.cdf(low_val - 1, n_val, p_val) if low_val > 0 else 0.0)

            prob_res_df = pd.DataFrame([
                [f"P(X = {k_val})", p_eq, f"Đúng {k_val} lần thành công"],
                [f"P(X <= {k_val})", p_le, f"Tối đa {k_val} lần thành công"],
                [f"P(X >= {k_val})", p_ge, f"Ít nhất {k_val} lần thành công"],
                [f"P({low_val} <= X <= {high_val})", p_between, f"Số lần thành công nằm trong đoạn [{low_val}, {high_val}]"]
            ], columns=["Biến cố", "Xác suất", "Diễn giải"])

            prob_res_df = compact_numeric_df(prob_res_df, decimals=4)
            show_table(prob_res_df, "Kết quả tính xác suất nhị thức")
            download_table_block(prob_res_df, "binomial_probability_results", "Xác suất nhị thức")

            fig, ax = plt.subplots(figsize=(10, 4.5))
            x_min = max(0, int(e_val - 3.5 * sd_val))
            x_max = min(n_val, int(e_val + 3.5 * sd_val) + 1)
            if x_max - x_min < 12:
                x_min = max(0, min(low_val, k_val) - 3)
                x_max = min(n_val, max(high_val, k_val) + 4)

            x_bars = np.arange(x_min, x_max + 1)
            y_bars = stats.binom.pmf(x_bars, n_val, p_val)
            colors = ['#E63946' if (low_val <= x <= high_val) else '#0B3A66' for x in x_bars]

            bars = ax.bar(x_bars, y_bars, color=colors, width=0.7, edgecolor='#333333', alpha=0.85)
            ax.set_xlabel('Số lần thành công (k)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Xác suất P(X = k)', fontsize=12, fontweight='bold')
            ax.set_title(f'Phân phối nhị thức B(n={n_val}, p={p_val}) — Tô đỏ khoảng [{low_val}; {high_val}]', fontsize=14, fontweight='bold')
            ax.set_xticks(x_bars)
            ax.grid(axis='y', linestyle='--', alpha=0.4)

            st.pyplot(fig)
            download_figure_block(fig, "binomial_distribution_chart")
            plt.close(fig)

    else:
        st.markdown("#### Biến ngẫu nhiên liên tục: $X \sim N(\mu, \sigma^2)$")
        c1, c2 = st.columns(2)
        with c1:
            mean_norm = st.number_input("Trung bình (Mean, μ)", value=100.00, format="%.3f")
        with c2:
            disp_norm_choice = st.radio("Chọn tham số độ phân tán để nhập:", ["Độ lệch chuẩn (σ)", "Phương sai (σ²)"], horizontal=True)
            if disp_norm_choice.startswith("Độ lệch chuẩn"):
                sd_norm = st.number_input("Độ lệch chuẩn (σ)", min_value=0.0001, value=15.00, step=1.0, format="%.3f")
                var_norm = sd_norm ** 2
                st.caption(f"Phương sai tương ứng ($σ^2$): **{var_norm:.3f}**")
            else:
                var_norm = st.number_input("Phương sai (σ²)", min_value=0.0001, value=225.00, step=5.0, format="%.3f")
                sd_norm = math.sqrt(var_norm)
                st.caption(f"Độ lệch chuẩn tương ứng ($σ$): **{sd_norm:.3f}**")

        mu_v = float(mean_norm)
        sigma_v = float(sd_norm)

        st.markdown("#### Thiết lập các mốc giá trị cần tính")
        c_k1, c_k2, c_k3 = st.columns(3)
        with c_k1:
            k_norm = st.number_input("Mốc giá trị k", value=mu_v + sigma_v, format="%.3f")
        with c_k2:
            m_norm_low = st.number_input("Cận dưới m", value=mu_v - sigma_v, format="%.3f")
        with c_k3:
            m_norm_high = st.number_input("Cận trên n*", value=mu_v + sigma_v, format="%.3f")

        k_norm_val = float(k_norm)
        m_low_val = float(min(m_norm_low, m_norm_high))
        m_high_val = float(max(m_norm_low, m_norm_high))

        plot_option = st.selectbox("Chọn vùng tô màu trên đồ thị hình chuông:", [
            f"Đoạn giữa m và n*: P({m_low_val:.2f} <= X <= {m_high_val:.2f})",
            f"Vùng bên trái: P(X <= {k_norm_val:.2f})",
            f"Vùng bên phải: P(X >= {k_norm_val:.2f})"
        ])

        if st.button("Tính xác suất phân phối chuẩn & Vẽ đồ thị", type="primary", use_container_width=True):
            z_k = (k_norm_val - mu_v) / sigma_v
            z_m = (m_low_val - mu_v) / sigma_v
            z_n = (m_high_val - mu_v) / sigma_v

            density_k = stats.norm.pdf(k_norm_val, loc=mu_v, scale=sigma_v)
            p_norm_le = stats.norm.cdf(k_norm_val, loc=mu_v, scale=sigma_v)
            p_norm_ge = 1.0 - p_norm_le
            p_norm_between = stats.norm.cdf(m_high_val, loc=mu_v, scale=sigma_v) - stats.norm.cdf(m_low_val, loc=mu_v, scale=sigma_v)

            norm_calc_df = pd.DataFrame([
                [f"P(X = {k_norm_val:.3f})", 0.000, density_k, f"Xác suất tại 1 điểm bằng 0 (Mật độ f(k) = {density_k:.4f})"],
                [f"P(X <= {k_norm_val:.3f})", p_norm_le, z_k, f"Xác suất tích lũy bên trái (Z = {z_k:.3f})"],
                [f"P(X >= {k_norm_val:.3f})", p_norm_ge, z_k, f"Xác suất phần đuôi bên phải (Z = {z_k:.3f})"],
                [f"P({m_low_val:.3f} <= X <= {m_high_val:.3f})", p_norm_between, f"Z1 = {z_m:.3f}, Z2 = {z_n:.3f}", f"Xác suất nằm trong khoảng [{m_low_val:.3f}, {m_high_val:.3f}]"]
            ], columns=["Biến cố", "Xác suất", "Điểm Z / Mật độ", "Ghi chú"])

            norm_calc_df = compact_numeric_df(norm_calc_df, decimals=4)
            show_table(norm_calc_df, "Kết quả tính xác suất phân phối chuẩn")
            download_table_block(norm_calc_df, "normal_probability_results", "Xác suất phân phối chuẩn")

            fig, ax = plt.subplots(figsize=(10, 4.5))
            x_axis = np.linspace(mu_v - 3.8 * sigma_v, mu_v + 3.8 * sigma_v, 1000)
            y_axis = stats.norm.pdf(x_axis, loc=mu_v, scale=sigma_v)

            ax.plot(x_axis, y_axis, color='#0B3A66', linewidth=2.5, label=f'Đường cong Gauss N(μ={mu_v:.1f}, σ={sigma_v:.1f})')

            if plot_option.startswith("Đoạn giữa"):
                x_fill = np.linspace(m_low_val, m_high_val, 500)
                y_fill = stats.norm.pdf(x_fill, loc=mu_v, scale=sigma_v)
                ax.fill_between(x_fill, y_fill, color='#E63946', alpha=0.5, label=f'P({m_low_val:.2f} <= X <= {m_high_val:.2f}) = {p_norm_between:.4f}')
                ax.axvline(m_low_val, color='#E63946', linestyle='--', linewidth=1.5)
                ax.axvline(m_high_val, color='#E63946', linestyle='--', linewidth=1.5)
            elif plot_option.startswith("Vùng bên trái"):
                x_fill = np.linspace(mu_v - 3.8 * sigma_v, k_norm_val, 500)
                y_fill = stats.norm.pdf(x_fill, loc=mu_v, scale=sigma_v)
                ax.fill_between(x_fill, y_fill, color='#2A9D8F', alpha=0.5, label=f'P(X <= {k_norm_val:.2f}) = {p_norm_le:.4f}')
                ax.axvline(k_norm_val, color='#2A9D8F', linestyle='--', linewidth=1.5)
            else:
                x_fill = np.linspace(k_norm_val, mu_v + 3.8 * sigma_v, 500)
                y_fill = stats.norm.pdf(x_fill, loc=mu_v, scale=sigma_v)
                ax.fill_between(x_fill, y_fill, color='#F4A261', alpha=0.5, label=f'P(X >= {k_norm_val:.2f}) = {p_norm_ge:.4f}')
                ax.axvline(k_norm_val, color='#F4A261', linestyle='--', linewidth=1.5)

            ax.axvline(mu_v, color='#666666', linestyle=':', linewidth=1.2, label=f'Trung bình μ = {mu_v:.1f}')
            ax.set_xlabel('Giá trị X', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mật độ xác suất f(x)', fontsize=12, fontweight='bold')
            ax.set_title('Biểu diễn diện tích xác suất trên phân phối chuẩn', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(axis='both', linestyle='--', alpha=0.3)

            st.pyplot(fig)
            download_figure_block(fig, "normal_distribution_chart")
            plt.close(fig)

# -----------------------------
# MÔ-ĐUN MỚI: AI TRỢ LÝ THÔNG MINH
# -----------------------------
# -----------------------------
# MÔ-ĐUN: AI TRỢ LÝ THÔNG MINH
# -----------------------------
elif section == "AI Trợ lý" and sub == "Giải toán & Trắc nghiệm":
    st.markdown("## 🤖 AI Trợ lý: Giải bài toán Thống kê Y sinh & Tạo trắc nghiệm")
    st.write("Dán văn bản hoặc dán/tải ảnh chụp bài toán. AI sẽ tự động nhận diện dạng toán, giải chi tiết (KTC + Kiểm định) và tạo bộ câu hỏi trắc nghiệm 4 phương án A, B, C, D.")

    # Import thư viện dán ảnh
    try:
        from streamlit_paste_button import paste_image_button
        has_paste = True
    except ImportError:
        has_paste = False

    c1, c2 = st.columns(2)
    with c1:
        txt_input = st.text_area(
            "Nhập hoặc dán văn bản đề bài:",
            value="Một nghiên cứu đánh giá một chương trình can thiệp kiểm soát đái tháo đường. Sau can thiệp, nhóm can thiệp có n1 = 40 đối tượng, x̄1 = 6.8% và s1 = 0.9%; nhóm chứng có n2 = 40 đối tượng, x̄2 = 7.4% và s2 = 1%. Nhà nghiên cứu muốn ước lượng hiệu trung bình bằng KTC 95%.",
            height=140
        )
    with c2:
        st.markdown("**Ảnh chụp đề bài:**")
        img_from_clipboard = None
        if has_paste:
            paste_result = paste_image_button(
                label="📋 Bấm vào đây để Dán ảnh từ Clipboard (Ctrl + V)",
                text_color="#ffffff",
                background_color="#0B3A66",
                hover_background_color="#1E40AF"
            )
            if paste_result.image_data is not None:
                img_from_clipboard = paste_result.image_data

        img_file = st.file_uploader("Hoặc tải ảnh từ máy tính (PNG, JPG):", type=["png", "jpg", "jpeg"])

        # Ưu tiên lấy ảnh dán từ clipboard, nếu không có thì lấy ảnh tải lên
        final_image = None
        if img_from_clipboard is not None:
            final_image = img_from_clipboard
            st.image(final_image, caption="Đã nhận ảnh dán từ Clipboard", use_container_width=True)
        elif img_file is not None:
            final_image = Image.open(img_file)
            st.image(final_image, caption="Đã nhận ảnh tải lên từ máy tính", use_container_width=True)

    c_cfg1, c_cfg2 = st.columns(2)
    with c_cfg1:
        action_mode = st.radio(
            "Chọn yêu cầu xử lý:",
            ["Giải bài toán chi tiết (KTC + Kiểm định)", "Tạo câu hỏi trắc nghiệm A, B, C, D", "Cả giải chi tiết và tạo trắc nghiệm"],
            horizontal=False
        )
    with c_cfg2:
        num_questions = st.number_input("Số lượng câu trắc nghiệm cần tạo:", min_value=1, max_value=20, value=4, step=1)

    if st.button("🚀 Bắt đầu phân tích với AI", type="primary", use_container_width=True):
        if not txt_input.strip() and final_image is None:
            st.warning("Vui lòng nhập văn bản đề bài hoặc dán/tải ảnh lên.")
        elif "GEMINI_API_KEY" not in st.secrets:
            st.error("Chưa cấu hình GEMINI_API_KEY trong Settings > Secrets của Streamlit Cloud.")
        else:
            status_box = st.empty()
        with status_box.status("⏳ Đang kết nối AI và phân tích...", expanded=True) as status:
            try:
                # 1. Tự động nén/thu nhỏ ảnh nếu dung lượng quá lớn để gửi đi siêu tốc
                processed_image = None
                if final_image is not None:
                    status.write("🖼️ Đang xử lý và tối ưu ảnh...")
                    img_copy = final_image.copy()
                    if max(img_copy.size) > 1200:
                        img_copy.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                    processed_image = img_copy

                # 2. Xây dựng prompt chuẩn xác
                prompt = f"""
Bạn là chuyên gia Thống kê Y học và giảng viên bộ môn Xác suất Thống kê Y Dược.
Nhiệm vụ: Nhận diện và giải quyết bài toán theo nội dung văn bản hoặc ảnh đính kèm.

Yêu cầu thực hiện ({action_mode}):
1. Nhận diện dạng toán (so sánh 2 trung bình độc lập, bắt cặp, tỷ lệ, kiểm định hay KTC...).
2. Trình bày bài giải chi tiết từng bước: Các giả thuyết H0/H1, sai số chuẩn (SE), giá trị thống kê kiểm định (t hoặc Z), bậc tự do df, p-value, Khoảng tin cậy KTC 95%, và kết luận ý nghĩa y học lâm sàng rõ ràng.
3. Nếu có tạo câu hỏi trắc nghiệm: Hãy tạo đúng {num_questions} câu hỏi 4 lựa chọn (A, B, C, D), có đáp án đúng và lời giải thích ngắn gọn cho mỗi câu.
"""
                parts = [prompt]
                if txt_input.strip():
                    parts.append(f"ĐỀ BÀI:\n{txt_input}")
                if processed_image is not None:
                    parts.append(processed_image)

                status.write("🧠 Đang tính toán và truyền dòng kết quả...")

                # 3. Kết nối trực tiếp vào model khả dụng trên AI Studio
                # Danh sách model thử nghiệm theo thứ tự ưu tiên
                candidate_models = ["gemini-3-flash-preview", "gemini-3.8-flash"]
                response = None

                for m_name in candidate_models:
                    try:
                        model = genai.GenerativeModel(m_name)
                        # Bật stream=True để AI sinh chữ tới đâu đẩy về màn hình tới đó
                        response = model.generate_content(parts, stream=True)
                        break
                    except Exception:
                        continue

                if response is None:
                    raise RuntimeError("Không thể kết nối với mô hình Gemini. Vui lòng kiểm tra lại API Key.")

                status.update(label="✅ Đã nhận diện xong đề bài!", state="complete", expanded=False)

                st.markdown("---")
                st.markdown("### 📋 Kết quả phân tích & Lời giải từ AI:")

                # 4. Hiển thị chữ chạy theo thời gian thực (Real-time Stream)
                def stream_output():
                    collected_text = ""
                    for chunk in response:
                        if chunk.text:
                            collected_text += chunk.text
                            yield chunk.text
                    st.session_state["ai_saved_result"] = collected_text

                st.write_stream(stream_output)

                # Nút tải kết quả về máy
                if "ai_saved_result" in st.session_state and st.session_state["ai_saved_result"]:
                    st.download_button(
                        "📥 Tải nội dung lời giải & trắc nghiệm (.txt)",
                        data=st.session_state["ai_saved_result"],
                        file_name="loi_giai_va_trac_nghiem.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                status.update(label="❌ Có lỗi xảy ra!", state="error", expanded=True)
                st.error(f"Lỗi chi tiết: {e}")
