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
    def _reg_parse_excel_to_df(text: str) -> pd.DataFrame:
        import re
        if not text or not str(text).strip():
            return pd.DataFrame()
        raw_lines = [ln for ln in str(text).splitlines() if ln.strip()]
        if not raw_lines:
            return pd.DataFrame()

        sample_line = raw_lines[0]
        if "\t" in sample_line:
            sep = "\t"
        elif ";" in sample_line:
            sep = ";"
        elif "," in sample_line:
            sep = ","
        else:
            sep = r"\s+"

        rows = []
        for ln in raw_lines:
            if sep == r"\s+":
                parts = [p.strip() for p in re.split(r"\s+", ln.strip()) if p.strip()]
            else:
                parts = [p.strip() for p in ln.split(sep)]
            rows.append(parts)

        max_cols = max(len(r) for r in rows)
        if max_cols < 2:
            return pd.DataFrame()
        rows = [r + [""] * (max_cols - len(r)) for r in rows]

        def _is_num(val_s: str) -> bool:
            if val_s == "":
                return False
            try:
                float(val_s.replace(",", "."))
                return True
            except ValueError:
                return False

        first_row = rows[0]
        non_empty_first = [c for c in first_row if c != ""]
        num_in_first = sum(1 for c in non_empty_first if _is_num(c))
        alpha_in_first = sum(1 for c in non_empty_first if re.search(r"[A-Za-zÀ-ỹ]", c))

        if len(rows) >= 2 and alpha_in_first >= 1 and num_in_first == 0:
            raw_headers = [first_row[j] if first_row[j] else f"Var_{j+1}" for j in range(max_cols)]
            seen = {}
            headers = []
            for h in raw_headers:
                if h in seen:
                    seen[h] += 1
                    headers.append(f"{h}_{seen[h]}")
                else:
                    seen[h] = 0
                    headers.append(h)
            data_rows = rows[1:]
        else:
            headers = [f"Var_{j+1}" for j in range(max_cols)]
            data_rows = rows

        df_out = pd.DataFrame(data_rows, columns=headers)
        missing_tokens = {"", "NA", "N/A", "na", "n/a", "nan", "NaN", "null", "NULL", ".", "-"}

        for col in df_out.columns:
            s_col = df_out[col].astype(str).str.strip()
            s_clean = s_col.replace(list(missing_tokens), np.nan)
            non_na = s_clean.dropna()
            if len(non_na) == 0:
                df_out[col] = s_clean
                continue
            s_num_try = non_na.str.replace(",", ".", regex=False)
            converted = pd.to_numeric(s_num_try, errors="coerce")
            if converted.notna().sum() >= max(1, 0.8 * len(non_na)):
                df_out[col] = pd.to_numeric(s_clean.str.replace(",", ".", regex=False), errors="coerce")
            else:
                df_out[col] = s_clean

        return df_out

    def _reg_data_input_panel(template_df: pd.DataFrame, template_name: str, state_key: str, name_key: str):
        mode_key = f"{state_key}_input_mode"
        paste_key = f"{state_key}_paste_text"
        if mode_key not in st.session_state:
            st.session_state[mode_key] = "📋 Dán dữ liệu từ Excel"
        if paste_key not in st.session_state:
            st.session_state[paste_key] = ""

        src_mode = st.radio(
            "Nguồn dữ liệu:",
            ["📋 Dán dữ liệu từ Excel", "📁 Tải file dữ liệu (XLSX/CSV)"],
            horizontal=True,
            key=mode_key
        )

        if src_mode == "📋 Dán dữ liệu từ Excel":
            txt_pasted = st.text_area(
                "Quét chọn vùng bảng trong Excel (bao gồm cả hàng tiêu đề tên biến ở dòng đầu) rồi dán vào đây:",
                height=160,
                key=paste_key,
                placeholder="Ví dụ:\ntarget_binary\tCRP\tWBC\n0\t10.2\t7.1\n1\t35.1\t12.5\n0\t8.7\t6.8\n1\t22.4\t10.3"
            )
            c_act1, c_act2 = st.columns([3, 1])
            with c_act1:
                st.button("📥 Nạp / Cập nhật dữ liệu đã dán", type="secondary", use_container_width=True, key=f"{state_key}_btn_apply_paste")
            with c_act2:
                if st.button("🗑️ Xóa dữ liệu dán", use_container_width=True, key=f"{state_key}_btn_clear_paste"):
                    st.session_state[paste_key] = ""
                    st.session_state[state_key] = None
                    st.session_state[name_key] = ""
                    st.rerun()

            if txt_pasted and txt_pasted.strip():
                parsed_df = _reg_parse_excel_to_df(txt_pasted)
                if not parsed_df.empty and parsed_df.shape[1] >= 2:
                    st.session_state[state_key] = parsed_df
                    st.session_state[name_key] = "Dữ liệu dán từ Excel"
                    st.caption(f"✅ Đã nhận diện bảng dữ liệu: **{parsed_df.shape[0]} hàng × {parsed_df.shape[1]} cột** ({', '.join(map(str, parsed_df.columns))})")
                    with st.expander("Xem trước dữ liệu đã dán (20 dòng đầu)", expanded=False):
                        st.dataframe(parsed_df.head(20), use_container_width=True)
                else:
                    st.warning("⚠️ Chưa nhận diện được bảng hợp lệ (cần ít nhất 2 cột phân tách bằng Tab từ Excel, dấu chấm phẩy ';' hoặc dấu phẩy ',').")
        else:
            data_input_panel(template_df, template_name, state_key, name_key)

    def _build_logistic_conclusion(tbl: pd.DataFrame, target_name: str) -> str:
        if tbl is None or tbl.empty:
            return ""
        cols = list(tbl.columns)
        term_col = cols[0]
        p_col = next((c for c in cols if "sig" in c.lower() or "p>" in c.lower() or c.lower() in ["p", "p-value", "pvalue"]), None)
        or_col = next((c for c in cols if "exp" in c.lower() or c.lower() in ["or", "odds ratio", "aor"]), None)
        ci_l_col = next((c for c in cols if "lower" in c.lower() or "2.5%" in c.lower()), None)
        ci_u_col = next((c for c in cols if "upper" in c.lower() or "97.5%" in c.lower()), None)
        if not p_col:
            return ""

        sig_items, nonsig_items = [], []
        for _, row in tbl.iterrows():
            term = str(row[term_col]).strip()
            if term.lower() in ["intercept", "constant", "(intercept)", "_cons"]:
                continue
            p_raw = str(row[p_col]).strip()
            is_sig = False
            if p_raw.startswith("<"):
                is_sig = True
                p_txt = f"p {p_raw}"
            else:
                try:
                    is_sig = (float(p_raw.replace(",", ".")) < 0.05)
                    p_txt = f"p = {p_raw}"
                except ValueError:
                    p_txt = f"p = {p_raw}"

            detail = f"**{term}** ("
            if or_col and str(row[or_col]).strip() != "":
                detail += f"OR = {row[or_col]}"
                if ci_l_col and ci_u_col and str(row[ci_l_col]).strip() != "" and str(row[ci_u_col]).strip() != "":
                    detail += f", KTC 95%: {row[ci_l_col]} – {row[ci_u_col]}"
                detail += f", {p_txt})"
            else:
                detail += f"{p_txt})"

            if is_sig:
                sig_items.append(detail)
            else:
                nonsig_items.append(detail)

        lines = [f"**Kết luận mô hình hồi quy Logistic (biến kết cục: {target_name}):**"]
        if sig_items:
            lines.append(f"- **Các yếu tố liên quan có ý nghĩa thống kê (p < 0,05):** {'; '.join(sig_items)}.")
        else:
            lines.append("- **Không ghi nhận yếu tố độc lập nào có liên quan có ý nghĩa thống kê** với biến kết cục (p ≥ 0,05).")
        if nonsig_items:
            lines.append(f"- **Các yếu tố chưa có ý nghĩa thống kê (p ≥ 0,05):** {'; '.join(nonsig_items)}.")
        return "\n\n".join(lines)

    logistic_template = pd.DataFrame({
        "target_binary": [0, 1, 0, 1],
        "CRP": [10.2, 35.1, 8.7, 22.4],
        "WBC": [7.1, 12.5, 6.8, 10.3],
    })
    if sub == "Data":
        st.markdown("## Logistic Regression — Data")
        _reg_data_input_panel(logistic_template, "logistic_template", LOGISTIC_KEY, "df_logistic_name")
    elif sub == "EDA":
        st.markdown("## Logistic Regression — EDA")
        _reg_data_input_panel(logistic_template, "logistic_template", LOGISTIC_KEY, "df_logistic_name")
        df = require_df(LOGISTIC_KEY)
        miss = df.isna().sum().reset_index()
        miss.columns = ["Variable", "Missing"]
        show_table(miss, "Missing Values")
        download_table_block(miss, "logistic_missing_values", "Missing Values")
    elif sub == "Modeling":
        st.markdown("## Logistic Regression — Modeling")
        _reg_data_input_panel(logistic_template, "logistic_template", LOGISTIC_KEY, "df_logistic_name")
        df = require_df(LOGISTIC_KEY)
        cols = list(df.columns)
        target = st.selectbox("Target column (binary 0/1)", options=cols)
        features = st.multiselect("Predictors (numeric / categorical)", options=[c for c in cols if c != target])
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

                conc_log = _build_logistic_conclusion(res["table"], str(target))
                if conc_log:
                    st.success(conc_log)
            except Exception as e:
                st.error(f"Modeling failed: {e}")
    elif sub == "Export":
        st.markdown("## Logistic Regression — Export")
        st.info("Each table/figure includes download and copy buttons.")
# -----------------------------
# LINEAR REGRESSION
# -----------------------------
elif section == "Linear Regression":
    def _reg_parse_excel_to_df(text: str) -> pd.DataFrame:
        import re
        if not text or not str(text).strip():
            return pd.DataFrame()
        raw_lines = [ln for ln in str(text).splitlines() if ln.strip()]
        if not raw_lines:
            return pd.DataFrame()

        sample_line = raw_lines[0]
        if "\t" in sample_line:
            sep = "\t"
        elif ";" in sample_line:
            sep = ";"
        elif "," in sample_line:
            sep = ","
        else:
            sep = r"\s+"

        rows = []
        for ln in raw_lines:
            if sep == r"\s+":
                parts = [p.strip() for p in re.split(r"\s+", ln.strip()) if p.strip()]
            else:
                parts = [p.strip() for p in ln.split(sep)]
            rows.append(parts)

        max_cols = max(len(r) for r in rows)
        if max_cols < 2:
            return pd.DataFrame()
        rows = [r + [""] * (max_cols - len(r)) for r in rows]

        def _is_num(val_s: str) -> bool:
            if val_s == "":
                return False
            try:
                float(val_s.replace(",", "."))
                return True
            except ValueError:
                return False

        first_row = rows[0]
        non_empty_first = [c for c in first_row if c != ""]
        num_in_first = sum(1 for c in non_empty_first if _is_num(c))
        alpha_in_first = sum(1 for c in non_empty_first if re.search(r"[A-Za-zÀ-ỹ]", c))

        if len(rows) >= 2 and alpha_in_first >= 1 and num_in_first == 0:
            raw_headers = [first_row[j] if first_row[j] else f"Var_{j+1}" for j in range(max_cols)]
            seen = {}
            headers = []
            for h in raw_headers:
                if h in seen:
                    seen[h] += 1
                    headers.append(f"{h}_{seen[h]}")
                else:
                    seen[h] = 0
                    headers.append(h)
            data_rows = rows[1:]
        else:
            headers = [f"Var_{j+1}" for j in range(max_cols)]
            data_rows = rows

        df_out = pd.DataFrame(data_rows, columns=headers)
        missing_tokens = {"", "NA", "N/A", "na", "n/a", "nan", "NaN", "null", "NULL", ".", "-"}

        for col in df_out.columns:
            s_col = df_out[col].astype(str).str.strip()
            s_clean = s_col.replace(list(missing_tokens), np.nan)
            non_na = s_clean.dropna()
            if len(non_na) == 0:
                df_out[col] = s_clean
                continue
            s_num_try = non_na.str.replace(",", ".", regex=False)
            converted = pd.to_numeric(s_num_try, errors="coerce")
            if converted.notna().sum() >= max(1, 0.8 * len(non_na)):
                df_out[col] = pd.to_numeric(s_clean.str.replace(",", ".", regex=False), errors="coerce")
            else:
                df_out[col] = s_clean

        return df_out

    def _reg_data_input_panel(template_df: pd.DataFrame, template_name: str, state_key: str, name_key: str):
        mode_key = f"{state_key}_input_mode"
        paste_key = f"{state_key}_paste_text"
        if mode_key not in st.session_state:
            st.session_state[mode_key] = "📋 Dán dữ liệu từ Excel"
        if paste_key not in st.session_state:
            st.session_state[paste_key] = ""

        src_mode = st.radio(
            "Nguồn dữ liệu:",
            ["📋 Dán dữ liệu từ Excel", "📁 Tải file dữ liệu (XLSX/CSV)"],
            horizontal=True,
            key=mode_key
        )

        if src_mode == "📋 Dán dữ liệu từ Excel":
            txt_pasted = st.text_area(
                "Quét chọn vùng bảng trong Excel (bao gồm cả hàng tiêu đề tên biến ở dòng đầu) rồi dán vào đây:",
                height=160,
                key=paste_key,
                placeholder="Ví dụ:\nY_outcome\tCRP\tWBC\n10.2\t10.2\t7.1\n12.1\t35.1\t12.5\n9.8\t8.7\t6.8\n14.0\t22.4\t10.3"
            )
            c_act1, c_act2 = st.columns([3, 1])
            with c_act1:
                st.button("📥 Nạp / Cập nhật dữ liệu đã dán", type="secondary", use_container_width=True, key=f"{state_key}_btn_apply_paste")
            with c_act2:
                if st.button("🗑️ Xóa dữ liệu dán", use_container_width=True, key=f"{state_key}_btn_clear_paste"):
                    st.session_state[paste_key] = ""
                    st.session_state[state_key] = None
                    st.session_state[name_key] = ""
                    st.rerun()

            if txt_pasted and txt_pasted.strip():
                parsed_df = _reg_parse_excel_to_df(txt_pasted)
                if not parsed_df.empty and parsed_df.shape[1] >= 2:
                    st.session_state[state_key] = parsed_df
                    st.session_state[name_key] = "Dữ liệu dán từ Excel"
                    st.caption(f"✅ Đã nhận diện bảng dữ liệu: **{parsed_df.shape[0]} hàng × {parsed_df.shape[1]} cột** ({', '.join(map(str, parsed_df.columns))})")
                    with st.expander("Xem trước dữ liệu đã dán (20 dòng đầu)", expanded=False):
                        st.dataframe(parsed_df.head(20), use_container_width=True)
                else:
                    st.warning("⚠️ Chưa nhận diện được bảng hợp lệ (cần ít nhất 2 cột phân tách bằng Tab từ Excel, dấu chấm phẩy ';' hoặc dấu phẩy ',').")
        else:
            data_input_panel(template_df, template_name, state_key, name_key)

    def _build_linear_conclusion(b_tbl: pd.DataFrame, y_name: str, r2_val=None, adj_r2_val=None) -> str:
        if b_tbl is None or b_tbl.empty:
            return ""
        sig_items, nonsig_items = [], []
        for _, row in b_tbl.iterrows():
            term = str(row["Term"]).strip()
            if term.lower() in ["intercept", "constant", "(intercept)", "_cons"]:
                continue
            p_raw = str(row["Sig."]).strip()
            p_txt = f"p {p_raw}" if p_raw.startswith("<") else f"p = {p_raw}"
            is_sig = (str(row.get("Significant (p<0.05)", "")).strip().lower() == "yes")
            detail = f"**{term}** (B = {row['B']}, KTC 95%: {row['CI 2.5%']} – {row['CI 97.5%']}, {p_txt})"
            if is_sig:
                sig_items.append(detail)
            else:
                nonsig_items.append(detail)

        r2_note = ""
        if r2_val is not None and adj_r2_val is not None:
            r2_note = f" (R² = {r2_val:.3f}, R² hiệu chỉnh = {adj_r2_val:.3f})"

        lines = [f"**Kết luận mô hình hồi quy tuyến tính đa biến (biến phụ thuộc: {y_name}){r2_note}:**"]
        if sig_items:
            lines.append(f"- **Các yếu tố có mối liên quan tuyến tính có ý nghĩa thống kê (p < 0,05):** {'; '.join(sig_items)}.")
        else:
            lines.append("- **Không ghi nhận biến độc lập nào có mối liên quan có ý nghĩa thống kê** với biến phụ thuộc (p ≥ 0,05).")
        if nonsig_items:
            lines.append(f"- **Các yếu tố chưa có ý nghĩa thống kê (p ≥ 0,05):** {'; '.join(nonsig_items)}.")
        return "\n\n".join(lines)

    linear_template = pd.DataFrame({
        "Y_outcome": [10.2, 12.1, 9.8, 14.0],
        "CRP": [10.2, 35.1, 8.7, 22.4],
        "WBC": [7.1, 12.5, 6.8, 10.3],
    })
    if sub == "Data":
        st.markdown("## Linear Regression — Data")
        _reg_data_input_panel(linear_template, "linear_template", LINEAR_KEY, "df_linear_name")
    elif sub == "Diagnostics":
        st.markdown("## Linear Regression — Assumptions & Diagnostics")
        _reg_data_input_panel(linear_template, "linear_template", LINEAR_KEY, "df_linear_name")
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
                bp_p = np.nan
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

                diag_notes = []
                if not np.isnan(sh_p):
                    diag_notes.append(
                        f"- **Phân phối chuẩn phần dư:** {'Đạt chuẩn (Shapiro-Wilk p = ' + format_p_value(sh_p) + ' > 0,05)' if sh_p > 0.05 else 'Không đạt phân phối chuẩn (Shapiro-Wilk p = ' + format_p_value(sh_p) + ' ≤ 0,05)'}."
                    )
                if not np.isnan(bp_p):
                    diag_notes.append(
                        f"- **Phương sai sai số đồng nhất:** {'Đạt đồng nhất (Breusch-Pagan p = ' + format_p_value(bp_p) + ' ≥ 0,05)' if bp_p >= 0.05 else 'Phương sai thay đổi (Breusch-Pagan p = ' + format_p_value(bp_p) + ' < 0,05)'}."
                    )
                if vif_rows:
                    max_vif = max(r[1] for r in vif_rows)
                    diag_notes.append(
                        f"- **Đa cộng tuyến (VIF lớn nhất = {max_vif:.3f}):** {'Không có đa cộng tuyến nghiêm trọng (VIF < 5)' if max_vif < 5 else 'Cảnh báo đa cộng tuyến (VIF ≥ 5)'}."
                    )
                if diag_notes:
                    st.info("**Đánh giá các giả định hồi quy tuyến tính:**\n\n" + "\n\n".join(diag_notes))
            except Exception as e:
                st.error(f"Diagnostics failed: {e}")
    elif sub == "Modeling":
        st.markdown("## Linear Regression — Modeling")
        _reg_data_input_panel(linear_template, "linear_template", LINEAR_KEY, "df_linear_name")
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

                r2_v = float(getattr(model, "rsquared", np.nan))
                adj_r2_v = float(getattr(model, "rsquared_adj", np.nan))
                conc_lin = _build_linear_conclusion(
                    b,
                    str(y_col),
                    None if np.isnan(r2_v) else r2_v,
                    None if np.isnan(adj_r2_v) else adj_r2_v
                )
                if conc_lin:
                    st.success(conc_lin)
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
    import re

    def _qt_try_float(tok: str):
        if tok is None:
            return None
        s = str(tok).strip()
        if s == "" or s.lower() in ["na", "n/a", "nan", "null", "none", ".", "-", "kph", "không"]:
            return None
        try:
            val = float(s.replace(",", "."))
            return None if (np.isnan(val) or np.isinf(val)) else val
        except ValueError:
            return None

    def _qt_smart_parse_1d(text: str) -> np.ndarray:
        if not text or not str(text).strip():
            return np.array([], dtype=float)
        raw_lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
        if not raw_lines:
            return np.array([], dtype=float)

        if len(raw_lines) >= 2:
            first_ln = raw_lines[0]
            if not re.search(r"[\t;]", first_ln) and _qt_try_float(first_ln) is None:
                parts_test = re.split(r"[,\s]+", first_ln)
                num_count = sum(1 for p in parts_test if _qt_try_float(p) is not None)
                if num_count <= 1 and re.search(r"[A-Za-zÀ-ỹ]", first_ln):
                    raw_lines = raw_lines[1:]

        s = "\n".join(raw_lines)
        if ";" in s:
            tokens = re.split(r"[;\s\n\t]+", s)
            return np.array([v for v in (_qt_try_float(t) for t in tokens) if v is not None], dtype=float)

        if "," in s:
            if "." in s:
                s = s.replace(",", " ")
            else:
                parts_ws = re.split(r"[\s\n\t]+", s)
                if len(parts_ws) > 1 and all(re.match(r"^[-+]?\d+,\d+$", t) for t in parts_ws if t):
                    s = s.replace(",", ".")
                elif len(parts_ws) == 1 and re.match(r"^[-+]?\d+,\d+$", parts_ws[0]):
                    s = s.replace(",", ".")
                else:
                    s = s.replace(",", " ")

        tokens = re.split(r"[\s\n\t]+", s)
        return np.array([v for v in (_qt_try_float(t) for t in tokens) if v is not None], dtype=float)

    def _qt_smart_parse_table(text: str, listwise: bool = False, allow_long_format: bool = True):
        if not text or not str(text).strip():
            return [], []
        raw_lines = [ln for ln in str(text).splitlines() if ln.strip()]
        if not raw_lines:
            return [], []

        def split_row(ln: str):
            if "\t" in ln:
                return [p.strip() for p in ln.split("\t")]
            elif ";" in ln:
                return [p.strip() for p in ln.split(";")]
            elif "," in ln:
                p_c = [p.strip() for p in ln.split(",")]
                if len(p_c) >= 2:
                    return p_c
            return [p.strip() for p in re.split(r"\s+", ln.strip()) if p.strip()]

        rows = [split_row(ln) for ln in raw_lines]
        max_cols = max(len(r) for r in rows)
        if max_cols < 2:
            return [], []

        rows = [r + [""] * (max_cols - len(r)) for r in rows]

        first_row = rows[0]
        first_non_empty = [c for c in first_row if c != ""]
        first_num_cnt = sum(1 for c in first_non_empty if _qt_try_float(c) is not None)
        first_alpha_cnt = sum(1 for c in first_non_empty if re.search(r"[A-Za-zÀ-ỹ]", c))

        if len(rows) >= 2 and first_alpha_cnt >= 1 and first_num_cnt == 0:
            headers = [first_row[j] if first_row[j] else f"Col {j+1}" for j in range(max_cols)]
            data_rows = rows[1:]
        else:
            headers = [f"Group {j+1}" for j in range(max_cols)]
            data_rows = rows

        if not data_rows:
            return [], []

        id_header_pat = re.compile(r"^(stt|id|mã(\s*bn|\s*số)?|code|subject|bn|no\.?|#)$", re.IGNORECASE)
        col_is_numeric = []
        for j in range(max_cols):
            if id_header_pat.match(headers[j].strip()):
                col_is_numeric.append(False)
                continue
            non_empty_vals = [r[j] for r in data_rows if r[j] != ""]
            if not non_empty_vals:
                col_is_numeric.append(False)
                continue
            num_cnt = sum(1 for v in non_empty_vals if _qt_try_float(v) is not None)
            col_is_numeric.append(num_cnt >= max(1, 0.5 * len(non_empty_vals)))

        num_col_indices = [j for j, is_num in enumerate(col_is_numeric) if is_num]
        text_col_indices = [j for j, is_num in enumerate(col_is_numeric) if not is_num]

        if allow_long_format and len(num_col_indices) == 1 and len(text_col_indices) >= 1:
            val_idx = num_col_indices[0]
            grp_idx = text_col_indices[-1]
            for idx_t in text_col_indices:
                if id_header_pat.match(headers[idx_t].strip()):
                    continue
                distinct_vals = set(r[idx_t] for r in data_rows if r[idx_t] != "")
                if 2 <= len(distinct_vals) < len(data_rows):
                    grp_idx = idx_t
                    break

            grouped = {}
            for r in data_rows:
                g_name = r[grp_idx].strip()
                v_num = _qt_try_float(r[val_idx])
                if g_name != "" and v_num is not None:
                    grouped.setdefault(g_name, []).append(v_num)

            out_names = list(grouped.keys())
            return out_names, [np.array(grouped[k], dtype=float) for k in out_names]

        if len(num_col_indices) >= 2:
            sel_headers = [headers[j] for j in num_col_indices]
            cols_data = [[] for _ in num_col_indices]
            for r in data_rows:
                parsed_row = [_qt_try_float(r[j]) for j in num_col_indices]
                if listwise:
                    if all(v is not None for v in parsed_row):
                        for k_i, v in enumerate(parsed_row):
                            cols_data[k_i].append(v)
                else:
                    for k_i, v in enumerate(parsed_row):
                        if v is not None:
                            cols_data[k_i].append(v)
            return sel_headers, [np.array(c, dtype=float) for c in cols_data]

        return [], []

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
        import re
        from scipy import stats

        # Hàm định dạng số: số nguyên hiện nguyên; mặc định 3 chữ số thập phân; nếu 3 số bằng 0 thì lấy 4-5 số
        def _fmt_t_val(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            if isinstance(val, (int, np.integer)):
                return str(int(val))
            if isinstance(val, (float, np.floating)):
                v = float(val)
                if abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                if abs(round(v, 3)) > 0:
                    return f"{v:.3f}"
                if v == 0.0:
                    return "0"
                if abs(round(v, 4)) > 0:
                    return f"{v:.4f}"
                if abs(round(v, 5)) > 0:
                    return f"{v:.5f}"
                return "< 0.00001"
            return str(val)

        def _fmt_t_df(df_in: pd.DataFrame) -> pd.DataFrame:
            df_out = df_in.copy()
            for col in df_out.columns:
                df_out[col] = df_out[col].apply(_fmt_t_val)
            return df_out

        # Hàm kiểm định phân phối chuẩn Shapiro-Wilk theo định dạng bảng SPSS (Tests of Normality)
        def _shapiro_test_table(samples: list):
            rows = []
            notes = []
            for label, arr in samples:
                clean = np.asarray(arr, dtype=float)
                clean = clean[~np.isnan(clean)]
                n = len(clean)
                if n < 3:
                    rows.append({
                        "Variable / Group": label,
                        "Shapiro-Wilk Statistic": "N/A (N < 3)",
                        "df": _fmt_t_val(n),
                        "Sig.": "N/A"
                    })
                    notes.append(f"{label}: N < 3 (cần tối thiểu 3 quan sát để kiểm định Shapiro-Wilk)")
                elif np.ptp(clean) == 0:
                    rows.append({
                        "Variable / Group": label,
                        "Shapiro-Wilk Statistic": "1",
                        "df": _fmt_t_val(n),
                        "Sig.": "1"
                    })
                    notes.append(f"{label}: dữ liệu hằng số")
                else:
                    w_stat, p_val = stats.shapiro(clean)
                    rows.append({
                        "Variable / Group": label,
                        "Shapiro-Wilk Statistic": _fmt_t_val(float(w_stat)),
                        "df": _fmt_t_val(n),
                        "Sig.": _fmt_t_val(float(p_val))
                    })
                    if p_val > 0.05:
                        notes.append(f"{label}: Sig. = {_fmt_t_val(float(p_val))} > 0.05 (phân phối chuẩn)")
                    else:
                        notes.append(f"{label}: Sig. = {_fmt_t_val(float(p_val))} ≤ 0.05 (không phân phối chuẩn)")
            return pd.DataFrame(rows), " | ".join(notes)

        # Gọi bộ phân giải bảng thông minh đã khai báo ở đầu Quantitative Tests
        def _parse_1d_nums(text: str) -> np.ndarray:
            return _qt_smart_parse_1d(text)

        def _parse_2cols_nums(text: str, paired: bool = False):
            _, cols = _qt_smart_parse_table(text, listwise=paired, allow_long_format=(not paired))
            if len(cols) >= 2:
                return cols[0], cols[1]
            return np.array([], dtype=float), np.array([], dtype=float)

        st.markdown("## Quantitative Tests — t Tests")
        input_source = st.radio(
            "Nguồn dữ liệu:",
            ["Dán / Nhập dữ liệu trực tiếp", "Tải file dữ liệu (XLSX/CSV)"],
            horizontal=True,
            key="ttest_input_source"
        )

        # ======================================================================
        # CHẾ ĐỘ 1: DÁN / NHẬP DỮ LIỆU TRỰC TIẾP
        # ======================================================================
        if input_source == "Dán / Nhập dữ liệu trực tiếp":
            test_type_paste = st.radio(
                "Test type",
                ["One-sample t test", "Paired-samples t test", "Independent-samples t test"],
                horizontal=True,
                key="ttest_paste_type"
            )

            # 1. ONE-SAMPLE T TEST
            if test_type_paste == "One-sample t test":
                txt_1s = st.text_area(
                    "Dán dữ liệu mẫu (cách nhau bằng khoảng trắng, dấu chấm phẩy ';' hoặc dấu phẩy ','):",
                    height=120,
                    key="ttest_1s_paste"
                )
                mu_val = st.number_input("Giá trị cần kiểm định (Test value μ₀):", value=0.0, format="%.3f", key="ttest_1s_mu")

                if st.button("Run One-Sample t Test", type="primary", use_container_width=True, key="btn_paste_1s"):
                    arr = _parse_1d_nums(txt_1s)
                    if len(arr) < 2:
                        st.warning("Vui lòng nhập ít nhất 2 giá trị số hợp lệ.")
                    else:
                        n = len(arr)
                        mean_v = float(np.mean(arr))
                        sd_v = float(np.std(arr, ddof=1))
                        se_v = sd_v / np.sqrt(n)
                        diff_v = mean_v - mu_val
                        df_v = n - 1
                        t_v = diff_v / se_v if se_v > 0 else 0.0
                        p_v = float(2 * (1 - stats.t.cdf(abs(t_v), df_v)))
                        t_crit = float(stats.t.ppf(0.975, df_v))
                        ci_low = diff_v - t_crit * se_v
                        ci_high = diff_v + t_crit * se_v

                        desc_tbl = _fmt_t_df(pd.DataFrame([{
                            "N": n,
                            "Mean": mean_v,
                            "SD": sd_v,
                            "SE Mean": se_v,
                            "Test Value (μ₀)": mu_val
                        }]))
                        norm_tbl, norm_note = _shapiro_test_table([("Sample", arr)])
                        res_tbl = _fmt_t_df(pd.DataFrame([{
                            "t": t_v,
                            "df": df_v,
                            "Sig. (2-tailed)": p_v,
                            "Mean Difference": diff_v,
                            "95% CI Lower": ci_low,
                            "95% CI Upper": ci_high
                        }]))

                        show_table(desc_tbl, "One-Sample Statistics")
                        download_table_block(desc_tbl, "one_sample_desc", "One-Sample Statistics")
                        show_table(norm_tbl, "Tests of Normality (Shapiro-Wilk)")
                        st.caption(norm_note)
                        download_table_block(norm_tbl, "one_sample_normality", "Tests of Normality")
                        show_table(res_tbl, "One-Sample Test")
                        download_table_block(res_tbl, "one_sample_ttest", "One-Sample t Test")

            # 2. PAIRED-SAMPLES T TEST (d = Sau - Trước)
            elif test_type_paste == "Paired-samples t test":
                pair_mode = st.radio(
                    "Cách nhập dữ liệu ghép cặp:",
                    [
                        "Dán 2 cột Trước, Sau (chép từ 2 cột Excel dán vào 1 ô)",
                        "Dán 2 ô riêng biệt (Trước & Sau)",
                        "Dán dãy hiệu số chênh lệch d (d = Sau - Trước)"
                    ],
                    horizontal=True,
                    key="ttest_pair_mode"
                )

                arr_pre, arr_post, arr_d = None, None, None
                diff_label = "d = After - Before"

                if pair_mode == "Dán 2 cột Trước, Sau (chép từ 2 cột Excel dán vào 1 ô)":
                    txt_2col_pair = st.text_area(
                        "Dán 2 cột từ Excel (Cột 1: Trước, Cột 2: Sau):",
                        height=140,
                        key="ttest_pair_2col"
                    )
                    arr_pre, arr_post = _parse_2cols_nums(txt_2col_pair, paired=True)
                elif pair_mode == "Dán 2 ô riêng biệt (Trước & Sau)":
                    c1, c2 = st.columns(2)
                    with c1:
                        txt_pre = st.text_area("Dữ liệu Trước (Before):", height=120, key="ttest_pair_pre")
                    with c2:
                        txt_post = st.text_area("Dữ liệu Sau (After):", height=120, key="ttest_pair_post")
                    arr_pre = _parse_1d_nums(txt_pre)
                    arr_post = _parse_1d_nums(txt_post)
                else:
                    txt_d = st.text_area(
                        "Dán dãy hiệu số d = Sau - Trước (cách nhau khoảng trắng, dấu chấm phẩy ';' hoặc dấu phẩy ','):",
                        height=120,
                        key="ttest_pair_d"
                    )
                    arr_d = _parse_1d_nums(txt_d)

                if st.button("Run Paired-Samples t Test", type="primary", use_container_width=True, key="btn_paste_pair"):
                    valid = False
                    if pair_mode != "Dán dãy hiệu số chênh lệch d (d = Sau - Trước)":
                        if len(arr_pre) != len(arr_post):
                            st.error(f"Số lượng quan sát không khớp: Cột Trước có {len(arr_pre)} số, Cột Sau có {len(arr_post)} số.")
                        elif len(arr_pre) < 2:
                            st.warning("Vui lòng nhập ít nhất 2 cặp dữ liệu hợp lệ.")
                        else:
                            # Sự khác biệt trước sau: d = Sau - Trước
                            arr_d = arr_post - arr_pre
                            valid = True
                    else:
                        if arr_d is None or len(arr_d) < 2:
                            st.warning("Vui lòng nhập ít nhất 2 giá trị hiệu số d hợp lệ.")
                        else:
                            valid = True

                    if valid:
                        n_p = len(arr_d)
                        mean_d = float(np.mean(arr_d))
                        sd_d = float(np.std(arr_d, ddof=1))
                        se_d = sd_d / np.sqrt(n_p)
                        df_p = n_p - 1
                        t_p = mean_d / se_d if se_d > 0 else 0.0
                        p_p = float(2 * (1 - stats.t.cdf(abs(t_p), df_p)))
                        t_crit_p = float(stats.t.ppf(0.975, df_p))
                        ci_low_p = mean_d - t_crit_p * se_d
                        ci_high_p = mean_d + t_crit_p * se_d

                        if arr_pre is not None and arr_post is not None and len(arr_pre) == n_p:
                            desc_pair = _fmt_t_df(pd.DataFrame([
                                {
                                    "Variable": "Before",
                                    "N": n_p,
                                    "Mean": float(np.mean(arr_pre)),
                                    "SD": float(np.std(arr_pre, ddof=1)),
                                    "SE Mean": float(np.std(arr_pre, ddof=1) / np.sqrt(n_p))
                                },
                                {
                                    "Variable": "After",
                                    "N": n_p,
                                    "Mean": float(np.mean(arr_post)),
                                    "SD": float(np.std(arr_post, ddof=1)),
                                    "SE Mean": float(np.std(arr_post, ddof=1) / np.sqrt(n_p))
                                }
                            ]))
                            show_table(desc_pair, "Paired Samples Statistics")
                            download_table_block(desc_pair, "ttest_paired_desc", "Paired Samples Statistics")

                        # Kiểm định Shapiro-Wilk cho biến hiệu số d = Sau - Trước
                        norm_pair, norm_pair_note = _shapiro_test_table([(diff_label, arr_d)])
                        show_table(norm_pair, "Tests of Normality (Shapiro-Wilk)")
                        st.caption(norm_pair_note)
                        download_table_block(norm_pair, "ttest_paired_normality", "Tests of Normality")

                        res_pair = _fmt_t_df(pd.DataFrame([{
                            "Pair": diff_label,
                            "N": n_p,
                            "Mean": mean_d,
                            "SD": sd_d,
                            "SE Mean": se_d,
                            "95% CI Lower": ci_low_p,
                            "95% CI Upper": ci_high_p,
                            "t": t_p,
                            "df": df_p,
                            "Sig. (2-tailed)": p_p
                        }]))
                        show_table(res_pair, "Paired Samples Test")
                        download_table_block(res_pair, "ttest_paired_results", "Paired Samples Test")

            # 3. INDEPENDENT-SAMPLES T TEST
            else:
                ind_mode = st.radio(
                    "Cách nhập dữ liệu 2 mẫu độc lập:",
                    [
                        "Dán 2 ô riêng biệt (Mẫu 1 & Mẫu 2)",
                        "Dán 2 cột tương ứng từ Excel vào 1 ô"
                    ],
                    horizontal=True,
                    key="ttest_ind_mode"
                )

                if ind_mode == "Dán 2 ô riêng biệt (Mẫu 1 & Mẫu 2)":
                    c1, c2 = st.columns(2)
                    with c1:
                        txt_g1 = st.text_area(
                            "Dữ liệu Mẫu 1 (cách nhau khoảng trắng, ';', ','):",
                            height=120,
                            key="ttest_ind_g1"
                        )
                    with c2:
                        txt_g2 = st.text_area(
                            "Dữ liệu Mẫu 2 (cách nhau khoảng trắng, ';', ','):",
                            height=120,
                            key="ttest_ind_g2"
                        )
                    s1 = _parse_1d_nums(txt_g1)
                    s2 = _parse_1d_nums(txt_g2)
                else:
                    txt_2col_ind = st.text_area(
                        "Dán 2 cột tương ứng từ Excel (Cột 1: Mẫu 1, Cột 2: Mẫu 2):",
                        height=140,
                        key="ttest_ind_2col"
                    )
                    s1, s2 = _parse_2cols_nums(txt_2col_ind, paired=False)

                if st.button("Run Independent-Samples t Test", type="primary", use_container_width=True, key="btn_paste_ind"):
                    if len(s1) < 2 or len(s2) < 2:
                        st.warning("Vui lòng nhập ít nhất 2 giá trị hợp lệ cho mỗi mẫu.")
                    else:
                        n1, n2 = len(s1), len(s2)
                        m1, m2 = float(np.mean(s1)), float(np.mean(s2))
                        sd1, sd2 = float(np.std(s1, ddof=1)), float(np.std(s2, ddof=1))
                        se1, se2 = sd1 / np.sqrt(n1), sd2 / np.sqrt(n2)
                        diff_m = m1 - m2

                        f_lev, p_lev = stats.levene(s1, s2)

                        # Equal variances assumed (Student)
                        sp2 = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
                        se_pool = float(np.sqrt(sp2 * (1 / n1 + 1 / n2)))
                        df_eq = n1 + n2 - 2
                        t_eq = diff_m / se_pool if se_pool > 0 else 0.0
                        p_eq = float(2 * (1 - stats.t.cdf(abs(t_eq), df_eq)))
                        tc_eq = float(stats.t.ppf(0.975, df_eq))
                        ci_eq_l, ci_eq_u = diff_m - tc_eq * se_pool, diff_m + tc_eq * se_pool

                        # Equal variances not assumed (Welch)
                        se_welch = float(np.sqrt(sd1**2 / n1 + sd2**2 / n2))
                        df_uneq = float((sd1**2 / n1 + sd2**2 / n2)**2 / ((sd1**2 / n1)**2 / (n1 - 1) + (sd2**2 / n2)**2 / (n2 - 1)))
                        t_uneq = diff_m / se_welch if se_welch > 0 else 0.0
                        p_uneq = float(2 * (1 - stats.t.cdf(abs(t_uneq), df_uneq)))
                        tc_uneq = float(stats.t.ppf(0.975, df_uneq))
                        ci_uneq_l, ci_uneq_u = diff_m - tc_uneq * se_welch, diff_m + tc_uneq * se_welch

                        grp_tbl = _fmt_t_df(pd.DataFrame([
                            {"Group": "Sample 1", "N": n1, "Mean": m1, "SD": sd1, "SE Mean": se1},
                            {"Group": "Sample 2", "N": n2, "Mean": m2, "SD": sd2, "SE Mean": se2}
                        ]))
                        # Kiểm định Shapiro-Wilk cho cả 2 mẫu độc lập
                        norm_ind, norm_ind_note = _shapiro_test_table([("Sample 1", s1), ("Sample 2", s2)])
                        lev_tbl = _fmt_t_df(pd.DataFrame([
                            {"Levene F": float(f_lev), "Sig.": float(p_lev)}
                        ]))
                        t_tbl = _fmt_t_df(pd.DataFrame([
                            {
                                "Assumption": "Equal variances assumed",
                                "t": t_eq,
                                "df": df_eq,
                                "Sig. (2-tailed)": p_eq,
                                "Mean Difference": diff_m,
                                "SE Difference": se_pool,
                                "95% CI Lower": ci_eq_l,
                                "95% CI Upper": ci_eq_u
                            },
                            {
                                "Assumption": "Equal variances not assumed",
                                "t": t_uneq,
                                "df": df_uneq,
                                "Sig. (2-tailed)": p_uneq,
                                "Mean Difference": diff_m,
                                "SE Difference": se_welch,
                                "95% CI Lower": ci_uneq_l,
                                "95% CI Upper": ci_uneq_u
                            }
                        ]))

                        show_table(grp_tbl, "Group Statistics")
                        download_table_block(grp_tbl, "ttest_ind_desc", "Group Statistics")
                        show_table(norm_ind, "Tests of Normality (Shapiro-Wilk)")
                        st.caption(norm_ind_note)
                        download_table_block(norm_ind, "ttest_ind_normality", "Tests of Normality")
                        show_table(lev_tbl, "Test of Homogeneity of Variances")
                        download_table_block(lev_tbl, "ttest_ind_levene", "Homogeneity of Variances")
                        show_table(t_tbl, "Independent Samples Test")
                        download_table_block(t_tbl, "ttest_ind_results", "Independent Samples Test")

        # ======================================================================
        # CHẾ ĐỘ 2: TẢI FILE DỮ LIỆU (GIỮ NGUYÊN CHỨC NĂNG CŨ + BỔ SUNG SHAPIRO-WILK)
        # ======================================================================
        else:
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
                        norm_t, norm_note = _shapiro_test_table([(str(v_col), arr)])
                        res_t = _fmt_t_df(one_sample_ttest_table(arr, mu_val))
                        show_table(norm_t, "Tests of Normality (Shapiro-Wilk)")
                        st.caption(norm_note)
                        download_table_block(norm_t, "one_sample_normality", "Tests of Normality")
                        show_table(res_t, "One-Sample Test")
                        download_table_block(res_t, "one_sample_ttest", "One-Sample t Test")
                elif test_type == "Independent-samples t test":
                    v_col = st.selectbox("Test variable", numeric_cols)
                    g_col = st.selectbox("Grouping variable", categorical_candidate_cols(df, exclude=[v_col]))
                    g1, g2 = level_selector(df, g_col, "ind_ttest")
                    if st.button("Run Independent-Samples t Test", type="primary", use_container_width=True):
                        d = df[df[g_col].astype(str).isin([g1, g2])].copy()
                        s1_up = numeric_series_from_df(d[d[g_col].astype(str) == str(g1)], v_col).values
                        s2_up = numeric_series_from_df(d[d[g_col].astype(str) == str(g2)], v_col).values
                        norm_ind_up, norm_ind_up_note = _shapiro_test_table([
                            (f"{g_col} = {g1}", s1_up),
                            (f"{g_col} = {g2}", s2_up)
                        ])
                        groups, lev_tbl, t_tbl = independent_ttest_tables(d, v_col, g_col)
                        desc_g = _fmt_t_df(descriptives_for_groups(groups))
                        lev_f = _fmt_t_df(lev_tbl)
                        t_f = _fmt_t_df(t_tbl)
                        show_table(desc_g, "Group Statistics")
                        download_table_block(desc_g, "ttest_ind_desc", "Group Statistics")
                        show_table(norm_ind_up, "Tests of Normality (Shapiro-Wilk)")
                        st.caption(norm_ind_up_note)
                        download_table_block(norm_ind_up, "ttest_ind_normality", "Tests of Normality")
                        show_table(lev_f, "Test of Homogeneity of Variances")
                        download_table_block(lev_f, "ttest_ind_levene", "Homogeneity of Variances")
                        show_table(t_f, "Independent Samples Test")
                        download_table_block(t_f, "ttest_ind_results", "Independent Samples Test")
                else:
                    c1, c2 = st.columns(2)
                    with c1: col1 = st.selectbox("Variable 1 (Before)", numeric_cols)
                    with c2: col2 = st.selectbox("Variable 2 (After)", [c for c in numeric_cols if c != col1])
                    if st.button("Run Paired-Samples t Test", type="primary", use_container_width=True):
                        d = paired_numeric_data(df, col1, col2)
                        arr_d_up = (d[col2] - d[col1]).values
                        norm_pair_up, norm_pair_up_note = _shapiro_test_table([(f"d = {col2} - {col1}", arr_d_up)])
                        groups, t_tbl = paired_ttest_table(d, col1, col2)
                        t_f = _fmt_t_df(t_tbl)
                        show_table(norm_pair_up, "Tests of Normality (Shapiro-Wilk)")
                        st.caption(norm_pair_up_note)
                        download_table_block(norm_pair_up, "ttest_paired_normality", "Tests of Normality")
                        show_table(t_f, "Paired Samples Test")
                        download_table_block(t_f, "ttest_paired_results", "Paired Samples Test")

    elif sub == "Nonparametric Tests":
        import re
        from scipy import stats

        # Hàm định dạng số chuẩn: nguyên hiện nguyên; mặc định 3 số thập phân; nếu 3 số bằng 0 thì lấy 4-5 số
        def _fmt_np_val(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            if isinstance(val, (int, np.integer)):
                return str(int(val))
            if isinstance(val, (float, np.floating)):
                v = float(val)
                if abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                if abs(round(v, 3)) > 0:
                    return f"{v:.3f}"
                if v == 0.0:
                    return "0"
                if abs(round(v, 4)) > 0:
                    return f"{v:.4f}"
                if abs(round(v, 5)) > 0:
                    return f"{v:.5f}"
                return "< 0.00001"
            return str(val)

        def _fmt_np_df(df_in: pd.DataFrame) -> pd.DataFrame:
            df_out = df_in.copy()
            for col in df_out.columns:
                df_out[col] = df_out[col].apply(_fmt_np_val)
            return df_out

        def _fmt_p_text(p_val: float) -> str:
            s = _fmt_np_val(p_val)
            return f"p {s}" if s.startswith("<") else f"p = {s}"

        # Gọi bộ phân giải bảng thông minh đã khai báo ở đầu Quantitative Tests
        def _np_parse_1d(text: str) -> np.ndarray:
            return _qt_smart_parse_1d(text)

        def _np_parse_multicol(text: str, listwise: bool = False):
            return _qt_smart_parse_table(text, listwise=listwise, allow_long_format=(not listwise))

        # Bảng thống kê mô tả phi tham số (Median & IQR)
        def _np_desc_table(named_arrays: list, label_col: str = "Group") -> pd.DataFrame:
            rows = []
            for name, arr in named_arrays:
                clean = np.asarray(arr, dtype=float)
                clean = clean[~np.isnan(clean)]
                if len(clean) == 0:
                    continue
                q25, med, q75 = np.percentile(clean, [25, 50, 75])
                rows.append({
                    label_col: name,
                    "N": len(clean),
                    "Median": float(med),
                    "Q1 (25th)": float(q25),
                    "Q3 (75th)": float(q75),
                    "IQR (Q3 - Q1)": float(q75 - q25),
                    "Min": float(np.min(clean)),
                    "Max": float(np.max(clean))
                })
            return _fmt_np_df(pd.DataFrame(rows))

        # Hàm chạy & hiển thị kết quả từng loại kiểm định phi tham số
        def _run_mann_whitney(name1: str, s1: np.ndarray, name2: str, s2: np.ndarray):
            n1, n2 = len(s1), len(s2)
            N = n1 + n2
            u_stat, p_u = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            combined = np.concatenate([s1, s2])
            ranks = stats.rankdata(combined)
            r1, r2 = ranks[:n1], ranks[n1:]
            sum_r1, sum_r2 = float(np.sum(r1)), float(np.sum(r2))
            w_stat = min(sum_r1, sum_r2)

            _, counts = np.unique(combined, return_counts=True)
            tie_corr = float(np.sum(counts**3 - counts))
            var_u = (n1 * n2 / 12.0) * ((N + 1) - tie_corr / (N * (N - 1))) if N > 1 else 0.0
            sd_u = np.sqrt(var_u) if var_u > 0 else 0.0
            z_u = float((u_stat - (n1 * n2 / 2.0)) / sd_u) if sd_u > 0 else 0.0

            desc_tbl = _np_desc_table([(name1, s1), (name2, s2)], "Group")
            ranks_tbl = _fmt_np_df(pd.DataFrame([
                {"Group": name1, "N": n1, "Mean Rank": float(np.mean(r1)), "Sum of Ranks": sum_r1},
                {"Group": name2, "N": n2, "Mean Rank": float(np.mean(r2)), "Sum of Ranks": sum_r2}
            ]))
            test_tbl = _fmt_np_df(pd.DataFrame([{
                "Mann-Whitney U": float(u_stat),
                "Wilcoxon W": float(w_stat),
                "Z": z_u,
                "Asymp. Sig. (2-tailed)": float(p_u)
            }]))

            show_table(desc_tbl, "Descriptive Statistics (Median & IQR)")
            download_table_block(desc_tbl, "mann_whitney_desc", "Descriptive Statistics")
            show_table(ranks_tbl, "Ranks")
            download_table_block(ranks_tbl, "mann_whitney_ranks", "Ranks")
            show_table(test_tbl, "Test Statistics (Mann-Whitney U)")
            download_table_block(test_tbl, "mann_whitney_results", "Mann-Whitney U Test")

            q1_1, med_1, q3_1 = np.percentile(s1, [25, 50, 75])
            q1_2, med_2, q3_2 = np.percentile(s2, [25, 50, 75])
            p_str = _fmt_p_text(float(p_u))
            summary_stats = (
                f"Trung vị (KTPV) của {name1} là {_fmt_np_val(med_1)} ({_fmt_np_val(q1_1)} – {_fmt_np_val(q3_1)}) "
                f"so với {name2} là {_fmt_np_val(med_2)} ({_fmt_np_val(q1_2)} – {_fmt_np_val(q3_2)})."
            )
            if p_u < 0.05:
                st.success(f"**Kết luận:** Sự khác biệt giữa hai nhóm có ý nghĩa thống kê (kiểm định Mann-Whitney U, {p_str}). {summary_stats}")
            else:
                st.info(f"**Kết luận:** Sự khác biệt giữa hai nhóm không có ý nghĩa thống kê (kiểm định Mann-Whitney U, {p_str}). {summary_stats}")

        def _run_kruskal_wallis(groups_list: list):
            arrays = [arr for _, arr in groups_list]
            names = [nm for nm, _ in groups_list]
            kw_h, p_kw = stats.kruskal(*arrays)
            df_kw = len(arrays) - 1

            combined = np.concatenate(arrays)
            all_ranks = stats.rankdata(combined)
            rank_rows = []
            idx_start = 0
            for nm, arr in groups_list:
                n_g = len(arr)
                g_ranks = all_ranks[idx_start: idx_start + n_g]
                idx_start += n_g
                rank_rows.append({"Group": nm, "N": n_g, "Mean Rank": float(np.mean(g_ranks))})

            desc_tbl = _np_desc_table(groups_list, "Group")
            ranks_tbl = _fmt_np_df(pd.DataFrame(rank_rows))
            test_tbl = _fmt_np_df(pd.DataFrame([{
                "Kruskal-Wallis H": float(kw_h),
                "df": df_kw,
                "Asymp. Sig.": float(p_kw)
            }]))

            show_table(desc_tbl, "Descriptive Statistics (Median & IQR)")
            download_table_block(desc_tbl, "kruskal_desc", "Descriptive Statistics")
            show_table(ranks_tbl, "Ranks")
            download_table_block(ranks_tbl, "kruskal_ranks", "Ranks")
            show_table(test_tbl, "Test Statistics (Kruskal-Wallis)")
            download_table_block(test_tbl, "kruskal_results", "Kruskal-Wallis Test")

            p_str = _fmt_p_text(float(p_kw))
            if p_kw < 0.05:
                st.success(f"**Kết luận:** Sự khác biệt giữa các nhóm ({', '.join(names)}) có ý nghĩa thống kê (kiểm định Kruskal-Wallis, H = {_fmt_np_val(kw_h)}, df = {df_kw}, {p_str}).")
            else:
                st.info(f"**Kết luận:** Sự khác biệt giữa các nhóm ({', '.join(names)}) không có ý nghĩa thống kê (kiểm định Kruskal-Wallis, H = {_fmt_np_val(kw_h)}, df = {df_kw}, {p_str}).")

        def _run_wilcoxon_paired(arr_d: np.ndarray, arr_pre: np.ndarray = None, arr_post: np.ndarray = None,
                                 label_pre: str = "Before", label_post: str = "After", label_d: str = "d = After - Before"):
            n_tot = len(arr_d)
            nz_mask = (arr_d != 0)
            nz_d = arr_d[nz_mask]
            n_nz = len(nz_d)
            ties_n = int(np.sum(~nz_mask))

            if n_nz == 0:
                z_w = 0.0
                p_w = 1.0
                neg_n, pos_n = 0, 0
                neg_mean, neg_sum = 0.0, 0.0
                pos_mean, pos_sum = 0.0, 0.0
            else:
                abs_ranks = stats.rankdata(np.abs(nz_d))
                neg_r = abs_ranks[nz_d < 0]
                pos_r = abs_ranks[nz_d > 0]
                neg_n, pos_n = len(neg_r), len(pos_r)
                neg_sum = float(np.sum(neg_r)) if neg_n > 0 else 0.0
                pos_sum = float(np.sum(pos_r)) if pos_n > 0 else 0.0
                neg_mean = float(np.mean(neg_r)) if neg_n > 0 else 0.0
                pos_mean = float(np.mean(pos_r)) if pos_n > 0 else 0.0

                w_res = stats.wilcoxon(nz_d, alternative="two-sided")
                p_w = float(w_res.pvalue)

                _, t_counts = np.unique(np.abs(nz_d), return_counts=True)
                t_corr = float(np.sum(t_counts**3 - t_counts))
                var_w = (n_nz * (n_nz + 1) * (2 * n_nz + 1) - 0.5 * t_corr) / 24.0
                sd_w = np.sqrt(var_w) if var_w > 0 else 0.0
                mean_w = n_nz * (n_nz + 1) / 4.0
                z_abs = abs(min(neg_sum, pos_sum) - mean_w) / sd_w if sd_w > 0 else 0.0
                z_w = -z_abs if pos_sum < neg_sum else z_abs

            desc_items = []
            if arr_pre is not None and arr_post is not None and len(arr_pre) == n_tot:
                desc_items.extend([(label_pre, arr_pre), (label_post, arr_post)])
            desc_items.append((label_d, arr_d))
            desc_tbl = _np_desc_table(desc_items, "Variable")

            ranks_tbl = _fmt_np_df(pd.DataFrame([
                {"Rank Group": f"Negative Ranks ({label_post} < {label_pre})", "N": neg_n, "Mean Rank": neg_mean, "Sum of Ranks": neg_sum},
                {"Rank Group": f"Positive Ranks ({label_post} > {label_pre})", "N": pos_n, "Mean Rank": pos_mean, "Sum of Ranks": pos_sum},
                {"Rank Group": f"Ties ({label_post} = {label_pre})", "N": ties_n, "Mean Rank": "", "Sum of Ranks": ""},
                {"Rank Group": "Total", "N": n_tot, "Mean Rank": "", "Sum of Ranks": ""}
            ]))

            test_tbl = _fmt_np_df(pd.DataFrame([{
                "Z": float(z_w),
                "Asymp. Sig. (2-tailed)": float(p_w)
            }]))

            show_table(desc_tbl, "Descriptive Statistics (Median & IQR)")
            download_table_block(desc_tbl, "wilcoxon_desc", "Descriptive Statistics")
            show_table(ranks_tbl, "Ranks")
            download_table_block(ranks_tbl, "wilcoxon_ranks", "Ranks")
            show_table(test_tbl, "Test Statistics (Wilcoxon Signed-Rank)")
            download_table_block(test_tbl, "wilcoxon_results", "Wilcoxon Signed-Rank Test")

            p_str = _fmt_p_text(float(p_w))
            q1_d, med_d, q3_d = np.percentile(arr_d, [25, 50, 75])
            diff_note = f"Trung vị (KTPV) của chênh lệch {label_d} là {_fmt_np_val(med_d)} ({_fmt_np_val(q1_d)} – {_fmt_np_val(q3_d)})."
            if p_w < 0.05:
                st.success(f"**Kết luận:** Sự khác biệt giữa hai thời điểm Trước và Sau có ý nghĩa thống kê (kiểm định Wilcoxon Signed-Rank, Z = {_fmt_np_val(z_w)}, {p_str}). {diff_note}")
            else:
                st.info(f"**Kết luận:** Sự khác biệt giữa hai thời điểm Trước và Sau không có ý nghĩa thống kê (kiểm định Wilcoxon Signed-Rank, Z = {_fmt_np_val(z_w)}, {p_str}). {diff_note}")

        def _run_friedman(cond_list: list):
            arrays = [arr for _, arr in cond_list]
            names = [nm for nm, _ in cond_list]
            n_subj = len(arrays[0])
            k_cond = len(arrays)
            fr_stat, p_fr = stats.friedmanchisquare(*arrays)
            df_fr = k_cond - 1

            mat = np.column_stack(arrays)
            row_ranks = np.apply_along_axis(stats.rankdata, 1, mat)
            mean_ranks = np.mean(row_ranks, axis=0)

            desc_tbl = _np_desc_table(cond_list, "Condition / Time")
            ranks_tbl = _fmt_np_df(pd.DataFrame([
                {"Condition / Time": names[j], "Mean Rank": float(mean_ranks[j])}
                for j in range(k_cond)
            ]))
            test_tbl = _fmt_np_df(pd.DataFrame([{
                "N": n_subj,
                "Chi-Square": float(fr_stat),
                "df": df_fr,
                "Asymp. Sig.": float(p_fr)
            }]))

            show_table(desc_tbl, "Descriptive Statistics (Median & IQR)")
            download_table_block(desc_tbl, "friedman_desc", "Descriptive Statistics")
            show_table(ranks_tbl, "Ranks")
            download_table_block(ranks_tbl, "friedman_ranks", "Ranks")
            show_table(test_tbl, "Test Statistics (Friedman)")
            download_table_block(test_tbl, "friedman_results", "Friedman Test")

            p_str = _fmt_p_text(float(p_fr))
            if p_fr < 0.05:
                st.success(f"**Kết luận:** Sự khác biệt giữa các thời điểm/điều kiện ({', '.join(names)}) có ý nghĩa thống kê (kiểm định Friedman, χ² = {_fmt_np_val(fr_stat)}, df = {df_fr}, {p_str}).")
            else:
                st.info(f"**Kết luận:** Sự khác biệt giữa các thời điểm/điều kiện ({', '.join(names)}) không có ý nghĩa thống kê (kiểm định Friedman, χ² = {_fmt_np_val(fr_stat)}, df = {df_fr}, {p_str}).")

        st.markdown("## Quantitative Tests — Nonparametric Tests")
        np_source = st.radio(
            "Nguồn dữ liệu:",
            ["Dán / Nhập dữ liệu trực tiếp", "Tải file dữ liệu (XLSX/CSV)"],
            horizontal=True,
            key="nonparam_input_source"
        )

        # ======================================================================
        # CHẾ ĐỘ 1: DÁN / NHẬP DỮ LIỆU TRỰC TIẾP
        # ======================================================================
        if np_source == "Dán / Nhập dữ liệu trực tiếp":
            np_test_paste = st.radio(
                "Chọn kiểm định phi tham số:",
                [
                    "Mann-Whitney U (2 mẫu độc lập)",
                    "Kruskal-Wallis (≥ 2 mẫu độc lập)",
                    "Wilcoxon Signed-Rank (Ghép cặp)",
                    "Friedman (Đo lường lặp lại)"
                ],
                horizontal=True,
                key="nonparam_paste_type"
            )

            # 1. MANN-WHITNEY U
            if np_test_paste == "Mann-Whitney U (2 mẫu độc lập)":
                mw_mode = st.radio(
                    "Cách nhập dữ liệu:",
                    [
                        "Dán 2 ô riêng biệt (Mẫu 1 & Mẫu 2)",
                        "Dán 2 cột tương ứng từ Excel vào 1 ô"
                    ],
                    horizontal=True,
                    key="mw_paste_mode"
                )
                if mw_mode == "Dán 2 ô riêng biệt (Mẫu 1 & Mẫu 2)":
                    c1, c2 = st.columns(2)
                    with c1:
                        txt_mw1 = st.text_area("Dữ liệu Mẫu 1 (cách nhau khoảng trắng, ';', ','):", height=120, key="mw_g1")
                    with c2:
                        txt_mw2 = st.text_area("Dữ liệu Mẫu 2 (cách nhau khoảng trắng, ';', ','):", height=120, key="mw_g2")
                    s1, s2 = _np_parse_1d(txt_mw1), _np_parse_1d(txt_mw2)
                    nm1, nm2 = "Sample 1", "Sample 2"
                else:
                    txt_mw_2col = st.text_area("Dán 2 cột từ Excel (Cột 1: Mẫu 1, Cột 2: Mẫu 2):", height=140, key="mw_2col")
                    hdrs, cols = _np_parse_multicol(txt_mw_2col, listwise=False)
                    s1 = cols[0] if len(cols) >= 1 else np.array([])
                    s2 = cols[1] if len(cols) >= 2 else np.array([])
                    nm1 = hdrs[0] if len(hdrs) >= 1 else "Sample 1"
                    nm2 = hdrs[1] if len(hdrs) >= 2 else "Sample 2"

                if st.button("Run Mann-Whitney U Test", type="primary", use_container_width=True, key="btn_paste_mw"):
                    if len(s1) < 2 or len(s2) < 2:
                        st.warning("Vui lòng nhập ít nhất 2 giá trị hợp lệ cho mỗi mẫu.")
                    else:
                        _run_mann_whitney(nm1, s1, nm2, s2)

            # 2. KRUSKAL-WALLIS
            elif np_test_paste == "Kruskal-Wallis (≥ 2 mẫu độc lập)":
                kw_mode = st.radio(
                    "Cách nhập dữ liệu:",
                    [
                        "Dán nhiều cột từ Excel vào 1 ô (mỗi cột là 1 nhóm)",
                        "Dán từng ô riêng biệt cho mỗi nhóm"
                    ],
                    horizontal=True,
                    key="kw_paste_mode"
                )
                groups_kw = []
                if kw_mode == "Dán nhiều cột từ Excel vào 1 ô (mỗi cột là 1 nhóm)":
                    txt_kw_multi = st.text_area(
                        "Dán dữ liệu nhiều cột từ Excel (mỗi cột là một nhóm độc lập, cho phép cỡ mẫu không bằng nhau):",
                        height=150,
                        key="kw_multicol_txt"
                    )
                    hdrs, cols = _np_parse_multicol(txt_kw_multi, listwise=False)
                    groups_kw = [(hdrs[j], cols[j]) for j in range(len(cols)) if len(cols[j]) > 0]
                else:
                    k_groups = st.number_input("Số nhóm độc lập (k):", min_value=2, max_value=10, value=3, step=1, key="kw_k_num")
                    cols_ui = st.columns(min(int(k_groups), 4))
                    for j in range(int(k_groups)):
                        with cols_ui[j % len(cols_ui)]:
                            txt_g = st.text_area(f"Nhóm {j+1}:", height=110, key=f"kw_box_{j}")
                            arr_g = _np_parse_1d(txt_g)
                            if len(arr_g) > 0:
                                groups_kw.append((f"Group {j+1}", arr_g))

                if st.button("Run Kruskal-Wallis Test", type="primary", use_container_width=True, key="btn_paste_kw"):
                    if len(groups_kw) < 2 or any(len(arr) < 1 for _, arr in groups_kw):
                        st.warning("Vui lòng nhập dữ liệu hợp lệ cho ít nhất 2 nhóm.")
                    else:
                        _run_kruskal_wallis(groups_kw)

            # 3. WILCOXON SIGNED-RANK (GHÉP CẶP)
            elif np_test_paste == "Wilcoxon Signed-Rank (Ghép cặp)":
                wx_mode = st.radio(
                    "Cách nhập dữ liệu ghép cặp:",
                    [
                        "Dán 2 cột Trước, Sau (chép từ 2 cột Excel dán vào 1 ô)",
                        "Dán 2 ô riêng biệt (Trước & Sau)",
                        "Dán dãy hiệu số chênh lệch d (d = Sau - Trước)"
                    ],
                    horizontal=True,
                    key="wx_paste_mode"
                )
                arr_pre, arr_post, arr_d = None, None, None
                lbl_pre, lbl_post = "Before", "After"

                if wx_mode == "Dán 2 cột Trước, Sau (chép từ 2 cột Excel dán vào 1 ô)":
                    txt_wx_2col = st.text_area("Dán 2 cột từ Excel (Cột 1: Trước, Cột 2: Sau):", height=140, key="wx_2col")
                    hdrs, cols = _np_parse_multicol(txt_wx_2col, listwise=True)
                    arr_pre = cols[0] if len(cols) >= 1 else np.array([])
                    arr_post = cols[1] if len(cols) >= 2 else np.array([])
                    if len(hdrs) >= 2 and hdrs[0] != "Group 1":
                        lbl_pre, lbl_post = hdrs[0], hdrs[1]
                elif wx_mode == "Dán 2 ô riêng biệt (Trước & Sau)":
                    c1, c2 = st.columns(2)
                    with c1:
                        txt_wx_pre = st.text_area("Dữ liệu Trước (Before):", height=120, key="wx_pre")
                    with c2:
                        txt_wx_post = st.text_area("Dữ liệu Sau (After):", height=120, key="wx_post")
                    arr_pre = _np_parse_1d(txt_wx_pre)
                    arr_post = _np_parse_1d(txt_wx_post)
                else:
                    txt_wx_d = st.text_area(
                        "Dán dãy hiệu số d = Sau - Trước (cách nhau khoảng trắng, dấu chấm phẩy ';' hoặc dấu phẩy ','):",
                        height=120,
                        key="wx_d"
                    )
                    arr_d = _np_parse_1d(txt_wx_d)

                if st.button("Run Wilcoxon Signed-Rank Test", type="primary", use_container_width=True, key="btn_paste_wx"):
                    if wx_mode != "Dán dãy hiệu số chênh lệch d (d = Sau - Trước)":
                        if len(arr_pre) != len(arr_post):
                            st.error(f"Số lượng quan sát không khớp: Cột Trước có {len(arr_pre)} số, Cột Sau có {len(arr_post)} số.")
                        elif len(arr_pre) < 2:
                            st.warning("Vui lòng nhập ít nhất 2 cặp dữ liệu hợp lệ.")
                        else:
                            arr_d = arr_post - arr_pre
                            _run_wilcoxon_paired(arr_d, arr_pre, arr_post, lbl_pre, lbl_post, f"d = {lbl_post} - {lbl_pre}")
                    else:
                        if arr_d is None or len(arr_d) < 2:
                            st.warning("Vui lòng nhập ít nhất 2 giá trị hiệu số d hợp lệ.")
                        else:
                            _run_wilcoxon_paired(arr_d, None, None, "Before", "After", "d = After - Before")

            # 4. FRIEDMAN (ĐO LƯỜNG LẶP LẠI)
            else:
                fr_mode = st.radio(
                    "Cách nhập dữ liệu đo lường lặp lại:",
                    [
                        "Dán nhiều cột từ Excel vào 1 ô (mỗi cột là 1 thời điểm/điều kiện)",
                        "Dán từng ô riêng biệt cho mỗi thời điểm (T1, T2, T3...)"
                    ],
                    horizontal=True,
                    key="fr_paste_mode"
                )
                conds_fr = []
                if fr_mode == "Dán nhiều cột từ Excel vào 1 ô (mỗi cột là 1 thời điểm/điều kiện)":
                    txt_fr_multi = st.text_area(
                        "Dán bảng dữ liệu từ Excel (mỗi hàng là 1 đối tượng, mỗi cột là 1 thời điểm T1, T2, T3...):",
                        height=150,
                        key="fr_multicol_txt"
                    )
                    hdrs, cols = _np_parse_multicol(txt_fr_multi, listwise=True)
                    if hdrs and hdrs[0] == "Group 1":
                        hdrs = [f"T{j+1}" for j in range(len(cols))]
                    conds_fr = [(hdrs[j], cols[j]) for j in range(len(cols)) if len(cols[j]) > 0]
                else:
                    k_times = st.number_input("Số thời điểm / điều kiện đo (k):", min_value=2, max_value=10, value=3, step=1, key="fr_k_num")
                    cols_ui = st.columns(min(int(k_times), 4))
                    raw_arrs = []
                    for j in range(int(k_times)):
                        with cols_ui[j % len(cols_ui)]:
                            txt_t = st.text_area(f"Thời điểm T{j+1}:", height=110, key=f"fr_box_{j}")
                            raw_arrs.append((f"T{j+1}", _np_parse_1d(txt_t)))
                    if all(len(a) > 0 for _, a in raw_arrs):
                        lengths = [len(a) for _, a in raw_arrs]
                        if len(set(lengths)) > 1:
                            st.error(f"Số lượng đối tượng ở các thời điểm không bằng nhau: {lengths}.")
                        else:
                            conds_fr = raw_arrs

                if st.button("Run Friedman Test", type="primary", use_container_width=True, key="btn_paste_fr"):
                    if len(conds_fr) < 2 or len(conds_fr[0][1]) < 2:
                        st.warning("Vui lòng nhập ít nhất 2 thời điểm và tối thiểu 2 đối tượng đầy đủ số liệu.")
                    else:
                        _run_friedman(conds_fr)

        # ======================================================================
        # CHẾ ĐỘ 2: TẢI FILE DỮ LIỆU (GIỮ NGUYÊN CHỨC NĂNG CŨ + ĐỊNH DẠNG CHUẨN)
        # ======================================================================
        else:
            _show_template_downloads("nonparam")
            up = st.file_uploader("Upload nonparametric data (XLSX/CSV)", type=["xlsx", "csv"], key="nonparam_upload")
            if up is not None:
                df = load_uploaded_file(up)
                st.dataframe(df.head(30), use_container_width=True)
                numeric_cols = numeric_candidate_cols(df)
                np_test_up = st.radio(
                    "Test type",
                    [
                        "Mann-Whitney U test (2 independent groups)",
                        "Kruskal-Wallis test (k independent groups)",
                        "Wilcoxon signed-rank test (Paired samples)",
                        "Friedman test (Repeated measures)"
                    ],
                    key="nonparam_up_type"
                )
                if np_test_up == "Mann-Whitney U test (2 independent groups)":
                    v_col = st.selectbox("Test variable", numeric_cols, key="np_up_mw_v")
                    g_col = st.selectbox("Grouping variable", categorical_candidate_cols(df, exclude=[v_col]), key="np_up_mw_g")
                    g1, g2 = level_selector(df, g_col, "ind_mw")
                    if st.button("Run Mann-Whitney U Test", type="primary", use_container_width=True, key="btn_up_mw"):
                        s1 = numeric_series_from_df(df[df[g_col].astype(str) == str(g1)], v_col).values
                        s2 = numeric_series_from_df(df[df[g_col].astype(str) == str(g2)], v_col).values
                        if len(s1) < 2 or len(s2) < 2:
                            st.warning("Mỗi nhóm cần ít nhất 2 quan sát hợp lệ.")
                        else:
                            _run_mann_whitney(str(g1), s1, str(g2), s2)

                elif np_test_up == "Kruskal-Wallis test (k independent groups)":
                    v_col = st.selectbox("Test variable", numeric_cols, key="np_up_kw_v")
                    g_col = st.selectbox("Grouping variable", categorical_candidate_cols(df, exclude=[v_col]), key="np_up_kw_g")
                    if st.button("Run Kruskal-Wallis Test", type="primary", use_container_width=True, key="btn_up_kw"):
                        groups_up = []
                        for lvl, grp_df in df.groupby(g_col, dropna=True):
                            arr_g = numeric_series_from_df(grp_df, v_col).values
                            if len(arr_g) > 0:
                                groups_up.append((str(lvl), arr_g))
                        if len(groups_up) < 2:
                            st.warning("Cần ít nhất 2 nhóm có dữ liệu hợp lệ.")
                        else:
                            _run_kruskal_wallis(groups_up)

                elif np_test_up == "Wilcoxon signed-rank test (Paired samples)":
                    c1, c2 = st.columns(2)
                    with c1:
                        col1 = st.selectbox("Variable 1 (Before)", numeric_cols, key="np_up_wx_c1")
                    with c2:
                        col2 = st.selectbox("Variable 2 (After)", [c for c in numeric_cols if c != col1], key="np_up_wx_c2")
                    if st.button("Run Wilcoxon Signed-Rank Test", type="primary", use_container_width=True, key="btn_up_wx"):
                        d = paired_numeric_data(df, col1, col2)
                        if len(d) < 2:
                            st.warning("Cần ít nhất 2 cặp dữ liệu hợp lệ.")
                        else:
                            arr1 = d[col1].values
                            arr2 = d[col2].values
                            _run_wilcoxon_paired(arr2 - arr1, arr1, arr2, str(col1), str(col2), f"d = {col2} - {col1}")

                else:
                    fmt_fr = st.radio("Data layout", ["Wide format (columns = time points)", "Long format (subject, time, value)"], horizontal=True, key="np_up_fr_fmt")
                    if fmt_fr == "Wide format (columns = time points)":
                        sel_cols = st.multiselect("Select time/condition columns (≥ 2)", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="np_up_fr_cols")
                        if st.button("Run Friedman Test", type="primary", use_container_width=True, key="btn_up_fr_wide"):
                            if len(sel_cols) < 2:
                                st.warning("Vui lòng chọn ít nhất 2 cột thời điểm.")
                            else:
                                sub_df = df[sel_cols].apply(pd.to_numeric, errors="coerce").dropna()
                                if len(sub_df) < 2:
                                    st.warning("Cần ít nhất 2 đối tượng có đủ số liệu ở tất cả các cột.")
                                else:
                                    conds_up = [(str(c), sub_df[c].values) for c in sel_cols]
                                    _run_friedman(conds_up)
                    else:
                        all_cols = list(df.columns)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            s_col = st.selectbox("Subject ID", all_cols, index=0, key="np_up_fr_s")
                        with c2:
                            t_col = st.selectbox("Time / Condition", [c for c in all_cols if c != s_col], index=0, key="np_up_fr_t")
                        with c3:
                            v_col = st.selectbox("Value", numeric_cols, key="np_up_fr_v")
                        if st.button("Run Friedman Test", type="primary", use_container_width=True, key="btn_up_fr_long"):
                            piv = df.pivot_table(index=s_col, columns=t_col, values=v_col, aggfunc="first").dropna()
                            if piv.shape[1] < 2 or piv.shape[0] < 2:
                                st.warning("Dữ liệu sau khi chuyển đổi cần ít nhất 2 thời điểm và 2 đối tượng hợp lệ.")
                            else:
                                conds_up = [(str(c), piv[c].values) for c in piv.columns]
                                _run_friedman(conds_up)

    elif sub == "ANOVA":
        import re
        from scipy import stats

        # ----------------------------------------------------------------------
        # 1. HÀM ĐỊNH DẠNG SỐ CHUẨN: NGUYÊN HIỆN NGUYÊN, 3 SỐ THẬP PHÂN, 0 THÌ LẤY 4-5 SỐ
        # ----------------------------------------------------------------------
        def _fmt_av_val(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            if isinstance(val, (int, np.integer)):
                return str(int(val))
            if isinstance(val, (float, np.floating)):
                v = float(val)
                if abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                if abs(round(v, 3)) > 0:
                    return f"{v:.3f}"
                if v == 0.0:
                    return "0"
                if abs(round(v, 4)) > 0:
                    return f"{v:.4f}"
                if abs(round(v, 5)) > 0:
                    return f"{v:.5f}"
                return "< 0.00001"
            return str(val)

        def _fmt_av_df(df_in: pd.DataFrame) -> pd.DataFrame:
            df_out = df_in.copy()
            for col in df_out.columns:
                df_out[col] = df_out[col].apply(_fmt_av_val)
            return df_out

        def _fmt_av_p(p_val: float) -> str:
            s = _fmt_av_val(p_val)
            return f"p {s}" if s.startswith("<") else f"p = {s}"

        # ----------------------------------------------------------------------
        # 2. CÁC HÀM PHÂN GIẢI DỮ LIỆU DÁN (1 CHIỀU, NHIỀU CỘT, 2 YẾU TỐ)
        # ----------------------------------------------------------------------
        # Gọi bộ phân giải bảng thông minh đã khai báo ở đầu Quantitative Tests
        def _av_parse_1d(text: str) -> np.ndarray:
            return _qt_smart_parse_1d(text)

        def _av_parse_multicol(text: str, listwise: bool = False):
            return _qt_smart_parse_table(text, listwise=listwise, allow_long_format=(not listwise))

        def _av_parse_twoway_3cols(text: str):
            """Tách dữ liệu 3 cột từ Excel: Factor A | Factor B | Value"""
            if not text or not str(text).strip():
                return None, "Factor A", "Factor B", "Value"
            lines = [ln for ln in str(text).splitlines() if ln.strip()]
            if not lines:
                return None, "Factor A", "Factor B", "Value"

            def split_r(ln):
                if "\t" in ln:
                    return [p.strip() for p in ln.split("\t")]
                elif ";" in ln:
                    return [p.strip() for p in ln.split(";")]
                elif "," in ln:
                    p_c = [p.strip() for p in ln.split(",")]
                    if len(p_c) >= 3:
                        return p_c
                return [p.strip() for p in re.split(r"\s+", ln.strip()) if p.strip()]

            rows = [r for r in (split_r(ln) for ln in lines) if len(r) >= 3]
            if not rows:
                return None, "Factor A", "Factor B", "Value"

            first = rows[0]
            has_header = False
            try:
                float(first[2].replace(",", "."))
            except ValueError:
                has_header = True

            if has_header:
                fa_name = first[0] if first[0] else "Factor A"
                fb_name = first[1] if first[1] else "Factor B"
                v_name = first[2] if first[2] else "Value"
                data_rows = rows[1:]
            else:
                fa_name, fb_name, v_name = "Factor A", "Factor B", "Value"
                data_rows = rows

            records = []
            for r in data_rows:
                try:
                    val = float(r[2].replace(",", "."))
                    if r[0] != "" and r[1] != "":
                        records.append({fa_name: r[0], fb_name: r[1], v_name: val})
                except ValueError:
                    continue

            if not records:
                return None, fa_name, fb_name, v_name
            return pd.DataFrame(records), fa_name, fb_name, v_name

        def _av_parse_twoway_crosstab(text: str, fb_label: str = "Factor B"):
            """Tách dữ liệu bảng chéo 2 chiều từ Excel: Cột 1 = Factor A, các cột sau = mức của Factor B"""
            if not text or not str(text).strip():
                return None, "Factor A", fb_label, "Value"
            lines = [ln for ln in str(text).splitlines() if ln.strip()]
            if len(lines) < 2:
                return None, "Factor A", fb_label, "Value"

            def split_tab(ln):
                if "\t" in ln:
                    return [p.strip() for p in ln.split("\t")]
                elif ";" in ln:
                    return [p.strip() for p in ln.split(";")]
                return [p.strip() for p in re.split(r"\s{2,}", ln.strip())]

            header_parts = split_tab(lines[0])
            if len(header_parts) < 3:
                return None, "Factor A", fb_label, "Value"

            fa_name = header_parts[0] if header_parts[0] else "Factor A"
            b_levels = [h if h else f"B{j}" for j, h in enumerate(header_parts[1:], start=1)]

            records = []
            last_a = None
            for ln in lines[1:]:
                parts = split_tab(ln)
                if len(parts) < 2:
                    continue
                a_val = parts[0] if parts[0] != "" else last_a
                if a_val is None:
                    continue
                last_a = a_val
                for j, b_lvl in enumerate(b_levels):
                    if j + 1 < len(parts) and parts[j + 1] != "":
                        cell_str = parts[j + 1]
                        nums = re.findall(r"[-+]?\d*(?:[.,]\d+)?", cell_str)
                        for num_s in nums:
                            if num_s and num_s not in ["-", "+", ".", ","]:
                                try:
                                    v = float(num_s.replace(",", "."))
                                    records.append({fa_name: a_val, fb_label: b_lvl, "Value": v})
                                except ValueError:
                                    pass

            if not records:
                return None, fa_name, fb_label, "Value"
            return pd.DataFrame(records), fa_name, fb_label, "Value"

        # ----------------------------------------------------------------------
        # 3. PIPELINE ONE-WAY ANOVA + KIỂM TRA ĐIỀU KIỆN + PHÉP KIỂM THAY THẾ
        # ----------------------------------------------------------------------
        def _run_oneway_anova_pipeline(groups_list: list):
            names = [nm for nm, _ in groups_list]
            arrays = [arr for _, arr in groups_list]
            k = len(arrays)

            # A. Bảng Thống kê mô tả
            desc_rows = []
            for nm, arr in groups_list:
                n_g = len(arr)
                m_g = float(np.mean(arr))
                sd_g = float(np.std(arr, ddof=1)) if n_g > 1 else 0.0
                se_g = sd_g / np.sqrt(n_g) if n_g > 0 else 0.0
                t_c = float(stats.t.ppf(0.975, n_g - 1)) if n_g > 1 else 0.0
                q25, med, q75 = np.percentile(arr, [25, 50, 75])
                desc_rows.append({
                    "Group": nm,
                    "N": n_g,
                    "Mean": m_g,
                    "SD": sd_g,
                    "SE": se_g,
                    "95% CI Lower": m_g - t_c * se_g,
                    "95% CI Upper": m_g + t_c * se_g,
                    "Median (IQR)": f"{_fmt_av_val(med)} ({_fmt_av_val(q25)}–{_fmt_av_val(q75)})"
                })
            desc_tbl = _fmt_av_df(pd.DataFrame(desc_rows))
            show_table(desc_tbl, "Descriptives")
            download_table_block(desc_tbl, "anova_descriptives", "Descriptives")

            # B. Điều kiện 1: Phân phối chuẩn (Shapiro-Wilk)
            norm_rows = []
            all_normal = True
            non_normal_groups = []
            too_small_groups = []
            for nm, arr in groups_list:
                n_g = len(arr)
                if n_g < 3:
                    norm_rows.append({
                        "Group": nm,
                        "Shapiro-Wilk Statistic": "N/A (N < 3)",
                        "df": n_g,
                        "Sig.": "N/A",
                        "Đánh giá": "Không đủ mẫu kiểm chuẩn"
                    })
                    all_normal = False
                    too_small_groups.append(nm)
                elif np.ptp(arr) == 0:
                    norm_rows.append({
                        "Group": nm,
                        "Shapiro-Wilk Statistic": "1",
                        "df": n_g,
                        "Sig.": "1",
                        "Đánh giá": "Dữ liệu hằng số"
                    })
                else:
                    w_stat, p_sw = stats.shapiro(arr)
                    is_norm = (p_sw > 0.05)
                    if not is_norm:
                        all_normal = False
                        non_normal_groups.append(f"{nm} ({_fmt_av_p(float(p_sw))})")
                    norm_rows.append({
                        "Group": nm,
                        "Shapiro-Wilk Statistic": float(w_stat),
                        "df": n_g,
                        "Sig.": float(p_sw),
                        "Đánh giá": "Phân phối chuẩn (p > 0.05)" if is_norm else "Không phân phối chuẩn (p ≤ 0.05)"
                    })
            norm_tbl = _fmt_av_df(pd.DataFrame(norm_rows))
            show_table(norm_tbl, "Điều kiện 1: Kiểm định phân phối chuẩn (Tests of Normality - Shapiro-Wilk)")
            download_table_block(norm_tbl, "anova_normality", "Tests of Normality")

            # C. Điều kiện 2: Phương sai đồng nhất (Levene)
            f_lev, p_lev = stats.levene(*arrays)
            df1_lev = k - 1
            df2_lev = sum(len(a) for a in arrays) - k
            equal_var = (float(p_lev) >= 0.05)

            lev_tbl = _fmt_av_df(pd.DataFrame([{
                "Levene Statistic": float(f_lev),
                "df1": df1_lev,
                "df2": df2_lev,
                "Sig.": float(p_lev),
                "Đánh giá": "Phương sai đồng nhất (p ≥ 0.05)" if equal_var else "Phương sai KHÔNG đồng nhất (p < 0.05)"
            }]))
            show_table(lev_tbl, "Điều kiện 2: Kiểm định phương sai đồng nhất (Test of Homogeneity of Variances - Levene)")
            download_table_block(lev_tbl, "anova_levene", "Test of Homogeneity of Variances")

            # D. Tính toán ANOVA chuẩn
            all_data = np.concatenate(arrays)
            N_tot = len(all_data)
            grand_mean = float(np.mean(all_data))
            ss_between = float(sum(len(a) * (float(np.mean(a)) - grand_mean) ** 2 for a in arrays))
            ss_within = float(sum(np.sum((a - float(np.mean(a))) ** 2) for a in arrays))
            ss_total = ss_between + ss_within
            df_between = k - 1
            df_within = N_tot - k
            df_total = N_tot - 1
            ms_between = ss_between / df_between if df_between > 0 else 0.0
            ms_within = ss_within / df_within if df_within > 0 else 0.0
            f_anova = ms_between / ms_within if ms_within > 0 else 0.0
            p_anova = float(1.0 - stats.f.cdf(f_anova, df_between, df_within)) if ms_within > 0 else 1.0
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

            anova_tbl = _fmt_av_df(pd.DataFrame([
                {"Source": "Between Groups", "Sum of Squares": ss_between, "df": df_between, "Mean Square": ms_between, "F": f_anova, "Sig.": p_anova, "Eta-squared (η²)": eta_sq},
                {"Source": "Within Groups", "Sum of Squares": ss_within, "df": df_within, "Mean Square": ms_within, "F": "", "Sig.": "", "Eta-squared (η²)": ""},
                {"Source": "Total", "Sum of Squares": ss_total, "df": df_total, "Mean Square": "", "F": "", "Sig.": "", "Eta-squared (η²)": ""}
            ]))

            # Nhánh 1: Thỏa 2 điều kiện -> One-way ANOVA chuẩn
            if all_normal and equal_var:
                st.info("✅ **Đánh giá điều kiện:** Dữ liệu các nhóm đều tuân theo phân phối chuẩn (Shapiro-Wilk p > 0.05) và phương sai đồng nhất (Levene p ≥ 0.05) → **Sử dụng phép kiểm One-way ANOVA chuẩn (Fisher's F)**.")
                show_table(anova_tbl, "ANOVA (One-Way)")
                download_table_block(anova_tbl, "oneway_anova_results", "One-Way ANOVA")

                if p_anova < 0.05 and k >= 3:
                    ph_rows = []
                    for i in range(k):
                        for j in range(i + 1, k):
                            diff_ij = float(np.mean(arrays[i]) - np.mean(arrays[j]))
                            se_ij = float(np.sqrt(ms_within * (1.0 / len(arrays[i]) + 1.0 / len(arrays[j]))))
                            q_ij = abs(diff_ij) / np.sqrt(ms_within * 0.5 * (1.0 / len(arrays[i]) + 1.0 / len(arrays[j]))) if ms_within > 0 else 0.0
                            try:
                                p_tukey = float(stats.studentized_range.sf(q_ij, k, df_within))
                                q_crit = float(stats.studentized_range.isf(0.05, k, df_within))
                                half_w = q_crit * np.sqrt(ms_within * 0.5 * (1.0 / len(arrays[i]) + 1.0 / len(arrays[j])))
                            except Exception:
                                t_ij = abs(diff_ij) / se_ij if se_ij > 0 else 0.0
                                n_pairs = k * (k - 1) / 2
                                p_tukey = min(1.0, float(2 * (1 - stats.t.cdf(t_ij, df_within)) * n_pairs))
                                half_w = float(stats.t.ppf(1 - 0.025 / n_pairs, df_within)) * se_ij
                            ph_rows.append({
                                "(I) Group": names[i],
                                "(J) Group": names[j],
                                "Mean Difference (I-J)": diff_ij,
                                "Std. Error": se_ij,
                                "Sig. (Tukey HSD)": p_tukey,
                                "95% CI Lower": diff_ij - half_w,
                                "95% CI Upper": diff_ij + half_w
                            })
                    ph_tbl = _fmt_av_df(pd.DataFrame(ph_rows))
                    show_table(ph_tbl, "Post-Hoc Multiple Comparisons (Tukey HSD)")
                    download_table_block(ph_tbl, "anova_posthoc_tukey", "Post-Hoc Tukey HSD")

                p_str = _fmt_av_p(p_anova)
                if p_anova < 0.05:
                    st.success(f"**Kết luận:** Sự khác biệt về giá trị trung bình giữa các nhóm ({', '.join(names)}) có ý nghĩa thống kê (kiểm định One-way ANOVA, F({df_between}, {df_within}) = {_fmt_av_val(f_anova)}, {p_str}, η² = {_fmt_av_val(eta_sq)}).")
                else:
                    st.info(f"**Kết luận:** Sự khác biệt về giá trị trung bình giữa các nhóm ({', '.join(names)}) không có ý nghĩa thống kê (kiểm định One-way ANOVA, F({df_between}, {df_within}) = {_fmt_av_val(f_anova)}, {p_str}).")

            # Nhánh 2: Chuẩn nhưng phương sai không đồng nhất -> Welch's ANOVA
            elif all_normal and not equal_var:
                st.warning(f"⚠️ **Đánh giá điều kiện:** Dữ liệu các nhóm có phân phối chuẩn nhưng **vi phạm điều kiện phương sai đồng nhất** (Levene {_fmt_av_p(float(p_lev))} < 0.05) → **Sử dụng phép kiểm thay thế Welch's ANOVA**.")
                ns = np.array([len(a) for a in arrays], dtype=float)
                means = np.array([np.mean(a) for a in arrays], dtype=float)
                vars_ = np.array([np.var(a, ddof=1) for a in arrays], dtype=float)
                w = ns / vars_
                w_sum = np.sum(w)
                x_bar_w = np.sum(w * means) / w_sum
                term = np.sum(((1.0 - w / w_sum) ** 2) / (ns - 1.0))
                df1_w = k - 1
                df2_w = float((k ** 2 - 1.0) / (3.0 * term))
                f_welch = float((np.sum(w * ((means - x_bar_w) ** 2)) / df1_w) / (1.0 + (2.0 * (k - 2.0) / (k ** 2 - 1.0)) * term))
                p_welch = float(1.0 - stats.f.cdf(f_welch, df1_w, df2_w))

                welch_tbl = _fmt_av_df(pd.DataFrame([{
                    "Test": "Welch's ANOVA",
                    "Statistic (F)": f_welch,
                    "df1": df1_w,
                    "df2": df2_w,
                    "Sig.": p_welch
                }]))
                show_table(welch_tbl, "Robust Tests of Equality of Means (Welch's ANOVA — Phép kiểm thay thế)")
                download_table_block(welch_tbl, "welch_anova_results", "Welch ANOVA")

                with st.expander("Xem bảng One-way ANOVA tham chiếu"):
                    show_table(anova_tbl, "ANOVA (Reference)")

                if p_welch < 0.05 and k >= 3:
                    gh_rows = []
                    for i in range(k):
                        for j in range(i + 1, k):
                            diff_ij = float(means[i] - means[j])
                            v_i, v_j = vars_[i] / ns[i], vars_[j] / ns[j]
                            se_ij = float(np.sqrt(v_i + v_j))
                            df_ij = float((v_i + v_j) ** 2 / ((v_i ** 2) / (ns[i] - 1.0) + (v_j ** 2) / (ns[j] - 1.0)))
                            t_ij = abs(diff_ij) / se_ij if se_ij > 0 else 0.0
                            q_ij = t_ij * np.sqrt(2.0)
                            try:
                                p_gh = float(stats.studentized_range.sf(q_ij, k, df_ij))
                                q_crit = float(stats.studentized_range.isf(0.05, k, df_ij))
                                half_w = (q_crit / np.sqrt(2.0)) * se_ij
                            except Exception:
                                n_pairs = k * (k - 1) / 2
                                p_gh = min(1.0, float(2 * (1 - stats.t.cdf(t_ij, df_ij)) * n_pairs))
                                half_w = float(stats.t.ppf(1 - 0.025 / n_pairs, df_ij)) * se_ij
                            gh_rows.append({
                                "(I) Group": names[i],
                                "(J) Group": names[j],
                                "Mean Difference (I-J)": diff_ij,
                                "Std. Error": se_ij,
                                "Sig. (Games-Howell)": p_gh,
                                "95% CI Lower": diff_ij - half_w,
                                "95% CI Upper": diff_ij + half_w
                            })
                    gh_tbl = _fmt_av_df(pd.DataFrame(gh_rows))
                    show_table(gh_tbl, "Post-Hoc Multiple Comparisons (Games-Howell)")
                    download_table_block(gh_tbl, "anova_posthoc_games_howell", "Post-Hoc Games-Howell")

                p_str = _fmt_av_p(p_welch)
                if p_welch < 0.05:
                    st.success(f"**Kết luận:** Sự khác biệt về giá trị trung bình giữa các nhóm ({', '.join(names)}) có ý nghĩa thống kê (kiểm định thay thế Welch's ANOVA do phương sai không đồng nhất, F({_fmt_av_val(df1_w)}, {_fmt_av_val(df2_w)}) = {_fmt_av_val(f_welch)}, {p_str}).")
                else:
                    st.info(f"**Kết luận:** Sự khác biệt về giá trị trung bình giữa các nhóm ({', '.join(names)}) không có ý nghĩa thống kê (kiểm định thay thế Welch's ANOVA do phương sai không đồng nhất, F({_fmt_av_val(df1_w)}, {_fmt_av_val(df2_w)}) = {_fmt_av_val(f_welch)}, {p_str}).")

            # Nhánh 3: Không phân phối chuẩn -> Kruskal-Wallis
            else:
                reason_txt = f"nhóm không đạt phân phối chuẩn: {', '.join(non_normal_groups)}" if non_normal_groups else f"cỡ mẫu nhóm nhỏ (< 3): {', '.join(too_small_groups)}"
                st.warning(f"⚠️ **Đánh giá điều kiện:** Vi phạm điều kiện phân phối chuẩn ({reason_txt}) → **Chuyển sang phép kiểm phi tham số thay thế Kruskal-Wallis H**.")

                kw_h, p_kw = stats.kruskal(*arrays)
                df_kw = k - 1
                all_ranks = stats.rankdata(all_data)
                rank_rows = []
                idx_s = 0
                for nm, arr in groups_list:
                    n_g = len(arr)
                    g_r = all_ranks[idx_s: idx_s + n_g]
                    idx_s += n_g
                    rank_rows.append({"Group": nm, "N": n_g, "Mean Rank": float(np.mean(g_r))})

                ranks_tbl = _fmt_av_df(pd.DataFrame(rank_rows))
                kw_tbl = _fmt_av_df(pd.DataFrame([{
                    "Test": "Kruskal-Wallis H",
                    "Chi-Square (H)": float(kw_h),
                    "df": df_kw,
                    "Asymp. Sig.": float(p_kw)
                }]))

                show_table(ranks_tbl, "Ranks (Kruskal-Wallis)")
                download_table_block(ranks_tbl, "anova_alt_kw_ranks", "Kruskal-Wallis Ranks")
                show_table(kw_tbl, "Test Statistics (Kruskal-Wallis H — Phép kiểm thay thế)")
                download_table_block(kw_tbl, "anova_alt_kw_results", "Kruskal-Wallis Test")

                with st.expander("Xem bảng One-way ANOVA tham chiếu"):
                    show_table(anova_tbl, "ANOVA (Reference)")

                if float(p_kw) < 0.05 and k >= 3:
                    n_pairs = k * (k - 1) / 2
                    pw_rows = []
                    for i in range(k):
                        for j in range(i + 1, k):
                            u_ij, p_raw = stats.mannwhitneyu(arrays[i], arrays[j], alternative="two-sided")
                            p_adj = min(1.0, float(p_raw) * n_pairs)
                            pw_rows.append({
                                "Sample 1": names[i],
                                "Sample 2": names[j],
                                "Mann-Whitney U": float(u_ij),
                                "Sig. (Raw)": float(p_raw),
                                "Adj. Sig. (Bonferroni)": p_adj
                            })
                    pw_tbl = _fmt_av_df(pd.DataFrame(pw_rows))
                    show_table(pw_tbl, "Pairwise Comparisons (Mann-Whitney U — Hiệu chỉnh Bonferroni)")
                    download_table_block(pw_tbl, "anova_alt_kw_posthoc", "Kruskal-Wallis Post-Hoc")

                p_str = _fmt_av_p(float(p_kw))
                if float(p_kw) < 0.05:
                    st.success(f"**Kết luận:** Sự khác biệt giữa các nhóm ({', '.join(names)}) có ý nghĩa thống kê (kiểm định phi tham số thay thế Kruskal-Wallis do dữ liệu không phân phối chuẩn, H = {_fmt_av_val(float(kw_h))}, df = {df_kw}, {p_str}).")
                else:
                    st.info(f"**Kết luận:** Sự khác biệt giữa các nhóm ({', '.join(names)}) không có ý nghĩa thống kê (kiểm định phi tham số thay thế Kruskal-Wallis do dữ liệu không phân phối chuẩn, H = {_fmt_av_val(float(kw_h))}, df = {df_kw}, {p_str}).")

        # ----------------------------------------------------------------------
        # 4. PIPELINE REPEATED-MEASURES ANOVA + KIỂM TRA ĐIỀU KIỆN + THAY THẾ
        # ----------------------------------------------------------------------
        def _run_rm_anova_pipeline(cond_list: list):
            names = [nm for nm, _ in cond_list]
            arrays = [arr for _, arr in cond_list]
            mat = np.column_stack(arrays)
            n_subj, k_cond = mat.shape

            desc_rows = []
            for nm, arr in cond_list:
                m_c = float(np.mean(arr))
                sd_c = float(np.std(arr, ddof=1))
                se_c = sd_c / np.sqrt(n_subj)
                q25, med, q75 = np.percentile(arr, [25, 50, 75])
                desc_rows.append({
                    "Time / Condition": nm,
                    "N": n_subj,
                    "Mean": m_c,
                    "SD": sd_c,
                    "SE": se_c,
                    "Median (IQR)": f"{_fmt_av_val(med)} ({_fmt_av_val(q25)}–{_fmt_av_val(q75)})"
                })
            desc_tbl = _fmt_av_df(pd.DataFrame(desc_rows))
            show_table(desc_tbl, "Descriptive Statistics")
            download_table_block(desc_tbl, "rm_anova_desc", "Descriptive Statistics")

            norm_rows = []
            all_normal = True
            non_normal_conds = []
            for nm, arr in cond_list:
                if n_subj < 3:
                    norm_rows.append({"Time / Condition": nm, "Shapiro-Wilk Statistic": "N/A (N < 3)", "df": n_subj, "Sig.": "N/A", "Đánh giá": "Không đủ mẫu"})
                    all_normal = False
                elif np.ptp(arr) == 0:
                    norm_rows.append({"Time / Condition": nm, "Shapiro-Wilk Statistic": "1", "df": n_subj, "Sig.": "1", "Đánh giá": "Dữ liệu hằng số"})
                else:
                    w_stat, p_sw = stats.shapiro(arr)
                    is_norm = (p_sw > 0.05)
                    if not is_norm:
                        all_normal = False
                        non_normal_conds.append(f"{nm} ({_fmt_av_p(float(p_sw))})")
                    norm_rows.append({
                        "Time / Condition": nm,
                        "Shapiro-Wilk Statistic": float(w_stat),
                        "df": n_subj,
                        "Sig.": float(p_sw),
                        "Đánh giá": "Phân phối chuẩn (p > 0.05)" if is_norm else "Không phân phối chuẩn (p ≤ 0.05)"
                    })
            norm_tbl = _fmt_av_df(pd.DataFrame(norm_rows))
            show_table(norm_tbl, "Điều kiện 1: Kiểm định phân phối chuẩn (Tests of Normality - Shapiro-Wilk)")
            download_table_block(norm_tbl, "rm_anova_normality", "Tests of Normality")

            grand_mean = float(np.mean(mat))
            subj_means = np.mean(mat, axis=1, keepdims=True)
            cond_means = np.mean(mat, axis=0, keepdims=True)

            ss_total = float(np.sum((mat - grand_mean) ** 2))
            ss_subj = float(k_cond * np.sum((subj_means - grand_mean) ** 2))
            ss_cond = float(n_subj * np.sum((cond_means - grand_mean) ** 2))
            ss_err = ss_total - ss_subj - ss_cond

            df_cond = k_cond - 1
            df_err = (n_subj - 1) * (k_cond - 1)
            ms_cond = ss_cond / df_cond if df_cond > 0 else 0.0
            ms_err = ss_err / df_err if df_err > 0 else 0.0
            f_rm = ms_cond / ms_err if ms_err > 0 else 0.0
            p_rm = float(1.0 - stats.f.cdf(f_rm, df_cond, df_err)) if ms_err > 0 else 1.0
            eta_p2 = ss_cond / (ss_cond + ss_err) if (ss_cond + ss_err) > 0 else 0.0

            S = np.cov(mat, rowvar=False, ddof=1)
            S_dc = S - np.mean(S, axis=1, keepdims=True) - np.mean(S, axis=0, keepdims=True) + np.mean(S)
            tr_S_dc = float(np.trace(S_dc))
            tr_S_dc2 = float(np.sum(S_dc ** 2))
            eps_gg = (tr_S_dc ** 2) / ((k_cond - 1) * tr_S_dc2) if tr_S_dc2 > 0 else 1.0
            eps_gg = float(np.clip(eps_gg, 1.0 / (k_cond - 1), 1.0))

            df_cond_gg = df_cond * eps_gg
            df_err_gg = df_err * eps_gg
            p_gg = float(1.0 - stats.f.cdf(f_rm, df_cond_gg, df_err_gg)) if ms_err > 0 else 1.0

            sphericity_violated = False
            if k_cond >= 3 and n_subj > k_cond:
                H = np.zeros((k_cond, k_cond - 1))
                for j in range(k_cond - 1):
                    H[:j + 1, j] = 1.0
                    H[j + 1, j] = -(j + 1.0)
                    H[:, j] /= np.linalg.norm(H[:, j])
                S_proj = H.T @ S @ H
                det_S = float(np.linalg.det(S_proj))
                tr_S = float(np.trace(S_proj))
                p_dim = k_cond - 1
                df_mauchly = int(p_dim * (p_dim + 1) / 2 - 1)
                if tr_S > 0 and det_S > 1e-15:
                    w_mauchly = float(det_S / ((tr_S / p_dim) ** p_dim))
                    f_m = (2.0 * (p_dim ** 2) + p_dim + 2.0) / (6.0 * p_dim * (n_subj - 1.0))
                    chi2_m = float(-(n_subj - 1.0) * (1.0 - f_m) * np.log(w_mauchly))
                    p_mauchly = float(1.0 - stats.chi2.cdf(chi2_m, df_mauchly))
                    sphericity_violated = (p_mauchly < 0.05)
                else:
                    w_mauchly, chi2_m, p_mauchly = 0.0, np.nan, 0.0
                    sphericity_violated = True
                mauchly_tbl = _fmt_av_df(pd.DataFrame([{
                    "Mauchly's W": w_mauchly,
                    "Approx. Chi-Square": chi2_m,
                    "df": df_mauchly,
                    "Sig.": p_mauchly,
                    "Greenhouse-Geisser Epsilon": eps_gg,
                    "Đánh giá": "Vi phạm tính cầu (p < 0.05) → Dùng hiệu chỉnh Greenhouse-Geisser" if sphericity_violated else "Đạt tính cầu (p ≥ 0.05)"
                }]))
                show_table(mauchly_tbl, "Điều kiện 2: Kiểm định tính cầu phương sai (Mauchly's Test of Sphericity)")
                download_table_block(mauchly_tbl, "rm_anova_mauchly", "Mauchly Test of Sphericity")

            if all_normal:
                if sphericity_violated:
                    st.warning("⚠️ **Đánh giá điều kiện:** Dữ liệu các thời điểm phân phối chuẩn nhưng **vi phạm tính cầu phương sai (Mauchly p < 0.05)** → **Đọc kết quả hiệu chỉnh Greenhouse-Geisser**.")
                else:
                    st.info("✅ **Đánh giá điều kiện:** Dữ liệu đạt phân phối chuẩn (Shapiro-Wilk p > 0.05) và tính cầu phương sai → **Sử dụng Repeated-Measures ANOVA (Sphericity Assumed)**.")

                rm_tbl = _fmt_av_df(pd.DataFrame([
                    {
                        "Correction": "Sphericity Assumed",
                        "Sum of Squares": ss_cond,
                        "df": df_cond,
                        "Mean Square": ms_cond,
                        "F": f_rm,
                        "Sig.": p_rm,
                        "Partial Eta Squared (ηp²)": eta_p2
                    },
                    {
                        "Correction": "Greenhouse-Geisser",
                        "Sum of Squares": ss_cond,
                        "df": df_cond_gg,
                        "Mean Square": ss_cond / df_cond_gg if df_cond_gg > 0 else 0.0,
                        "F": f_rm,
                        "Sig.": p_gg,
                        "Partial Eta Squared (ηp²)": eta_p2
                    },
                    {
                        "Correction": "Error (Sphericity Assumed)",
                        "Sum of Squares": ss_err,
                        "df": df_err,
                        "Mean Square": ms_err,
                        "F": "",
                        "Sig.": "",
                        "Partial Eta Squared (ηp²)": ""
                    }
                ]))
                show_table(rm_tbl, "Tests of Within-Subjects Effects (Repeated-Measures ANOVA)")
                download_table_block(rm_tbl, "rm_anova_results", "Repeated-Measures ANOVA")

                active_p = p_gg if sphericity_violated else p_rm
                active_df1 = _fmt_av_val(df_cond_gg if sphericity_violated else df_cond)
                active_df2 = _fmt_av_val(df_err_gg if sphericity_violated else df_err)
                method_lbl = "Repeated-Measures ANOVA hiệu chỉnh Greenhouse-Geisser" if sphericity_violated else "Repeated-Measures ANOVA"

                if active_p < 0.05 and k_cond >= 3:
                    n_pairs = k_cond * (k_cond - 1) / 2
                    pw_rows = []
                    for i in range(k_cond):
                        for j in range(i + 1, k_cond):
                            d_ij = arrays[i] - arrays[j]
                            m_d = float(np.mean(d_ij))
                            se_d = float(np.std(d_ij, ddof=1) / np.sqrt(n_subj))
                            t_d = abs(m_d) / se_d if se_d > 0 else 0.0
                            p_raw = float(2 * (1 - stats.t.cdf(t_d, n_subj - 1)))
                            p_bonf = min(1.0, p_raw * n_pairs)
                            tc_b = float(stats.t.ppf(1 - 0.025 / n_pairs, n_subj - 1))
                            pw_rows.append({
                                "(I) Time": names[i],
                                "(J) Time": names[j],
                                "Mean Difference (I-J)": m_d,
                                "Std. Error": se_d,
                                "Sig. (Bonferroni)": p_bonf,
                                "95% CI Lower": m_d - tc_b * se_d,
                                "95% CI Upper": m_d + tc_b * se_d
                            })
                    pw_tbl = _fmt_av_df(pd.DataFrame(pw_rows))
                    show_table(pw_tbl, "Pairwise Comparisons (Bonferroni)")
                    download_table_block(pw_tbl, "rm_anova_posthoc", "Pairwise Comparisons")

                p_str = _fmt_av_p(active_p)
                if active_p < 0.05:
                    st.success(f"**Kết luận:** Sự khác biệt giữa các thời điểm/điều kiện ({', '.join(names)}) có ý nghĩa thống kê (kiểm định {method_lbl}, F({active_df1}, {active_df2}) = {_fmt_av_val(f_rm)}, {p_str}, ηp² = {_fmt_av_val(eta_p2)}).")
                else:
                    st.info(f"**Kết luận:** Sự khác biệt giữa các thời điểm/điều kiện ({', '.join(names)}) không có ý nghĩa thống kê (kiểm định {method_lbl}, F({active_df1}, {active_df2}) = {_fmt_av_val(f_rm)}, {p_str}).")

            else:
                reason_rm = f"thời điểm không đạt phân phối chuẩn: {', '.join(non_normal_conds)}" if non_normal_conds else "cỡ mẫu N < 3"
                st.warning(f"⚠️ **Đánh giá điều kiện:** Vi phạm điều kiện phân phối chuẩn ({reason_rm}) → **Chuyển sang phép kiểm phi tham số thay thế Friedman Test**.")

                fr_stat, p_fr = stats.friedmanchisquare(*arrays)
                df_fr = k_cond - 1
                row_ranks = np.apply_along_axis(stats.rankdata, 1, mat)
                mean_ranks = np.mean(row_ranks, axis=0)

                ranks_fr_tbl = _fmt_av_df(pd.DataFrame([
                    {"Time / Condition": names[j], "Mean Rank": float(mean_ranks[j])}
                    for j in range(k_cond)
                ]))
                fr_tbl = _fmt_av_df(pd.DataFrame([{
                    "Test": "Friedman Test",
                    "N": n_subj,
                    "Chi-Square (χ²)": float(fr_stat),
                    "df": df_fr,
                    "Asymp. Sig.": float(p_fr)
                }]))

                show_table(ranks_fr_tbl, "Ranks (Friedman Test)")
                download_table_block(ranks_fr_tbl, "rm_alt_friedman_ranks", "Friedman Ranks")
                show_table(fr_tbl, "Test Statistics (Friedman Test — Phép kiểm thay thế)")
                download_table_block(fr_tbl, "rm_alt_friedman_results", "Friedman Test")

                if float(p_fr) < 0.05 and k_cond >= 3:
                    n_pairs = k_cond * (k_cond - 1) / 2
                    pw_wx_rows = []
                    for i in range(k_cond):
                        for j in range(i + 1, k_cond):
                            d_ij = arrays[j] - arrays[i]
                            nz_d = d_ij[d_ij != 0]
                            if len(nz_d) > 0:
                                w_res = stats.wilcoxon(nz_d, alternative="two-sided")
                                p_raw = float(w_res.pvalue)
                            else:
                                p_raw = 1.0
                            p_adj = min(1.0, p_raw * n_pairs)
                            pw_wx_rows.append({
                                "Time 1": names[i],
                                "Time 2": names[j],
                                "Sig. (Raw Wilcoxon)": p_raw,
                                "Adj. Sig. (Bonferroni)": p_adj
                            })
                    pw_wx_tbl = _fmt_av_df(pd.DataFrame(pw_wx_rows))
                    show_table(pw_wx_tbl, "Pairwise Comparisons (Wilcoxon — Hiệu chỉnh Bonferroni)")
                    download_table_block(pw_wx_tbl, "rm_alt_friedman_posthoc", "Friedman Post-Hoc")

                p_str = _fmt_av_p(float(p_fr))
                if float(p_fr) < 0.05:
                    st.success(f"**Kết luận:** Sự khác biệt giữa các thời điểm/điều kiện ({', '.join(names)}) có ý nghĩa thống kê (kiểm định phi tham số thay thế Friedman do dữ liệu không phân phối chuẩn, χ² = {_fmt_av_val(float(fr_stat))}, df = {df_fr}, {p_str}).")
                else:
                    st.info(f"**Kết luận:** Sự khác biệt giữa các thời điểm/điều kiện ({', '.join(names)}) không có ý nghĩa thống kê (kiểm định phi tham số thay thế Friedman do dữ liệu không phân phối chuẩn, χ² = {_fmt_av_val(float(fr_stat))}, df = {df_fr}, {p_str}).")

        # ----------------------------------------------------------------------
        # 5. PIPELINE TWO-WAY ANOVA + KIỂM TRA 2 ĐIỀU KIỆN + THAY THẾ SCHEIRER-RAY-HARE
        # ----------------------------------------------------------------------
        def _calc_twoway_type3(df_in: pd.DataFrame, fa_col: str, fb_col: str, v_col: str):
            a_levels = sorted(df_in[fa_col].unique())
            b_levels = sorted(df_in[fb_col].unique())
            a_cnt = len(a_levels)
            b_cnt = len(b_levels)
            n_obs = len(df_in)
            y = df_in[v_col].values
            grand_mean = float(np.mean(y))
            ss_total = float(np.sum((y - grand_mean) ** 2))

            def effect_matrix(series, levels):
                k = len(levels)
                M = np.zeros((len(series), k - 1))
                for idx, val in enumerate(series):
                    pos = levels.index(val)
                    if pos < k - 1:
                        M[idx, pos] = 1.0
                    else:
                        M[idx, :] = -1.0
                return M

            Xa = effect_matrix(df_in[fa_col].tolist(), a_levels)
            Xb = effect_matrix(df_in[fb_col].tolist(), b_levels)
            Xab_cols = []
            for i in range(Xa.shape[1]):
                for j in range(Xb.shape[1]):
                    Xab_cols.append((Xa[:, i] * Xb[:, j]).reshape(-1, 1))
            Xab = np.hstack(Xab_cols) if Xab_cols else np.empty((n_obs, 0))
            ones = np.ones((n_obs, 1))

            def rss(X):
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                res = y - X @ beta
                return float(np.sum(res ** 2)), res

            X_full = np.hstack([ones, Xa, Xb, Xab])
            rss_full, residuals = rss(X_full)

            rss_no_a, _ = rss(np.hstack([ones, Xb, Xab]))
            rss_no_b, _ = rss(np.hstack([ones, Xa, Xab]))
            rss_no_ab, _ = rss(np.hstack([ones, Xa, Xb]))

            ss_a = max(0.0, rss_no_a - rss_full)
            ss_b = max(0.0, rss_no_b - rss_full)
            ss_ab = max(0.0, rss_no_ab - rss_full)
            ss_err = max(0.0, rss_full)

            df_a = a_cnt - 1
            df_b = b_cnt - 1
            df_ab = df_a * df_b
            df_err = n_obs - (a_cnt * b_cnt)

            ms_a = ss_a / df_a if df_a > 0 else 0.0
            ms_b = ss_b / df_b if df_b > 0 else 0.0
            ms_ab = ss_ab / df_ab if df_ab > 0 else 0.0
            ms_err = ss_err / df_err if df_err > 0 else 0.0

            f_a = ms_a / ms_err if ms_err > 0 else 0.0
            f_b = ms_b / ms_err if ms_err > 0 else 0.0
            f_ab = ms_ab / ms_err if ms_err > 0 else 0.0

            p_a = float(1.0 - stats.f.cdf(f_a, df_a, df_err)) if (df_a > 0 and df_err > 0) else 1.0
            p_b = float(1.0 - stats.f.cdf(f_b, df_b, df_err)) if (df_b > 0 and df_err > 0) else 1.0
            p_ab = float(1.0 - stats.f.cdf(f_ab, df_ab, df_err)) if (df_ab > 0 and df_err > 0) else 1.0

            eta_a = ss_a / (ss_a + ss_err) if (ss_a + ss_err) > 0 else 0.0
            eta_b = ss_b / (ss_b + ss_err) if (ss_b + ss_err) > 0 else 0.0
            eta_ab = ss_ab / (ss_ab + ss_err) if (ss_ab + ss_err) > 0 else 0.0

            return {
                "ss_a": ss_a, "df_a": df_a, "ms_a": ms_a, "f_a": f_a, "p_a": p_a, "eta_a": eta_a,
                "ss_b": ss_b, "df_b": df_b, "ms_b": ms_b, "f_b": f_b, "p_b": p_b, "eta_b": eta_b,
                "ss_ab": ss_ab, "df_ab": df_ab, "ms_ab": ms_ab, "f_ab": f_ab, "p_ab": p_ab, "eta_ab": eta_ab,
                "ss_err": ss_err, "df_err": df_err, "ms_err": ms_err,
                "ss_tot": ss_total, "df_tot": n_obs - 1,
                "residuals": residuals
            }

        def _run_twoway_anova_pipeline(df_raw: pd.DataFrame, fa_col: str, fb_col: str, v_col: str):
            sub2 = df_raw[[fa_col, fb_col, v_col]].dropna().copy()
            sub2[fa_col] = sub2[fa_col].astype(str).str.strip()
            sub2[fb_col] = sub2[fb_col].astype(str).str.strip()
            sub2[v_col] = pd.to_numeric(sub2[v_col], errors="coerce")
            sub2 = sub2.dropna()

            a_levels = sorted(sub2[fa_col].unique())
            b_levels = sorted(sub2[fb_col].unique())
            if len(a_levels) < 2 or len(b_levels) < 2:
                st.warning("Mỗi yếu tố (Factor A và Factor B) cần có ít nhất 2 mức (nhóm) khác nhau.")
                return
            if len(sub2) <= len(a_levels) * len(b_levels):
                st.warning("Số quan sát cần lớn hơn số ô tổ hợp (Factor A × Factor B) để tính sai số dư (Error df > 0).")
                return

            # A. Thống kê mô tả theo từng tổ hợp ô (Factor A × Factor B)
            desc_rows = []
            cell_arrays = []
            for la in a_levels:
                for lb in b_levels:
                    vals = sub2[(sub2[fa_col] == la) & (sub2[fb_col] == lb)][v_col].values
                    if len(vals) > 0:
                        cell_arrays.append((f"{la} × {lb}", vals))
                        q25, med, q75 = np.percentile(vals, [25, 50, 75])
                        desc_rows.append({
                            str(fa_col): la,
                            str(fb_col): lb,
                            "N": len(vals),
                            "Mean": float(np.mean(vals)),
                            "SD": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "Median (IQR)": f"{_fmt_av_val(med)} ({_fmt_av_val(q25)}–{_fmt_av_val(q75)})"
                        })
            desc_2w = _fmt_av_df(pd.DataFrame(desc_rows))
            show_table(desc_2w, "Descriptive Statistics (Theo từng tổ hợp Factor A × Factor B)")
            download_table_block(desc_2w, "twoway_descriptives", "Descriptive Statistics")

            # Tính Type III Two-Way ANOVA
            tw_res = _calc_twoway_type3(sub2, fa_col, fb_col, v_col)
            res_arr = tw_res["residuals"]

            # B. KIỂM TRA ĐIỀU KIỆN 1: PHÂN PHỐI CHUẨN (SHAPIRO-WILK)
            norm_rows = []
            is_normal_2w = True
            if len(res_arr) >= 3 and np.ptp(res_arr) > 0:
                w_res, p_res = stats.shapiro(res_arr)
                is_normal_2w = (float(p_res) > 0.05)
                norm_rows.append({
                    "Đối tượng kiểm định": "Phần dư mô hình (Model Residuals)",
                    "Shapiro-Wilk Statistic": float(w_res),
                    "df": len(res_arr),
                    "Sig.": float(p_res),
                    "Đánh giá": "Phân phối chuẩn (p > 0.05)" if is_normal_2w else "Không phân phối chuẩn (p ≤ 0.05)"
                })
            # Kiểm tra thêm từng ô tổ hợp nếu ô có N >= 3
            for cell_lbl, vals in cell_arrays:
                if len(vals) >= 3 and np.ptp(vals) > 0:
                    w_c, p_c = stats.shapiro(vals)
                    norm_rows.append({
                        "Đối tượng kiểm định": f"Ô tổ hợp: {cell_lbl}",
                        "Shapiro-Wilk Statistic": float(w_c),
                        "df": len(vals),
                        "Sig.": float(p_c),
                        "Đánh giá": "Phân phối chuẩn (p > 0.05)" if p_c > 0.05 else "Không phân phối chuẩn (p ≤ 0.05)"
                    })
            if norm_rows:
                norm_2w_tbl = _fmt_av_df(pd.DataFrame(norm_rows))
                show_table(norm_2w_tbl, "Điều kiện 1: Kiểm định phân phối chuẩn (Tests of Normality - Shapiro-Wilk)")
                download_table_block(norm_2w_tbl, "twoway_normality", "Tests of Normality")

            # C. KIỂM TRA ĐIỀU KIỆN 2: PHƯƠNG SAI ĐỒNG NHẤT GIỮA CÁC Ô (LEVENE'S TEST)
            valid_cells = [vals for _, vals in cell_arrays if len(vals) >= 2]
            equal_var_2w = True
            p_lev2 = 1.0
            if len(valid_cells) >= 2:
                f_lev2, p_lev2 = stats.levene(*valid_cells)
                df1_l2 = len(valid_cells) - 1
                df2_l2 = sum(len(v) for v in valid_cells) - len(valid_cells)
                equal_var_2w = (float(p_lev2) >= 0.05)
                lev_2w_tbl = _fmt_av_df(pd.DataFrame([{
                    "Levene Statistic": float(f_lev2),
                    "df1": df1_l2,
                    "df2": df2_l2,
                    "Sig.": float(p_lev2),
                    "Đánh giá": "Phương sai đồng nhất (p ≥ 0.05)" if equal_var_2w else "Phương sai KHÔNG đồng nhất (p < 0.05)"
                }]))
                show_table(lev_2w_tbl, "Điều kiện 2: Kiểm định phương sai đồng nhất các ô tổ hợp (Levene's Test)")
                download_table_block(lev_2w_tbl, "twoway_levene", "Test of Homogeneity of Variances")

            # Bảng Two-Way ANOVA chuẩn (Type III)
            anova_2w_tbl = _fmt_av_df(pd.DataFrame([
                {"Source": str(fa_col), "Sum of Squares": tw_res["ss_a"], "df": tw_res["df_a"], "Mean Square": tw_res["ms_a"], "F": tw_res["f_a"], "Sig.": tw_res["p_a"], "Partial Eta Squared (ηp²)": tw_res["eta_a"]},
                {"Source": str(fb_col), "Sum of Squares": tw_res["ss_b"], "df": tw_res["df_b"], "Mean Square": tw_res["ms_b"], "F": tw_res["f_b"], "Sig.": tw_res["p_b"], "Partial Eta Squared (ηp²)": tw_res["eta_b"]},
                {"Source": f"{fa_col} * {fb_col}", "Sum of Squares": tw_res["ss_ab"], "df": tw_res["df_ab"], "Mean Square": tw_res["ms_ab"], "F": tw_res["f_ab"], "Sig.": tw_res["p_ab"], "Partial Eta Squared (ηp²)": tw_res["eta_ab"]},
                {"Source": "Error", "Sum of Squares": tw_res["ss_err"], "df": tw_res["df_err"], "Mean Square": tw_res["ms_err"], "F": "", "Sig.": "", "Partial Eta Squared (ηp²)": ""},
                {"Source": "Total", "Sum of Squares": tw_res["ss_tot"], "df": tw_res["df_tot"], "Mean Square": "", "F": "", "Sig.": "", "Partial Eta Squared (ηp²)": ""}
            ]))

            # D. CHỌN PHÉP KIỂM PHÙ HỢP DỰA TRÊN 2 ĐIỀU KIỆN
            if is_normal_2w and equal_var_2w:
                st.info("✅ **Đánh giá điều kiện:** Phần dư mô hình đạt phân phối chuẩn (Shapiro-Wilk p > 0.05) và phương sai các ô tổ hợp đồng nhất (Levene p ≥ 0.05) → **Sử dụng phép kiểm Two-way ANOVA chuẩn (Type III)**.")
                show_table(anova_2w_tbl, "Tests of Between-Subjects Effects (Two-Way ANOVA)")
                download_table_block(anova_2w_tbl, "twoway_anova_results", "Two-Way ANOVA")

                # Kết luận chi tiết 3 tác động
                c_a = f"- **Tác động chính của {fa_col}:** " + (
                    f"Có ý nghĩa thống kê (F({tw_res['df_a']}, {tw_res['df_err']}) = {_fmt_av_val(tw_res['f_a'])}, {_fmt_av_p(tw_res['p_a'])}, ηp² = {_fmt_av_val(tw_res['eta_a'])})."
                    if tw_res["p_a"] < 0.05 else
                    f"Không có ý nghĩa thống kê (F({tw_res['df_a']}, {tw_res['df_err']}) = {_fmt_av_val(tw_res['f_a'])}, {_fmt_av_p(tw_res['p_a'])})."
                )
                c_b = f"- **Tác động chính của {fb_col}:** " + (
                    f"Có ý nghĩa thống kê (F({tw_res['df_b']}, {tw_res['df_err']}) = {_fmt_av_val(tw_res['f_b'])}, {_fmt_av_p(tw_res['p_b'])}, ηp² = {_fmt_av_val(tw_res['eta_b'])})."
                    if tw_res["p_b"] < 0.05 else
                    f"Không có ý nghĩa thống kê (F({tw_res['df_b']}, {tw_res['df_err']}) = {_fmt_av_val(tw_res['f_b'])}, {_fmt_av_p(tw_res['p_b'])})."
                )
                c_ab = f"- **Tác động tương tác ({fa_col} × {fb_col}):** " + (
                    f"Có ý nghĩa thống kê (F({tw_res['df_ab']}, {tw_res['df_err']}) = {_fmt_av_val(tw_res['f_ab'])}, {_fmt_av_p(tw_res['p_ab'])}, ηp² = {_fmt_av_val(tw_res['eta_ab'])})."
                    if tw_res["p_ab"] < 0.05 else
                    f"Không có ý nghĩa thống kê (F({tw_res['df_ab']}, {tw_res['df_err']}) = {_fmt_av_val(tw_res['f_ab'])}, {_fmt_av_p(tw_res['p_ab'])})."
                )
                st.success(f"**Kết luận phân tích phương sai 2 yếu tố (Two-way ANOVA):**\n\n{c_a}\n\n{c_b}\n\n{c_ab}")

            else:
                # Vi phạm điều kiện -> Phép kiểm phi tham số thay thế Scheirer-Ray-Hare Test
                viol_reasons = []
                if not is_normal_2w:
                    viol_reasons.append("vi phạm phân phối chuẩn (Shapiro-Wilk p ≤ 0.05)")
                if not equal_var_2w:
                    viol_reasons.append(f"vi phạm phương sai đồng nhất (Levene {_fmt_av_p(float(p_lev2))} < 0.05)")
                st.warning(f"⚠️ **Đánh giá điều kiện:** Dữ liệu {' và '.join(viol_reasons)} → **Chuyển sang phép kiểm phi tham số thay thế Scheirer-Ray-Hare Test (Two-way ANOVA trên hạng)**.")

                sub_rank = sub2.copy()
                sub_rank["__rank__"] = stats.rankdata(sub_rank[v_col].values)
                rk_res = _calc_twoway_type3(sub_rank, fa_col, fb_col, "__rank__")
                n_obs_r = len(sub_rank)
                ms_tot_r = rk_res["ss_tot"] / (n_obs_r - 1) if n_obs_r > 1 else 1.0

                h_a = rk_res["ss_a"] / ms_tot_r if ms_tot_r > 0 else 0.0
                h_b = rk_res["ss_b"] / ms_tot_r if ms_tot_r > 0 else 0.0
                h_ab = rk_res["ss_ab"] / ms_tot_r if ms_tot_r > 0 else 0.0

                p_ha = float(1.0 - stats.chi2.cdf(h_a, rk_res["df_a"])) if rk_res["df_a"] > 0 else 1.0
                p_hb = float(1.0 - stats.chi2.cdf(h_b, rk_res["df_b"])) if rk_res["df_b"] > 0 else 1.0
                p_hab = float(1.0 - stats.chi2.cdf(h_ab, rk_res["df_ab"])) if rk_res["df_ab"] > 0 else 1.0

                srh_tbl = _fmt_av_df(pd.DataFrame([
                    {"Source": str(fa_col), "Sum of Squares (Ranks)": rk_res["ss_a"], "df": rk_res["df_a"], "H Statistic": h_a, "Sig. (Chi-Square)": p_ha},
                    {"Source": str(fb_col), "Sum of Squares (Ranks)": rk_res["ss_b"], "df": rk_res["df_b"], "H Statistic": h_b, "Sig. (Chi-Square)": p_hb},
                    {"Source": f"{fa_col} * {fb_col}", "Sum of Squares (Ranks)": rk_res["ss_ab"], "df": rk_res["df_ab"], "H Statistic": h_ab, "Sig. (Chi-Square)": p_hab}
                ]))
                show_table(srh_tbl, "Scheirer-Ray-Hare Test (Two-Way Nonparametric ANOVA — Phép kiểm thay thế)")
                download_table_block(srh_tbl, "scheirer_ray_hare_results", "Scheirer-Ray-Hare Test")

                with st.expander("Xem bảng Two-way ANOVA tham chiếu"):
                    show_table(anova_2w_tbl, "Tests of Between-Subjects Effects (Reference)")

                c_a = f"- **Tác động chính của {fa_col}:** " + (
                    f"Có ý nghĩa thống kê (H = {_fmt_av_val(h_a)}, df = {rk_res['df_a']}, {_fmt_av_p(p_ha)})."
                    if p_ha < 0.05 else
                    f"Không có ý nghĩa thống kê (H = {_fmt_av_val(h_a)}, df = {rk_res['df_a']}, {_fmt_av_p(p_ha)})."
                )
                c_b = f"- **Tác động chính của {fb_col}:** " + (
                    f"Có ý nghĩa thống kê (H = {_fmt_av_val(h_b)}, df = {rk_res['df_b']}, {_fmt_av_p(p_hb)})."
                    if p_hb < 0.05 else
                    f"Không có ý nghĩa thống kê (H = {_fmt_av_val(h_b)}, df = {rk_res['df_b']}, {_fmt_av_p(p_hb)})."
                )
                c_ab = f"- **Tác động tương tác ({fa_col} × {fb_col}):** " + (
                    f"Có ý nghĩa thống kê (H = {_fmt_av_val(h_ab)}, df = {rk_res['df_ab']}, {_fmt_av_p(p_hab)})."
                    if p_hab < 0.05 else
                    f"Không có ý nghĩa thống kê (H = {_fmt_av_val(h_ab)}, df = {rk_res['df_ab']}, {_fmt_av_p(p_hab)})."
                )
                st.success(f"**Kết luận phân tích phi tham số 2 yếu tố (Scheirer-Ray-Hare Test):**\n\n{c_a}\n\n{c_b}\n\n{c_ab}")

        # ----------------------------------------------------------------------
        # 6. GIAO DIỆN CHÍNH CỦA PHÂN HỆ ANOVA
        # ----------------------------------------------------------------------
        st.markdown("## Quantitative Tests — ANOVA")
        av_source = st.radio(
            "Nguồn dữ liệu:",
            ["Dán / Nhập dữ liệu trực tiếp", "Tải file dữ liệu (XLSX/CSV)"],
            horizontal=True,
            key="anova_input_source"
        )

        # ======================================================================
        # CHẾ ĐỘ 1: DÁN / NHẬP DỮ LIỆU TRỰC TIẾP (ĐẦY ĐỦ 1 YẾU TỐ, LẶP LẠI, 2 YẾU TỐ)
        # ======================================================================
        if av_source == "Dán / Nhập dữ liệu trực tiếp":
            av_type_paste = st.radio(
                "Chọn dạng phân tích phương sai:",
                [
                    "One-way ANOVA (Phân tích phương sai 1 yếu tố)",
                    "Repeated-Measures ANOVA (Phân tích phương sai đo lường lặp lại)",
                    "Two-way ANOVA (Phân tích phương sai 2 yếu tố)"
                ],
                horizontal=True,
                key="anova_paste_type"
            )

            # 1. ONE-WAY ANOVA
            if av_type_paste == "One-way ANOVA (Phân tích phương sai 1 yếu tố)":
                ow_mode = st.radio(
                    "Cách nhập dữ liệu các nhóm:",
                    [
                        "Dán nhiều cột từ Excel vào 1 ô (mỗi cột là 1 nhóm)",
                        "Dán từng ô riêng biệt cho mỗi nhóm"
                    ],
                    horizontal=True,
                    key="anova_ow_mode"
                )
                groups_ow = []
                if ow_mode == "Dán nhiều cột từ Excel vào 1 ô (mỗi cột là 1 nhóm)":
                    txt_ow_multi = st.text_area(
                        "Dán dữ liệu nhiều cột từ Excel (mỗi cột là một nhóm độc lập, cho phép cỡ mẫu không bằng nhau):",
                        height=150,
                        key="anova_ow_multicol"
                    )
                    hdrs, cols = _av_parse_multicol(txt_ow_multi, listwise=False)
                    groups_ow = [(hdrs[j], cols[j]) for j in range(len(cols)) if len(cols[j]) > 0]
                else:
                    k_ow = st.number_input("Số nhóm độc lập (k):", min_value=2, max_value=10, value=3, step=1, key="anova_ow_k")
                    cols_ui = st.columns(min(int(k_ow), 4))
                    for j in range(int(k_ow)):
                        with cols_ui[j % len(cols_ui)]:
                            txt_g = st.text_area(f"Nhóm {j+1}:", height=110, key=f"anova_ow_box_{j}")
                            arr_g = _av_parse_1d(txt_g)
                            if len(arr_g) > 0:
                                groups_ow.append((f"Group {j+1}", arr_g))

                if st.button("Run One-Way ANOVA", type="primary", use_container_width=True, key="btn_paste_ow_anova"):
                    if len(groups_ow) < 2 or any(len(arr) < 2 for _, arr in groups_ow):
                        st.warning("Vui lòng nhập ít nhất 2 nhóm, mỗi nhóm tối thiểu 2 giá trị số hợp lệ.")
                    else:
                        _run_oneway_anova_pipeline(groups_ow)

            # 2. REPEATED-MEASURES ANOVA
            elif av_type_paste == "Repeated-Measures ANOVA (Phân tích phương sai đo lường lặp lại)":
                rm_mode = st.radio(
                    "Cách nhập dữ liệu đo lường lặp lại:",
                    [
                        "Dán nhiều cột từ Excel vào 1 ô (mỗi cột là 1 thời điểm/điều kiện)",
                        "Dán từng ô riêng biệt cho mỗi thời điểm (T1, T2, T3...)"
                    ],
                    horizontal=True,
                    key="anova_rm_mode"
                )
                conds_rm = []
                if rm_mode == "Dán nhiều cột từ Excel vào 1 ô (mỗi cột là 1 thời điểm/điều kiện)":
                    txt_rm_multi = st.text_area(
                        "Dán bảng dữ liệu từ Excel (mỗi hàng là 1 đối tượng, mỗi cột là 1 thời điểm T1, T2, T3...):",
                        height=150,
                        key="anova_rm_multicol"
                    )
                    hdrs, cols = _av_parse_multicol(txt_rm_multi, listwise=True)
                    if hdrs and hdrs[0] == "Group 1":
                        hdrs = [f"T{j+1}" for j in range(len(cols))]
                    conds_rm = [(hdrs[j], cols[j]) for j in range(len(cols)) if len(cols[j]) > 0]
                else:
                    k_rm = st.number_input("Số thời điểm / điều kiện đo (k):", min_value=2, max_value=10, value=3, step=1, key="anova_rm_k")
                    cols_ui = st.columns(min(int(k_rm), 4))
                    raw_arrs = []
                    for j in range(int(k_rm)):
                        with cols_ui[j % len(cols_ui)]:
                            txt_t = st.text_area(f"Thời điểm T{j+1}:", height=110, key=f"anova_rm_box_{j}")
                            raw_arrs.append((f"T{j+1}", _av_parse_1d(txt_t)))
                    if all(len(a) > 0 for _, a in raw_arrs):
                        lengths = [len(a) for _, a in raw_arrs]
                        if len(set(lengths)) > 1:
                            st.error(f"Số lượng đối tượng ở các thời điểm không bằng nhau: {lengths}.")
                        else:
                            conds_rm = raw_arrs

                if st.button("Run Repeated-Measures ANOVA", type="primary", use_container_width=True, key="btn_paste_rm_anova"):
                    if len(conds_rm) < 2 or len(conds_rm[0][1]) < 2:
                        st.warning("Vui lòng nhập ít nhất 2 thời điểm và tối thiểu 2 đối tượng đầy đủ số liệu.")
                    else:
                        _run_rm_anova_pipeline(conds_rm)

            # 3. TWO-WAY ANOVA
            else:
                tw_mode = st.radio(
                    "Cách nhập dữ liệu 2 yếu tố:",
                    [
                        "Dán 3 cột từ Excel (Factor A | Factor B | Value)",
                        "Dán bảng chéo 2 chiều từ Excel (Cột 1 = Factor A, các cột sau = mức Factor B)"
                    ],
                    horizontal=True,
                    key="anova_tw_mode"
                )
                df_tw_parsed, fa_nm, fb_nm, v_nm = None, "Factor A", "Factor B", "Value"

                if tw_mode == "Dán 3 cột từ Excel (Factor A | Factor B | Value)":
                    txt_tw_3c = st.text_area(
                        "Dán 3 cột từ Excel (Cột 1: Yếu tố A, Cột 2: Yếu tố B, Cột 3: Giá trị định lượng):",
                        height=160,
                        key="anova_tw_3c_txt"
                    )
                    df_tw_parsed, fa_nm, fb_nm, v_nm = _av_parse_twoway_3cols(txt_tw_3c)
                else:
                    fb_custom_name = st.text_input("Tên Yếu tố B (đại diện cho các cột):", value="Factor B", key="anova_tw_fb_name")
                    txt_tw_ct = st.text_area(
                        "Dán bảng chéo từ Excel (Dòng 1: Tên yếu tố A và tên các mức yếu tố B; các dòng sau là mức A và các giá trị):",
                        height=160,
                        key="anova_tw_ct_txt"
                    )
                    df_tw_parsed, fa_nm, fb_nm, v_nm = _av_parse_twoway_crosstab(txt_tw_ct, fb_label=fb_custom_name.strip() or "Factor B")

                if st.button("Run Two-Way ANOVA", type="primary", use_container_width=True, key="btn_paste_tw_anova"):
                    if df_tw_parsed is None or len(df_tw_parsed) < 4:
                        st.warning("Vui lòng dán đầy đủ dữ liệu hợp lệ cho Two-way ANOVA (ít nhất 2 mức cho mỗi yếu tố).")
                    else:
                        _run_twoway_anova_pipeline(df_tw_parsed, fa_nm, fb_nm, v_nm)

        # ======================================================================
        # CHẾ ĐỘ 2: TẢI FILE DỮ LIỆU (GIỮ NGUYÊN TEMPLATE & ĐẦY ĐỦ KIỂM TRA ĐIỀU KIỆN)
        # ======================================================================
        else:
            _show_template_downloads("anova")
            up = st.file_uploader("Upload ANOVA data (XLSX/CSV)", type=["xlsx", "csv"], key="anova_upload")
            if up is not None:
                df = load_uploaded_file(up)
                st.dataframe(df.head(30), use_container_width=True)
                numeric_cols = numeric_candidate_cols(df)
                av_up_type = st.radio(
                    "ANOVA type",
                    ["One-way ANOVA", "Repeated-measures ANOVA", "Two-way ANOVA"],
                    horizontal=True,
                    key="anova_up_type"
                )

                if av_up_type == "One-way ANOVA":
                    v_col = st.selectbox("Dependent variable", numeric_cols, key="av_up_ow_v")
                    g_col = st.selectbox("Factor (Grouping variable)", categorical_candidate_cols(df, exclude=[v_col]), key="av_up_ow_g")
                    if st.button("Run One-Way ANOVA", type="primary", use_container_width=True, key="btn_up_ow_anova"):
                        groups_up = []
                        for lvl, grp_df in df.groupby(g_col, dropna=True):
                            arr_g = numeric_series_from_df(grp_df, v_col).values
                            if len(arr_g) > 0:
                                groups_up.append((str(lvl), arr_g))
                        if len(groups_up) < 2 or any(len(a) < 2 for _, a in groups_up):
                            st.warning("Cần ít nhất 2 nhóm, mỗi nhóm có tối thiểu 2 quan sát hợp lệ.")
                        else:
                            _run_oneway_anova_pipeline(groups_up)

                elif av_up_type == "Repeated-measures ANOVA":
                    fmt_rm = st.radio("Data layout", ["Long format (subject, time, value)", "Wide format (columns = time points)"], horizontal=True, key="av_up_rm_fmt")
                    if fmt_rm == "Long format (subject, time, value)":
                        all_cols = list(df.columns)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            s_col = st.selectbox("Subject ID", all_cols, index=0, key="av_up_rm_s")
                        with c2:
                            t_col = st.selectbox("Time / Condition", [c for c in all_cols if c != s_col], index=0, key="av_up_rm_t")
                        with c3:
                            v_col = st.selectbox("Value", numeric_cols, key="av_up_rm_v")
                        if st.button("Run Repeated-Measures ANOVA", type="primary", use_container_width=True, key="btn_up_rm_long"):
                            piv = df.pivot_table(index=s_col, columns=t_col, values=v_col, aggfunc="first").dropna()
                            if piv.shape[1] < 2 or piv.shape[0] < 2:
                                st.warning("Dữ liệu cần ít nhất 2 thời điểm và 2 đối tượng hợp lệ.")
                            else:
                                conds_up = [(str(c), piv[c].values) for c in piv.columns]
                                _run_rm_anova_pipeline(conds_up)
                    else:
                        sel_cols = st.multiselect("Select time/condition columns (≥ 2)", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="av_up_rm_cols")
                        if st.button("Run Repeated-Measures ANOVA", type="primary", use_container_width=True, key="btn_up_rm_wide"):
                            if len(sel_cols) < 2:
                                st.warning("Vui lòng chọn ít nhất 2 cột thời điểm.")
                            else:
                                sub_df = df[sel_cols].apply(pd.to_numeric, errors="coerce").dropna()
                                if len(sub_df) < 2:
                                    st.warning("Cần ít nhất 2 đối tượng có đủ số liệu ở tất cả các cột.")
                                else:
                                    conds_up = [(str(c), sub_df[c].values) for c in sel_cols]
                                    _run_rm_anova_pipeline(conds_up)

                else:  # Two-way ANOVA
                    v_col = st.selectbox("Dependent variable", numeric_cols, key="av_up_2w_v")
                    cat_cols = categorical_candidate_cols(df, exclude=[v_col])
                    c1, c2 = st.columns(2)
                    with c1:
                        fa_col = st.selectbox("Factor A", cat_cols, index=0, key="av_up_2w_a")
                    with c2:
                        fb_col = st.selectbox("Factor B", [c for c in cat_cols if c != fa_col], index=0, key="av_up_2w_b")
                    if st.button("Run Two-Way ANOVA", type="primary", use_container_width=True, key="btn_up_2w"):
                        _run_twoway_anova_pipeline(df, fa_col, fb_col, v_col)

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
# MÔ-ĐUN: AI TRỢ LÝ THÔNG MINH (FIX TRIỆT ĐỂ BẢNG WEB & WORD, GIỮ NGUYÊN CÔNG THỨC EQUATION)
# -----------------------------
elif section == "AI Trợ lý" and sub == "Giải toán & Trắc nghiệm":
  st.markdown(
      "## 🤖 AI Trợ lý: Giải bài toán Thống kê Y sinh & Tạo trắc nghiệm"
  )

  # CSS hiển thị web phông Times New Roman 13 (18px), đen tuyền, rõ nét
  st.markdown(
      """
    <style>
    .katex { font-size: 1.15em !important; }
    .ai-doc-view {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 18px !important;
        color: #000000 !important;
        line-height: 1.5 !important;
        background: #ffffff !important;
        padding: 25px 30px !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07) !important;
    }
    .ai-doc-view table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 18px 0 !important;
    }
    .ai-doc-view th, .ai-doc-view td {
        border: 1px solid #94a3b8 !important;
        padding: 8px 14px !important;
        text-align: center !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .ai-doc-view th {
        background-color: #f1f5f9 !important;
        font-weight: bold !important;
    }
    .ai-doc-view h1, .ai-doc-view h2, .ai-doc-view h3, .ai-doc-view h4 {
        font-family: 'Times New Roman', Times, serif !important;
        color: #0B3A66 !important;
        font-weight: 800 !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
    }
    .ai-doc-view p, .ai-doc-view li, .ai-doc-view span {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 18px !important;
        color: #000000 !important;
        font-weight: 500 !important;
    }
    .ai-doc-view strong {
        font-weight: 800 !important;
        color: #000000 !important;
    }
    </style>
    """,
      unsafe_allow_html=True,
  )

  # Nút dán ảnh từ Clipboard
  try:
    from streamlit_paste_button import paste_image_button

    has_paste = True
  except ImportError:
    has_paste = False

  # Hàm tự động quét toàn bộ API Key được cấu hình trong Secrets
  def get_all_configured_api_keys() -> List[str]:
    keys = []
    sec_keys = st.secrets.get("GEMINI_API_KEYS", None)
    if sec_keys is not None:
      if isinstance(sec_keys, (list, tuple)):
        keys.extend([str(k).strip() for k in sec_keys if str(k).strip()])
      elif isinstance(sec_keys, str):
        keys.extend(
            [str(k).strip() for k in sec_keys.split(",") if str(k).strip()]
        )

    single_key = st.secrets.get("GEMINI_API_KEY", "")
    if single_key:
      if isinstance(single_key, (list, tuple)):
        keys.extend([str(k).strip() for k in single_key if str(k).strip()])
      elif isinstance(single_key, str):
        if "," in single_key:
          keys.extend([k.strip() for k in single_key.split(",") if k.strip()])
        else:
          keys.append(single_key.strip())

    seen = set()
    ordered = []
    for k in keys:
      if k not in seen:
        seen.add(k)
        ordered.append(k)
    return ordered

  # Hàm xử lý chuỗi: Bảo toàn nguyên vẹn bảng biểu và chuẩn hóa cho Word
  def format_clean_markdown_for_docx(text: str) -> str:
    import re

    # 1. Bắt buộc cắt bỏ toàn bộ lời chào/mở bài trước PHẦN 1 hoặc BÀI GIẢI
    idx_p1 = re.search(
        r"(#{1,4}\s*(PHẦN\s*1|BÀI\s*GIẢI|BỘ\s*CÂU\s*HỎI)|(PHẦN\s*1|BÀI\s*GIẢI|BỘ\s*CÂU\s*HỎI)\s*:)",
        text,
        re.IGNORECASE,
    )
    if idx_p1:
      cleaned_text = text[idx_p1.start() :]
    else:
      cleaned_text = text

    # 2. Tự động in đậm "Câu 1:", "Câu 2:" nếu AI chưa in đậm
    cleaned_text = re.sub(
        r"(?<!\*\*)(Câu\s+\d+[:\.])(?!\*\*)", r"**\1**", cleaned_text
    )

    # 3. Tách tiêu đề bảng nếu vô tình nằm cùng 1 dòng với hàng đầu tiên của bảng
    cleaned_text = re.sub(
        r"^(?!\|)(.*?[:*]+)\s*(\|[^\n]+\|)\s*$",
        r"\1\n\n\2",
        cleaned_text,
        flags=re.MULTILINE,
    )

    # 4. Xử lý trường hợp Tóm tắt đề bài bị viết dính liền trên 1 dòng duy nhất
    def repl_summary(match):
      content = match.group(1)
      items = re.split(r"\s+-\s+(?=[A-ZÀ-Ỹa-z\d\s\(\)]+:)", content)
      res = "Tóm tắt đề bài:\n\n"
      for it in items:
        it = it.strip().lstrip("-").strip()
        if it:
          res += f"* {it}\n"
      return res

    cleaned_text = re.sub(
        r"Tóm tắt đề bài:\s*-\s*(.+?)(?=\n|$)", repl_summary, cleaned_text
    )

    # 5. Phân tích từng dòng và tạo ngắt đoạn chuẩn
    lines = cleaned_text.split("\n")
    cleaned = []
    in_table = False
    in_list = False

    for line in lines:
      s = line.strip()

      # XỬ LÝ BẢNG BIỂU: Giữ nguyên vẹn 100% nội dung hàng bảng, đảm bảo có dòng trống trước/sau
      if s.startswith("|") and s.endswith("|"):
        if not in_table:
          if cleaned and cleaned[-1] != "":
            cleaned.append("")
          in_table = True
        cleaned.append(s)
        continue
      else:
        if in_table:
          cleaned.append("")
          in_table = False

      # Bỏ qua các dòng rác chứa ký tự đơn lẻ
      if not s or s in ["*", "-", "•", "**", "--", "o", "▪"]:
        if in_list:
          cleaned.append("")
          in_list = False
        continue

      # Khử bullet đặt sai vị trí trước công thức hoặc tiêu đề
      s = re.sub(r"^[\*\-•o▪]\s*(?=\$\$|\$)", "", s)
      s = re.sub(r"^[\*\-•o▪]\s*(\d+\.\s+)", r"\1", s)
      s = re.sub(r"^[\*\-•o▪]\s*(#{1,4}\s+)", r"\1", s)
      s = re.sub(r"^[\*\-•o▪]\s*(\*\*[A-D]\.\*\*|[A-D]\.\s+)", r"\1", s)

      # Danh sách bullet chuẩn
      if re.match(r"^[\*\-•]\s+", s):
        if not in_list:
          cleaned.append("")
          in_list = True
        bullet_body = re.sub(r"^[\*\-•]\s+", "", s)
        cleaned.append(f"* {bullet_body}")
        continue
      else:
        if in_list:
          cleaned.append("")
          in_list = False

      # Tách đoạn cho tiêu đề, câu hỏi, phương án trắc nghiệm, công thức
      if re.match(
          r"^(\*\*Câu\s+\d+[:\.]\*\*|Câu\s+\d+[:\.]|[A-D]\.\s+|\d+\.\s+|#{1,4}\s+)",
          s,
      ):
        cleaned.append("")
        cleaned.append(s)
        cleaned.append("")
      elif s.startswith("$$"):
        cleaned.append("")
        cleaned.append(s)
        cleaned.append("")
      else:
        cleaned.append(s)

    if in_table:
      cleaned.append("")

    res = "\n".join(cleaned)
    res = re.sub(r"\n{3,}", "\n\n", res)
    return res

  # Hàm chuyển đổi riêng cho Web: Biến ### thành <h3> để không bao giờ bị hiện chữ ###
  def render_markdown_to_web_html(text: str) -> str:
    import re

    web_text = format_clean_markdown_for_docx(text)
    web_text = re.sub(
        r"^###\s+(.+)$",
        r'<h3 style="color:#0B3A66; font-weight:800; margin-top:16px;'
        r' margin-bottom:8px;">\1</h3>',
        web_text,
        flags=re.MULTILINE,
    )
    web_text = re.sub(
        r"^##\s+(.+)$",
        r'<h2 style="color:#0B3A66; font-weight:800; margin-top:20px;'
        r' margin-bottom:10px;">\1</h2>',
        web_text,
        flags=re.MULTILINE,
    )
    web_text = re.sub(
        r"^#\s+(.+)$",
        r'<h1 style="color:#0B3A66; font-weight:800; margin-top:24px;'
        r' margin-bottom:12px;">\1</h1>',
        web_text,
        flags=re.MULTILINE,
    )
    return web_text

  # Hàm xuất file Word (.docx) chứa công thức Equation, Bảng biểu chuẩn Table Grid và dãn dòng 1.2
  def generate_word_docx_with_equations(md_text: str) -> Optional[bytes]:
    import os
    import tempfile

    try:
      import pypandoc

      with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_docx = tmp.name

      ready_md = format_clean_markdown_for_docx(md_text)

      pypandoc.convert_text(
          ready_md,
          "docx",
          format="markdown+tex_math_dollars",
          outputfile=tmp_docx,
      )

      try:
        from docx import Document
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt

        doc = Document(tmp_docx)
        for s in doc.sections:
          s.top_margin = Cm(2.0)
          s.bottom_margin = Cm(2.0)
          s.left_margin = Cm(2.0)
          s.right_margin = Cm(2.0)

        style = doc.styles["Normal"]
        style.font.name = "Times New Roman"
        style.font.size = Pt(13)

        for p in doc.paragraphs:
          p.paragraph_format.line_spacing = 1.2
          p.paragraph_format.space_after = Pt(2)
          p.paragraph_format.space_before = Pt(0)
          for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(13)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

        # Định dạng bảng biểu chuẩn: kẻ viền đen Table Grid, căn giữa và phông chữ 12pt
        for t in doc.tables:
          try:
            t.style = "Table Grid"
          except Exception:
            pass
          t.alignment = WD_TABLE_ALIGNMENT.CENTER
          for row in t.rows:
            for cell in row.cells:
              for cp in cell.paragraphs:
                cp.paragraph_format.line_spacing = 1.15
                cp.paragraph_format.space_after = Pt(2)
                cp.paragraph_format.space_before = Pt(2)
                for cr in cp.runs:
                  cr.font.name = "Times New Roman"
                  cr.font.size = Pt(12)

        doc.save(tmp_docx)
      except Exception:
        pass

      with open(tmp_docx, "rb") as f:
        data = f.read()
      os.remove(tmp_docx)
      return data
    except Exception:
      return None

  if "ai_solution_text" not in st.session_state:
    st.session_state["ai_solution_text"] = ""

  c1, c2 = st.columns(2)
  with c1:
    txt_input = st.text_area(
        "Nhập hoặc dán văn bản đề bài:",
        value=(
            "Một nghiên cứu đánh giá một chương trình can thiệp kiểm soát đái"
            " tháo đường. Sau can thiệp, nhóm can thiệp có n1 = 40 đối tượng,"
            " x̄1 = 6.8% và s1 = 0.9%; nhóm chứng có n2 = 40 đối tượng, x̄2 ="
            " 7.4% và s2 = 1%. Nhà nghiên cứu muốn ước lượng hiệu trung bình"
            " bằng KTC 95%."
        ),
        height=140,
    )
  with c2:
    st.markdown("**Ảnh chụp đề bài:**")
    pasted_image = None
    if has_paste:
      paste_res = paste_image_button(
          label="📋 Bấm vào đây để Dán ảnh từ Clipboard (Ctrl + V)",
          text_color="#ffffff",
          background_color="#0B3A66",
          hover_background_color="#1E40AF",
          key="btn_paste_ai_image_fix",
      )
      if paste_res.image_data is not None:
        pasted_image = paste_res.image_data

    img_file = st.file_uploader(
        "Hoặc tải ảnh từ máy tính (PNG, JPG):",
        type=["png", "jpg", "jpeg"],
        key="uploader_ai_img",
    )

    final_image = None
    if pasted_image is not None:
      final_image = pasted_image
      st.image(
          final_image,
          caption="Đã nhận ảnh dán từ Clipboard",
          use_container_width=True,
      )
    elif img_file is not None:
      final_image = Image.open(img_file)
      st.image(
          final_image,
          caption="Đã nhận ảnh tải lên từ máy tính",
          use_container_width=True,
      )

  c_cfg1, c_cfg2, c_cfg3 = st.columns([1.5, 1, 1])
  with c_cfg1:
    action_mode = st.radio(
        "Yêu cầu xử lý:",
        [
            "Giải chi tiết bài toán",
            "Tạo câu hỏi trắc nghiệm A, B, C, D",
            "Cả giải chi tiết và tạo trắc nghiệm",
        ],
        horizontal=False,
    )
  with c_cfg2:
    if action_mode == "Giải chi tiết bài toán":
      st.caption("ℹ️ Chế độ chỉ giải bài tập, không tạo trắc nghiệm.")
      num_questions = 0
    else:
      num_questions = st.number_input(
          "Số câu trắc nghiệm:", min_value=1, max_value=20, value=4, step=1
      )
  with c_cfg3:
    model_choice = st.selectbox(
        "Mô hình AI:",
        ["gemini-3-flash-preview", "gemini-3.8-flash"],
        index=0,
    )
    target_model_name = model_choice.split()[0]

  c_btn1, c_btn2 = st.columns([2, 1])
  with c_btn1:
    btn_run = st.button(
        "🚀 Bắt đầu phân tích với AI", type="primary", use_container_width=True
    )
  with c_btn2:
    if st.session_state["ai_solution_text"]:
      if st.button("🗑️ Làm bài mới / Xóa", use_container_width=True):
        st.session_state["ai_solution_text"] = ""
        st.rerun()

  # Xử lý khi nhấn nút
  if btn_run:
    all_keys = get_all_configured_api_keys()

    if not txt_input.strip() and final_image is None:
      st.warning("Vui lòng cung cấp văn bản hoặc hình ảnh đề bài.")
    elif not all_keys:
      st.error(
          "Chưa cấu hình GEMINI_API_KEY hoặc GEMINI_API_KEYS trong Streamlit"
          " Secrets."
      )
    else:
      core_rules = """
Bạn là chuyên gia Thống kê Y học và giảng viên bộ môn Xác suất Thống kê Y Dược.
Nhiệm vụ: Giải bài toán theo đúng các quy chuẩn sau đây.

CÁC NGUYÊN TẮC BẮT BUỘC TUÂN THỦ:
1. KHÔNG LỜI CHÀO HỎI: Bắt đầu câu trả lời ngay lập tức bằng tiêu đề Markdown (###). Tuyệt đối không chào hỏi, không giới thiệu danh xưng.
2. TÓM TẮT ĐỀ BÀI: Dưới mục "Tóm tắt đề bài:", xuống dòng và đặt dấu '* ' ở đầu mỗi dòng liệt kê để tạo bullet tròn đen.
3. BẢNG DỮ LIỆU / BẢNG LIÊN ĐỊNH: BẮT BUỘC phải để một dòng trống trước và sau bảng Markdown. Bảng phải có đầy đủ các cột và các hàng (ô quan sát, tổng hàng, tổng cột). Mỗi hàng của bảng phải nằm trọn vẹn trên 1 dòng duy nhất dạng: | Cột 1 | Cột 2 | Cột 3 |
4. CÔNG THỨC TOÁN DẠNG CHUỖI 3 VẾ GỌN GÀNG TRÊN 1 DÒNG DUY NHẤT:
   - Đặt trong $$...$$ ở dòng riêng biệt: $$Tên = Công\\ thức\\ chữ = Ráp\\ số = Đáp\\ số$$
   - Không chia nhỏ các bước tính toán phụ vụn vặt.
   - Khoảng tin cậy KTC (nếu có) bắt buộc đặt trong ngoặc vuông: [cận dưới; cận trên] (ngăn cách bằng dấu chấm phẩy ';').
"""

      if action_mode == "Giải chi tiết bài toán":
        mode_prompt = """
CHẾ ĐỘ YÊU CẦU: CHỈ GIẢI CHI TIẾT BÀI TOÁN.
TUYỆT ĐỐI KHÔNG TẠO CÂU HỎI TRẮC NGHIỆM, KHÔNG CÓ PHẦN 2.

Cấu trúc trình bày:
### PHẦN 1: BÀI GIẢI CHI TIẾT
- Tóm tắt đề bài & Bảng dữ liệu (nếu có).
- Xác định phương pháp: Nêu dạng bài và công thức chuẩn áp dụng.
- Các bước tính toán: Trình bày chuỗi công thức 3 vế trực diện, chính xác.
- Kết luận: Nêu đáp số và ý nghĩa thực tế/lâm sàng.
"""
      elif action_mode == "Tạo câu hỏi trắc nghiệm A, B, C, D":
        mode_prompt = f"""
CHẾ ĐỘ YÊU CẦU: CHỈ TẠO BỘ CÂU HỎI TRẮC NGHIỆM ĐỘC LẬP.
TUYỆT ĐỐI KHÔNG GIẢI BÀI TOÁN, KHÔNG CÓ PHẦN BÀI GIẢI CHI TIẾT.

Cấu trúc trình bày:
### BỘ CÂU HỎI TRẮC NGHIỆM ĐỘC LẬP
- Soạn đúng {num_questions} câu hỏi trắc nghiệm độc lập hoàn toàn.
- TỰ ĐỦ DỮ LIỆU: BẮT BUỘC lặp lại tóm tắt đầy đủ các số liệu đề bài trong phần dẫn của TỪNG CÂU HỎI. Không viết "theo nghiên cứu trên", "theo câu 1".
- BẮT BUỘC IN ĐẬM TÊN CÂU: **Câu 1:**, **Câu 2:**, **Câu 3:**...
- ĐÁP ÁN: PHƯƠNG ÁN A LUÔN LUÔN LÀ ĐÁP ÁN ĐÚNG. B, C, D là các phương án sai.
- TUYỆT ĐỐI KHÔNG ghi dòng "Đáp án đúng" và TUYỆT ĐỐI KHÔNG ghi dòng "Giải thích".
- Mỗi phương án A, B, C, D nằm riêng biệt trên một dòng:
  **Câu [X]:** [Câu dẫn đầy đủ số liệu đề bài] [Nội dung câu hỏi]?

  A. [Phương án đúng]

  B. [Phương án sai]

  C. [Phương án sai]

  D. [Phương án sai]
"""
      else:
        mode_prompt = f"""
CHẾ ĐỘ YÊU CẦU: CẢ GIẢI CHI TIẾT VÀ TẠO CÂU HỎI TRẮC NGHIỆM.

### PHẦN 1: BÀI GIẢI CHI TIẾT
- Tóm tắt đề bài & Bảng dữ liệu (nếu có).
- Xác định phương pháp: Nêu dạng bài và công thức chuẩn áp dụng.
- Các bước tính toán: Trình bày chuỗi công thức 3 vế trực diện, chính xác.
- Kết luận: Nêu đáp số và ý nghĩa thực tế/lâm sàng.

### PHẦN 2: BỘ CÂU HỎI TRẮC NGHIỆM ĐỘC LẬP
- Soạn đúng {num_questions} câu hỏi trắc nghiệm độc lập hoàn toàn.
- TỰ ĐỦ DỮ LIỆU trong từng câu hỏi (lặp lại số liệu đề bài).
- BẮT BUỘC IN ĐẬM TÊN CÂU: **Câu 1:**, **Câu 2:**...
- PHƯƠNG ÁN A LUÔN LUÔN LÀ ĐÁP ÁN ĐÚNG. B, C, D là các phương án sai.
- TUYỆT ĐỐI KHÔNG ghi dòng "Đáp án đúng" và TUYỆT ĐỐI KHÔNG ghi dòng "Giải thích".
- Mỗi phương án A, B, C, D nằm riêng biệt trên một dòng:
  **Câu [X]:** [Câu dẫn đầy đủ số liệu đề bài] [Nội dung câu hỏi]?

  A. [Phương án đúng]

  B. [Phương án sai]

  C. [Phương án sai]

  D. [Phương án sai]
"""

      prompt = core_rules + "\n" + mode_prompt

      parts = [prompt]
      if txt_input.strip():
        parts.append(f"ĐỀ BÀI:\n{txt_input}")
      if final_image is not None:
        img_to_send = final_image.copy()
        if max(img_to_send.size) > 1000:
          img_to_send.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        parts.append(img_to_send)

      status_box = st.empty()
      success = False

      for idx, api_key in enumerate(all_keys):
        try:
          genai.configure(api_key=api_key)
          try:
            model = genai.GenerativeModel(target_model_name)
          except Exception:
            model = genai.GenerativeModel("gemini-3-flash-preview")

          try:
            fast_config = {
                "temperature": 0.1,
                "thinking_config": {"thinking_budget": 0},
            }
            response = model.generate_content(
                parts, stream=True, generation_config=fast_config
            )
          except Exception:
            response = model.generate_content(
                parts, stream=True, generation_config={"temperature": 0.1}
            )

          status_box.empty()
          st.markdown("---")
          st.markdown("### 📋 Kết quả phân tích từ AI:")

          raw_text = ""
          placeholder = st.empty()
          for chunk in response:
            if chunk.text:
              raw_text += chunk.text
              formatted_web = render_markdown_to_web_html(raw_text)
              placeholder.markdown(
                  f'<div class="ai-doc-view">{formatted_web}</div>',
                  unsafe_allow_html=True,
              )

          st.session_state["ai_solution_text"] = format_clean_markdown_for_docx(
              raw_text
          )
          success = True
          break

        except Exception as e:
          err_str = str(e)
          if "429" in err_str or "quota" in err_str.lower():
            if idx + 1 < len(all_keys):
              status_box.info(
                  f"🔄 Đang tự động chuyển sang Key dự phòng số {idx + 2}..."
              )
              continue
            else:
              status_box.error(
                  "⚠️ Tất cả các API Key hiện có đều đã dùng hết hạn mức hôm nay"
                  " (20 lượt/ngày/project)!\n\n👉 Hãy vào"
                  " https://aistudio.google.com/apikey, bấm **Create API key**"
                  " và chọn **Create API key in new project** rồi dán vào"
                  " Secrets để tiếp tục."
              )
              break
          else:
            status_box.error(f"Lỗi: {e}")
            break

  # HIỂN THỊ KẾT QUẢ ĐÃ LƯU & CÁC NÚT TẢI VỀ
  if st.session_state["ai_solution_text"]:
    if not btn_run:
      st.markdown("---")
      st.markdown("### 📋 Kết quả phân tích từ AI:")
      formatted_web = render_markdown_to_web_html(
          st.session_state["ai_solution_text"]
      )
      st.markdown(
          f'<div class="ai-doc-view">{formatted_web}</div>',
          unsafe_allow_html=True,
      )

    st.write("")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
      docx_bytes = generate_word_docx_with_equations(
          st.session_state["ai_solution_text"]
      )
      if docx_bytes:
        st.download_button(
            "📥 Tải tài liệu Word chuẩn Equation (.docx)",
            data=docx_bytes,
            file_name="Ket_qua_Thong_ke.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
      else:
        st.info(
            "💡 Cần có 'pypandoc-binary' và 'python-docx' trong requirements.txt"
            " để tải file Word Equation."
        )

    with col_dl2:
      st.download_button(
          "📄 Tải dạng văn bản thuần (.txt)",
          data=st.session_state["ai_solution_text"],
          file_name="Ket_qua_Thong_ke.txt",
          mime="text/plain",
          use_container_width=True,
      )
