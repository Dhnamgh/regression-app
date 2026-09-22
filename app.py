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

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multicomp import pairwise_tukeyhsd

try:
    from statsmodels.stats.diagnostic import lilliefors
except ImportError:
    lilliefors = None

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="Data Analysis in Health Sciences",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
/* Sidebar background */
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
  font-weight: 700 !important;
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
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div > div{
  background: rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
  border: 1px dashed rgba(255,255,255,0.35) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{
  color: #ffffff !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button{
  background: rgba(255,255,255,0.14) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.40) !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
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
}
.header-banner p{
  margin: 8px 0 0 0;
  opacity: 0.90;
  font-size: 15px;
}

div[data-testid="stDataFrame"]{
  font-size: 21px;
  font-weight: 400;
}
div[data-testid="stDataFrame"] *{
  font-size: 20px !important;
  font-weight: 400 !important;
}
[data-testid="stTable"] table{
  font-size: 21px !important;
  font-weight: 400 !important;
}

.analysis-table-wrap{
  width: 100%;
  overflow-x: auto;
  margin: 0.5rem 0 1.2rem 0;
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
  padding: 12px 14px;
  text-align: left;
  white-space: normal;
  line-height: 1.25;
}
.analysis-table td{
  border: 1.2px solid #e2e8f0;
  padding: 12px 14px;
  color: #0f172a;
  font-weight: 400;
  line-height: 1.35;
  white-space: normal;
}
.analysis-table tbody tr:nth-child(even){
  background: #f8fafc;
}
</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# Header
# =========================================================
st.markdown(
    """
<div class="header-banner">
  <h1>Data Analysis in Health Sciences</h1>
  <p>Regression, categorical analysis, quantitative tests and diagnostics for health sciences.</p>
</div>
""",
    unsafe_allow_html=True
)
st.write("")

# =========================================================
# Smart Rounding & Formatting Functions
# =========================================================
def smart_round_val(val, min_dec: int = 3) -> str:
    """Làm tròn đến 3 chữ số thập phân. Nếu khác 0 mà làm tròn ra 0 thì tự mở rộng 4, 5,... chữ số."""
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
    if p < 0.0001:
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
# Download & Copy Helpers
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
    st.markdown(
        f"""
<div class="analysis-table-wrap">
{html}
</div>
""",
        unsafe_allow_html=True
    )

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
# Confidence Intervals Functions
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

# =========================================================
# Categorical helpers
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
# Logistic regression helpers
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
    case_summary["Percent"] = case_summary["Percent"].round(1)

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
# Navigation State
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

    with st.expander("Confidence Intervals", expanded=(st.session_state.section == "Confidence Intervals")):
        if st.button("Mean, SD & Variance CI", key="ci_1", use_container_width=True):
            set_nav("Confidence Intervals", "Mean & Variance")

# =========================================================
# Routing Pages
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
        if st.button("Run Chi-square", type="primary", use_container_width=True):
            try:
                obs = get_observed_matrix(observed_df)
                chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)
                chi_tbl = pd.DataFrame([["Pearson Chi-Square", chi2, dof, format_p_value(p), "Yes" if p < 0.05 else "No"]],
                                       columns=["Test", "Value", "df", "Asymp. Sig. (2-sided)", "Significant (p<0.05)"])
                chi_tbl = compact_numeric_df(chi_tbl, decimals=3)
                show_table(chi_tbl, "Chi-Square Tests")
                download_table_block(chi_tbl, "chisq_tests", "Chi-Square Tests")
                if obs.shape == (2, 2):
                    meas = two_by_two_measures(obs, alpha=0.05)
                    show_table(meas, "2×2 Measures")
                    download_table_block(meas, "chisq_2x2_measures", "2×2 Measures")
            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Fisher 2×2":
        st.markdown("## Categorical Tests — Fisher's Exact Test (2×2)")
        counts_df, observed_df = contingency_editor("fisher", ["Group 1", "Group 2"], ["Outcome +", "Outcome -"], np.array([[10, 30], [20, 15]]))
        show_table(counts_df, "Observed Frequencies")
        if st.button("Run Fisher's Exact", type="primary", use_container_width=True):
            try:
                obs = require_2x2(observed_df)
                oddsratio, p = stats.fisher_exact(obs, alternative="two-sided")
                tbl = compact_numeric_df(pd.DataFrame([["Fisher's Exact Test", oddsratio, format_p_value(p), "Yes" if p < 0.05 else "No"]],
                                                      columns=["Test", "Odds Ratio", "Exact Sig. (2-sided)", "Significant (p<0.05)"]), decimals=3)
                show_table(tbl, "Fisher's Exact Test")
                download_table_block(tbl, "fisher_exact", "Fisher's Exact")
            except Exception as e:
                st.error(f"Failed: {e}")

# -----------------------------
# CONFIDENCE INTERVALS — MEAN, SD & VARIANCE
# -----------------------------
elif section == "Confidence Intervals" and sub == "Mean & Variance":
    st.markdown("## Confidence Intervals — Mean & Variance")

    template = pd.DataFrame({"X": [1.2, 2.0, 1.8, 2.2, 1.6]})
    st.download_button(
        "Download Excel template",
        data=df_to_excel_bytes({"ci_template": template}),
        file_name="ci_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False
    )

    method = st.radio("Input method", ["Upload file (template)", "Paste values"], horizontal=True)

    x = None
    if method == "Upload file (template)":
        up = st.file_uploader("Upload CI template (XLSX/CSV)", type=["xlsx", "csv"], key="ci_upload")
        if up is not None:
            df = load_uploaded_file(up)
            st.dataframe(df.head(50), use_container_width=True)
            if "X" not in df.columns:
                st.error("Template must have a column named 'X'.")
            else:
                x = pd.to_numeric(df["X"], errors="coerce").dropna().values
    else:
        txt = st.text_area(
            "Paste numeric values (separated by ; , space or newline)",
            value="3;5;6;8;4;5;8;5;9",
            height=100
        )
        if txt.strip():
            # Chuẩn hóa triệt để dấu chấm phẩy, dấu phẩy, khoảng trắng
            normalized = txt.replace(";", " ").replace(",", " ")
            parts = [p for p in normalized.split() if p.strip()]
            vals = pd.to_numeric(pd.Series(parts), errors="coerce").dropna()
            x = vals.values

    conf_level = st.slider("Confidence level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    alpha_tail = 1.0 - conf_level

    force_boot = st.checkbox("Force bootstrap (recommended if non-normal)", value=False)
    n_boot = st.number_input("Bootstrap resamples", min_value=1000, max_value=20000, value=5000, step=500)

    st.markdown("### Confidence Interval Results")
    if st.button("Compute CI", type="primary", use_container_width=True):
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

                q1 = float(np.percentile(x, 25))
                q3 = float(np.percentile(x, 75))
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                has_outliers = "Có" if np.any((x < lower_bound) | (x > upper_bound)) else "Không"

                mode_res = stats.mode(x, keepdims=True)
                mode_v = mode_res.mode[0] if len(mode_res.mode) > 0 else np.nan

                # =========================================================
                # BẢNG 1: THỐNG KÊ MÔ TẢ (Descriptive Statistics)
                # =========================================================
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

                # =========================================================
                # BẢNG 2: KIỂM ĐỊNH CHUẨN VÀ OUTLIER (Normality & Diagnostics)
                # =========================================================
                # Shapiro-Wilk
                if 3 <= n <= 5000:
                    sw_stat, sw_p = stats.shapiro(x)
                else:
                    sw_stat, sw_p = np.nan, np.nan

                # Kolmogorov-Smirnov (Lilliefors corrected nếu có statsmodels, nếu không dùng kstest)
                if lilliefors is not None and n >= 4:
                    ks_stat, ks_p = lilliefors(x, dist='norm')
                else:
                    ks_res = stats.kstest(x, 'norm', args=(mean_v, s_v))
                    ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)

                # Quyết định phân phối chuẩn dựa trên Shapiro-Wilk (tiêu chuẩn cho n < 50)
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

                # =========================================================
                # BẢNG 3: BẢNG ƯỚC LƯỢNG KHOẢNG TIN CẬY GỘP DUY NHẤT (CI Estimates)
                # =========================================================
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
