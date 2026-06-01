import io
import math
import os
import textwrap
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

hide_streamlit_style = """
<style>

#MainMenu {display:none !important;}
footer {display:none !important;}
header {display:none !important;}

.stDeployButton {display:none !important;}

[data-testid="stToolbar"] {display:none !important;}
[data-testid="stDecoration"] {display:none !important;}
[data-testid="stStatusWidget"] {display:none !important;}

.viewerBadge_container__1QSob {display:none !important;}

</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ===== LOGIN =====
def check_password():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 Đăng nhập")

    password = st.text_input(
        "Nhập mật khẩu",
        type="password"
    )

    if st.button("Đăng nhập"):

        if password == st.secrets["APP_PASSWORD"]:
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
#MainMenu {display:none !important;}
header {display:none !important;}
footer {display:none !important;}
[data-testid="stToolbar"] {display:none !important;}
[data-testid="stDecoration"] {display:none !important;}
[data-testid="stStatusWidget"] {display:none !important;}
[data-testid="stFooter"] {display:none !important;}
.viewerBadge_container__1QSob {display:none !important;}
.viewerBadge_link__1S137 {display:none !important;}
.viewerBadge_container__r5tak {display:none !important;}
a[href*="streamlit.io"] {display:none !important;}
/* Sidebar background */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0B3A66 0%, #0A2D4E 100%);
}

/* Sidebar text (safe) */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div{
  color: #ffffff !important;
}

/* Sidebar buttons */
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

/* Sidebar expander header */
section[data-testid="stSidebar"] details summary{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  font-weight: 800 !important;
}

/* File uploader style in sidebar */
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

/* Header banner */
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

/* Larger readable output tables */
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
  font-size: 22px;
  color: #0f172a;
  background: #ffffff;
}
.analysis-table th{
  background: #f3f4f6;
  color: #0f172a;
  font-weight: 900;
  border: 1.5px solid #cbd5e1;
  padding: 14px 16px;
  text-align: left;
  white-space: normal;
  line-height: 1.25;
}
.analysis-table td{
  border: 1.2px solid #dbe3ef;
  padding: 14px 16px;
  color: #0f172a;
  font-weight: 400;
  line-height: 1.35;
  white-space: normal;
}
.analysis-table tbody tr:nth-child(even){
  background: #fafafa;
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
# Download helpers
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
    import matplotlib.pyplot as plt
    import io
    import textwrap

    # 1. Tiền xử lý dữ liệu
    df_plot = df.copy().fillna("-").astype(str)
    n_rows, n_cols = df_plot.shape

    # 2. CẤU HÌNH NGẮT DÒNG (WRAPPING)
    width_limit = 15 
    
    def get_line_count(text, width):
        if not text or text == "-": return 1
        lines = textwrap.wrap(text, width=width)
        return len(lines) if len(lines) > 0 else 1

    # 3. TÍNH TOÁN SỐ DÒNG THỰC TẾ TRONG TỪNG HÀNG
    # Tính cho Header
    header_max_lines = max([get_line_count(col, width_limit) for col in df_plot.columns])
    
    # Tính cho từng hàng dữ liệu
    row_line_counts = []
    for _, row in df_plot.iterrows():
        max_lines_in_row = max([get_line_count(val, width_limit) for val in row])
        row_line_counts.append(max_lines_in_row)

    # 4. THIẾT LẬP KÍCH THƯỚC HÌNH ẢNH (CHỐNG CO CHỮ)
    line_unit_height = 0.35  # Chiều cao mỗi dòng chữ (inch)
    padding = 0.5            # Khoảng đệm an toàn mỗi ô
    
    # h_height: Chiều cao vùng header
    h_height = (header_max_lines * line_unit_height) + padding
    # b_height: Tổng chiều cao các hàng dữ liệu
    b_height = sum([(lc * line_unit_height) + padding for lc in row_line_counts])
    
    # Chiều rộng ảnh tỉ lệ với số cột
    fig_width = max(12, n_cols * 2.5)
    # Chiều cao ảnh = Header + Data + Lề cho Tiêu đề chính
    fig_height = h_height + b_height + 2.5 

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # 5. CHUẨN BỊ NỘI DUNG ĐÃ WRAP
    wrapped_headers = [textwrap.fill(col, width=width_limit) for col in df_plot.columns]
    wrapped_data = []
    for _, row in df_plot.iterrows():
        wrapped_data.append([textwrap.fill(val, width=width_limit) for val in row])

    # 6. VẼ BẢNG
    table = ax.table(
        cellText=wrapped_data,
        colLabels=wrapped_headers,
        cellLoc='center',
        loc='center'
    )

    # 7. CẤU HÌNH CHI TIẾT Ô 
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    
    for i in range(n_cols):
        # Thiết lập chiều cao Header 
        header_cell = table[0, i]
        header_cell.set_height(h_height / fig_height)
        
        # Thiết lập chiều cao từng hàng dữ liệu
        for j in range(n_rows):
            cell = table[j+1, i]
            # Tỷ lệ chiều cao ô so với tổng chiều cao ảnh
            cell_h = (row_line_counts[j] * line_unit_height + padding) / fig_height
            cell.set_height(cell_h)

    # 8. STYLE
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#333333')
        cell.set_linewidth(1.2)
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')
        else:
            cell.set_facecolor('white')

    if title:
        plt.title(title, fontsize=22, pad=60, weight='bold')

    # 9. XUẤT FILE
    bio = io.BytesIO()
    plt.savefig(bio, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.6)
    plt.close(fig)
    
    return bio.getvalue()

def download_table_block(df: pd.DataFrame, base_name: str, title: str = ""):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Download Excel",
            data=df_to_excel_bytes({base_name: df}),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            on_click="ignore"
        )
    with c2:
        st.download_button(
            "Download PNG",
            data=df_to_png_bytes(df, title=title),
            file_name=f"{base_name}.png",
            mime="image/png",
            use_container_width=True,
            on_click="ignore"
        )

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
# Output formatting
# =========================================================
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
    return f"{p:.3f}"

def clean_cell(x):
    # Make NaN/None blank
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

def compact_numeric_df(df, decimals: int = 4):
    """
    Compact numeric formatting for both DataFrame and Series.
    This avoids errors when the input is a Series and avoids issues with blank strings.
    """
    out = df.copy()

    if isinstance(out, pd.Series):
        if pd.api.types.is_numeric_dtype(out):
            out = out.round(decimals)
        return out.map(clean_cell)

    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)

    return out.apply(lambda col: col.map(clean_cell))

def clean_term_name(s: str) -> str:
    # Remove Q("...") wrappers used by formula
    if isinstance(s, str) and s.startswith('Q("') and s.endswith('")'):
        return s[3:-2]
    return s


# =========================================================
# File loading
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


# =========================================================
# Data storage (separate per module to avoid overwriting)
# =========================================================
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
    """
    Always-visible input panel (per page):
    - Download template
    - Upload CSV/XLSX
    - Store dataset to session_state[store_key]
    """
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
# Contingency table editor (labels editable + Total)
# =========================================================
def contingency_editor(
    key: str,
    default_rows: List[str],
    default_cols: List[str],
    default_counts: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      counts_df: display table with Group + Total row/col
      observed_df: r×c numeric-only table for stats
    """
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
        # Rows
        new_rows = []
        for i, old in enumerate(df["Group"].astype(str).tolist()):
            new_rows.append(st.text_input(f"Row {i+1} label", value=old, key=f"{key}_rowlbl_{i}"))
        df["Group"] = new_rows

        # Cols (excluding Group)
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

    # numeric conversion (counts)
    cat_cols = [c for c in df.columns if c != "Group"]
    for c in cat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(0).astype(int)
        df[c] = df[c].clip(lower=0)

    st.markdown("#### Observed counts (edit cells)")
    edited = st.data_editor(
        df,
        key=f"{key}_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Group": st.column_config.TextColumn("Group")
        } | {
            c: st.column_config.NumberColumn(c, min_value=0, step=1, format="%d")
            for c in cat_cols
        }
    )

    edited = edited.copy()
    edited["Group"] = edited["Group"].astype(str)
    for c in cat_cols:
        edited[c] = pd.to_numeric(edited[c], errors="coerce").fillna(0).round(0).astype(int)
        edited[c] = edited[c].clip(lower=0)

    st.session_state[ss_key] = edited
    observed_df = edited[cat_cols].copy()

    # totals
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

    counts_df, observed_df = contingency_editor(
        key=key,
        default_rows=[f"Group {i+1}" for i in range(default_r)],
        default_cols=[f"Category {j+1}" for j in range(default_c)],
        default_counts=np.ones((default_r, default_c), dtype=int),
    )
    return counts_df, observed_df


# =========================================================
# Categorical pipeline (prevents "Group text" errors)
# =========================================================
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


# =========================================================
# 2×2 measures (OR, RR, VE + basic diagnostic accuracy + CI)
# =========================================================
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

    # continuity correction for OR/RR
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

    # Diagnostic metrics (interpret row0 as Test+, row1 as Test-; col0 outcome+, col1 outcome-)
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
    return compact_numeric_df(df, decimals=4)


# =========================================================
# Logistic regression (statsmodels): Variables in the Equation
# =========================================================
def hosmer_lemeshow_table(y_true, y_prob, g=10):
    """
    Hosmer-Lemeshow test table.
    Groups predicted probabilities into up to g groups and compares observed vs expected.
    """
    tmp = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_prob, dtype=float)})
    tmp = tmp.dropna()

    if tmp.empty or tmp["p"].nunique() < 2:
        raise ValueError("Hosmer-Lemeshow test cannot be computed because predicted probabilities have insufficient variation.")

    q = min(int(g), int(tmp["p"].nunique()), len(tmp))
    tmp["group"] = pd.qcut(tmp["p"], q=q, duplicates="drop")

    rows = []
    chi2 = 0.0

    for i, (_, d) in enumerate(tmp.groupby("group", observed=False), start=1):
        n = int(len(d))
        obs1 = float(d["y"].sum())
        obs0 = float(n - obs1)
        exp1 = float(d["p"].sum())
        exp0 = float(n - exp1)

        if exp1 > 0:
            chi2 += (obs1 - exp1) ** 2 / exp1
        if exp0 > 0:
            chi2 += (obs0 - exp0) ** 2 / exp0

        rows.append([i, obs0, exp0, obs1, exp1, n])

    detail = pd.DataFrame(rows, columns=[
        "Step", "Observed 0", "Expected 0", "Observed 1", "Expected 1", "Total"
    ])

    df_hl = max(len(detail) - 2, 1)
    pval = float(stats.chi2.sf(chi2, df_hl))

    summary = pd.DataFrame([[
        "Hosmer and Lemeshow Test", chi2, df_hl, format_p_value(pval),
        "Yes" if pval < 0.05 else "No"
    ]], columns=["Test", "Chi-square", "df", "Sig.", "Significant (p<0.05)"])

    return summary, detail


def run_logistic_statsmodels(df: pd.DataFrame, target: str, features: List[str], cutoff: float = 0.5) -> Dict[str, object]:
    """
    Fit binary logistic regression and return output tables:
    Case Processing Summary, Dependent Variable Encoding, Omnibus Tests,
    Model Summary, Hosmer-Lemeshow, Classification Table, Variables in the Equation.
    """
    if not features:
        raise ValueError("Please select at least one predictor.")

    raw_n = int(len(df))
    if raw_n == 0:
        raise ValueError("Dataset is empty.")

    data = df[[target] + features].copy()

    # Convert blanks / spaces / non-numeric values to NaN, then drop them.
    for col in data.columns:
        data[col] = data[col].replace(r"^\s*$", np.nan, regex=True)
        data[col] = pd.to_numeric(data[col], errors="coerce")

    missing_any = data.isna().any(axis=1)
    excluded_n = int(missing_any.sum())

    data = data.dropna().copy()
    included_n = int(len(data))

    if included_n < 10:
        raise ValueError("Not enough valid cases after removing missing/non-numeric values. At least 10 valid rows are recommended.")

    unique_y = sorted(data[target].unique())
    if len(unique_y) != 2:
        raise ValueError("Target must have exactly 2 numeric values, preferably 0 and 1.")

    # Convert target to internal 0/1 coding.
    if set(unique_y) == {0, 1}:
        y = data[target].astype(int)
        encoding_rows = [["0", 0], ["1", 1]]
    else:
        mapping = {unique_y[0]: 0, unique_y[1]: 1}
        y = data[target].map(mapping).astype(int)
        encoding_rows = [[str(unique_y[0]), 0], [str(unique_y[1]), 1]]

    X = data[features].astype(float)

    # Case Processing Summary    
    case_summary = pd.DataFrame([
        ["Selected Cases", "Included in Analysis", included_n, included_n / raw_n * 100],
        ["Selected Cases", "Missing Cases", excluded_n, excluded_n / raw_n * 100],
        ["Selected Cases", "Total", raw_n, 100.0],
    ], columns=["Case Type", "Status", "N", "Percent"])
    case_summary["Percent"] = case_summary["Percent"].round(1)

    # Dependent Variable Encoding
    encoding_tbl = pd.DataFrame(encoding_rows, columns=["Original Value", "Internal Value"])

    # Null model and full model
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

    # Omnibus Tests of Model Coefficients
    omnibus_tbl = pd.DataFrame([
        ["Step 1", "Step", chi2_model, df_model, format_p_value(p_model)],
        ["Step 1", "Block", chi2_model, df_model, format_p_value(p_model)],
        ["Step 1", "Model", chi2_model, df_model, format_p_value(p_model)],
    ], columns=["Step", "", "Chi-square", "df", "Sig."])

    # Model Summary
    n = int(len(y))
    cox_snell = 1 - np.exp((2 / n) * (ll_null - ll_model))
    nagelkerke_den = 1 - np.exp((2 / n) * ll_null)
    nagelkerke = cox_snell / nagelkerke_den if nagelkerke_den != 0 else np.nan

    model_summary = pd.DataFrame([[
        1,
        minus2ll,
        cox_snell,
        nagelkerke
    ]], columns=["Step", "-2 Log likelihood", "Cox & Snell R Square", "Nagelkerke R Square"])

    # Variables in the Equation
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

    # Classification Table
    probs = model.predict(X_sm)
    pred = (probs >= cutoff).astype(int)

    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())

    row0_total = tn + fp
    row1_total = fn + tp
    total = row0_total + row1_total

    classification_tbl = pd.DataFrame([
        ["Step 1", "0", tn, fp, tn / row0_total * 100 if row0_total else np.nan],
        ["Step 1", "1", fn, tp, tp / row1_total * 100 if row1_total else np.nan],
        ["Step 1", "Overall Percentage", "", "", (tn + tp) / total * 100 if total else np.nan],
    ], columns=["Step", "Observed", "Predicted 0", "Predicted 1", "Percentage Correct"])

    classification_cutoff = pd.DataFrame([[
        f"The cut value is {cutoff:.2f}"
    ]], columns=["Classification cutoff"])

    # Hosmer-Lemeshow
    hl_tbl, hl_detail = hosmer_lemeshow_table(y, probs, g=10)

    return {
        "model": model,
        "y": y,
        "X": X,
        "prob": probs,
        "case_summary": compact_numeric_df(case_summary, 4),
        "encoding": encoding_tbl,
        "omnibus": compact_numeric_df(omnibus_tbl, 4),
        "model_summary": compact_numeric_df(model_summary, 4),
        "hosmer": compact_numeric_df(hl_tbl, 4),
        "hosmer_detail": compact_numeric_df(hl_detail, 4),
        "classification_cutoff": classification_cutoff,
        "classification": compact_numeric_df(classification_tbl, 4),
        "table": compact_numeric_df(variables_tbl, 4),
    }


# =========================================================
# ROC (manual, no sklearn)
# =========================================================
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

    auc_tbl = pd.DataFrame([["Area Under the Curve", auc_val]], columns=["Measure", "Value"])
    auc_tbl["Value"] = pd.to_numeric(auc_tbl["Value"], errors="coerce").round(4)
    auc_tbl = auc_tbl.apply(lambda col: col.map(clean_cell))

    cutoff_tbl = pd.DataFrame([[
        thr[best_idx], tpr[best_idx], specificity[best_idx], youden[best_idx]
    ]], columns=["Optimal cutoff", "Sensitivity", "Specificity", "Youden index"])
    cutoff_tbl = compact_numeric_df(cutoff_tbl, decimals=4)

    roc_tbl = pd.DataFrame({
        "Threshold": thr,
        "Sensitivity (TPR)": tpr,
        "Specificity": specificity,
        "1 - Specificity (FPR)": fpr,
        "Youden index": youden
    })
    roc_tbl = compact_numeric_df(roc_tbl, decimals=4)

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


# =========================================================
# Linear regression (formula) + diagnostics
# =========================================================
def fit_linear_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]):
    data = df[[y_col] + x_cols].dropna().copy()
    if len(data) < 3:
        raise ValueError("Not enough rows after dropping missing values.")

    def q(name: str) -> str:
        return f'Q("{name}")'

    formula = q(y_col) + " ~ " + " + ".join(q(c) for c in x_cols)
    model = smf.ols(formula=formula, data=data).fit()
    return model, data


# =========================================================
# Confidence intervals: mean & variance (parametric + bootstrap)
# =========================================================
def ci_mean(x: np.ndarray, alpha=0.05) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    tcrit = float(stats.t.ppf(1 - alpha/2, df=n-1))
    se = sd / math.sqrt(n)
    lo = mean - tcrit * se
    hi = mean + tcrit * se
    df = pd.DataFrame([["Mean", mean, lo, hi]], columns=["Parameter", "Estimate", "CI 2.5%", "CI 97.5%"])
    return compact_numeric_df(df, decimals=4)

def ci_variance(x: np.ndarray, alpha=0.05) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    n = len(x)
    s2 = float(np.var(x, ddof=1))
    chi2_lo = float(stats.chi2.ppf(alpha/2, df=n-1))
    chi2_hi = float(stats.chi2.ppf(1 - alpha/2, df=n-1))
    lo = (n-1)*s2 / chi2_hi
    hi = (n-1)*s2 / chi2_lo
    df = pd.DataFrame([["Variance", s2, lo, hi]], columns=["Parameter", "Estimate", "CI 2.5%", "CI 97.5%"])
    return compact_numeric_df(df, decimals=4)

def bootstrap_ci(x: np.ndarray, func, n_boot=5000, alpha=0.05, seed=123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    stats_ = []
    for _ in range(int(n_boot)):
        samp = rng.choice(x, size=n, replace=True)
        stats_.append(func(samp))
    stats_ = np.sort(np.array(stats_, dtype=float))
    lo = float(np.quantile(stats_, alpha/2))
    hi = float(np.quantile(stats_, 1 - alpha/2))
    return lo, hi

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
        if c in exclude:
            continue
        x = df[c].dropna()
        if x.empty:
            continue
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
        raise ValueError("Not enough paired observations after removing missing/non-numeric values.")
    return d

def long_numeric_group_data(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    d = df[[value_col, group_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d[group_col] = d[group_col].astype(str)
    d = d.dropna()
    if d.empty:
        raise ValueError("No valid numeric observations after removing missing/non-numeric values.")
    return d

def normality_by_group_table(groups: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, arr in groups.items():
        x = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().astype(float).values
        n = len(x)
        if 3 <= n <= 5000:
            stat, pval = stats.shapiro(x)
        else:
            stat, pval = np.nan, np.nan
        rows.append([name, n, stat, format_p_value(pval), "Yes" if isinstance(pval, float) and not np.isnan(pval) and pval >= 0.05 else "No"])
    return compact_numeric_df(pd.DataFrame(rows, columns=["Group", "N", "Shapiro-Wilk", "Sig.", "Normal assumption"]), 4)

def descriptives_for_groups(groups: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, arr in groups.items():
        x = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().astype(float).values
        rows.append([name, len(x), np.mean(x) if len(x) else np.nan, np.std(x, ddof=1) if len(x)>1 else np.nan, np.median(x) if len(x) else np.nan, np.min(x) if len(x) else np.nan, np.max(x) if len(x) else np.nan])
    return compact_numeric_df(pd.DataFrame(rows, columns=["Group", "N", "Mean", "Std. Deviation", "Median", "Minimum", "Maximum"]), 4)

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

def assumption_recommendation(normal_ok: bool, equal_var_ok=None, parametric_name: str = "parametric test", nonparametric_name: str = "nonparametric alternative") -> str:
    if not normal_ok:
        return f"Normality assumption is not met. Prefer {nonparametric_name}."
    if equal_var_ok is False:
        return f"Normality is acceptable but equal variances are not met. Prefer Welch/robust version of {parametric_name}."
    return f"Main assumptions are acceptable. {parametric_name} can be used."

def normality_overall_ok(groups: Dict[str, np.ndarray]) -> bool:
    ok = True
    for arr in groups.values():
        x = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().astype(float).values
        if 3 <= len(x) <= 5000:
            _, pval = stats.shapiro(x)
            if float(pval) < 0.05:
                ok = False
    return ok

def recommendation_table(recommendation: str) -> pd.DataFrame:
    return pd.DataFrame([[recommendation]], columns=["Recommendation"])

def nonparam_result_table(test_name: str, statistic: float, pval: float) -> pd.DataFrame:
    out = pd.DataFrame([[test_name, statistic, format_p_value(pval), "Yes" if pval < 0.05 else "No", conclusion_text(pval)]], columns=["Test", "Statistic", "Sig.", "Significant (p<0.05)", "Conclusion"])
    return compact_numeric_df(out, 4)

def ttest_result_table(test_name: str, statistic: float, dfree, pval: float, mean_diff: float = np.nan, ci=None) -> pd.DataFrame:
    if ci is None:
        ci = (np.nan, np.nan)
    out = pd.DataFrame([[test_name, statistic, dfree, format_p_value(pval), mean_diff, ci[0], ci[1], "Yes" if pval < 0.05 else "No", conclusion_text(pval)]], columns=["Test", "t", "df", "Sig. (2-tailed)", "Mean Difference", "CI 2.5%", "CI 97.5%", "Significant (p<0.05)", "Conclusion"])
    return compact_numeric_df(out, 4)

def chi_square_expected_assumption_table(expected: np.ndarray) -> pd.DataFrame:
    expected = np.asarray(expected, dtype=float)
    total_cells = expected.size
    cells_lt5 = int((expected < 5).sum())
    min_expected = float(np.min(expected)) if total_cells else np.nan
    pct_lt5 = cells_lt5 / total_cells * 100 if total_cells else np.nan
    ok_strict = bool(cells_lt5 == 0)
    ok_spss = bool(min_expected >= 1 and pct_lt5 <= 20)
    return compact_numeric_df(pd.DataFrame([[total_cells, cells_lt5, pct_lt5, min_expected, "Yes" if ok_strict else "No", "Yes" if ok_spss else "No"]], columns=["Cells", "Expected < 5", "% Expected < 5", "Minimum Expected Count", "All expected >= 5", "Common rule acceptable"]), 4)

def chi_square_guidance(obs: np.ndarray, expected: np.ndarray) -> str:
    expected = np.asarray(expected, dtype=float)
    if (expected >= 5).all():
        return "Expected count condition is satisfied. Pearson Chi-square is appropriate."
    if obs.shape == (2, 2):
        return "Some expected counts are below 5. Prefer Fisher's Exact Test for a 2x2 table."
    return "Some expected counts are below 5. Consider combining sparse categories or using an exact/Monte Carlo test instead of relying only on Pearson Chi-square."


def chi_square_alternative_test_table(obs: np.ndarray, expected: np.ndarray, n_resamples: int = 10000, seed: int = 123) -> pd.DataFrame:
    obs = np.asarray(obs, dtype=int)
    expected = np.asarray(expected, dtype=float)

    if obs.shape == (2, 2):
        oddsratio, pval = stats.fisher_exact(obs, alternative="two-sided")
        out = pd.DataFrame([[
            "Fisher's Exact Test",
            oddsratio,
            format_p_value(pval),
            "Yes" if pval < 0.05 else "No",
            conclusion_text(pval)
        ]], columns=["Alternative test", "Statistic / Odds Ratio", "Sig.", "Significant (p<0.05)", "Conclusion"])
        return compact_numeric_df(out, 4)

    if not hasattr(stats, "random_table"):
        out = pd.DataFrame([[
            "Exact / Monte Carlo test",
            "",
            "",
            "",
            "For tables larger than 2x2, use an exact or Monte Carlo test, or combine sparse categories."
        ]], columns=["Alternative test", "Statistic / Odds Ratio", "Sig.", "Significant (p<0.05)", "Conclusion"])
        return out

    row_sums = obs.sum(axis=1)
    col_sums = obs.sum(axis=0)

    if obs.sum() <= 0:
        raise ValueError("Total count must be > 0.")

    obs_stat = float(np.nansum((obs - expected) ** 2 / expected))
    rng = np.random.default_rng(seed)
    greater_equal = 0

    for _ in range(int(n_resamples)):
        try:
            sim = stats.random_table.rvs(row_sums, col_sums, random_state=rng)
        except TypeError:
            sim = stats.random_table(row_sums, col_sums).rvs(random_state=rng)

        sim_expected = np.outer(sim.sum(axis=1), sim.sum(axis=0)) / sim.sum()
        sim_stat = float(np.nansum((sim - sim_expected) ** 2 / sim_expected))
        if sim_stat >= obs_stat - 1e-12:
            greater_equal += 1

    pval = (greater_equal + 1) / (int(n_resamples) + 1)
    out = pd.DataFrame([[
        f"Monte Carlo Chi-square ({int(n_resamples)} samples)",
        obs_stat,
        format_p_value(pval),
        "Yes" if pval < 0.05 else "No",
        conclusion_text(pval)
    ]], columns=["Alternative test", "Statistic / Odds Ratio", "Sig.", "Significant (p<0.05)", "Conclusion"])
    return compact_numeric_df(out, 4)


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
    lev_tbl = compact_numeric_df(pd.DataFrame([["Levene's Test for Equality of Variances", lev_stat, format_p_value(lev_p), "Equal variances assumed" if lev_p >= 0.05 else "Equal variances not assumed"]], columns=["Test", "F", "Sig.", "Decision"]), 4)
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
    t_tbl = compact_numeric_df(pd.DataFrame(rows, columns=["Assumption", "t", "df", "Sig. (2-tailed)", "Mean Difference", "CI 2.5%", "CI 97.5%", "Significant (p<0.05)"]), 4)
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
            a[col] = pd.to_numeric(a[col], errors="coerce").round(4)
    return a.apply(lambda col: col.map(clean_cell))


def pvalue_float_from_text(x):
    try:
        if isinstance(x, str) and x.strip().startswith("<"):
            return 0.0005
        return float(x)
    except Exception:
        return np.nan

def effect_size_table(rows):
    return compact_numeric_df(pd.DataFrame(rows, columns=["Effect size", "Estimate", "Interpretation"]), 4)

def cohen_d_one_sample(x, mu=0.0):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().astype(float).values
    if len(x) < 2:
        return np.nan
    sd = np.std(x, ddof=1)
    return np.nan if sd == 0 else float((np.mean(x) - mu) / sd)

def cohen_d_independent(x1, x2):
    x1 = pd.to_numeric(pd.Series(x1), errors="coerce").dropna().astype(float).values
    x2 = pd.to_numeric(pd.Series(x2), errors="coerce").dropna().astype(float).values
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    pooled = math.sqrt(((len(x1)-1)*np.var(x1, ddof=1) + (len(x2)-1)*np.var(x2, ddof=1)) / (len(x1)+len(x2)-2))
    return np.nan if pooled == 0 else float((np.mean(x1)-np.mean(x2))/pooled)

def cohen_d_paired(diff):
    diff = pd.to_numeric(pd.Series(diff), errors="coerce").dropna().astype(float).values
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    return np.nan if sd == 0 else float(np.mean(diff)/sd)

def cohen_interpretation(d):
    try:
        a = abs(float(d))
    except Exception:
        return ""
    if np.isnan(a):
        return ""
    if a < 0.2:
        return "Very small"
    if a < 0.5:
        return "Small"
    if a < 0.8:
        return "Medium"
    return "Large"

def cramers_v_from_table(obs):
    obs = np.asarray(obs, dtype=float)
    if obs.ndim != 2 or obs.sum() <= 0:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(obs, correction=False)
    n = obs.sum()
    k = min(obs.shape[0]-1, obs.shape[1]-1)
    return np.nan if k <= 0 else float(math.sqrt(chi2/(n*k)))

def chi_square_effect_table(obs):
    v = cramers_v_from_table(obs)
    return effect_size_table([["Cramer's V", v, "Association strength for contingency tables"]])

def eta_squared_from_anova_table(a):
    df = a.copy()
    if "Sum Sq" not in df.columns or "Source" not in df.columns:
        return pd.DataFrame()
    ss = pd.to_numeric(df["Sum Sq"], errors="coerce")
    total = ss.sum(skipna=True)
    rows = []
    for _, r in df.iterrows():
        src = str(r.get("Source", ""))
        if src.lower() in {"residual", "error"}:
            continue
        val = pd.to_numeric(pd.Series([r.get("Sum Sq")]), errors="coerce").iloc[0]
        eta = val / total if total and not np.isnan(val) else np.nan
        rows.append([src, eta])
    return compact_numeric_df(pd.DataFrame(rows, columns=["Source", "Eta squared (η²)"]), 4)

def tukey_posthoc_table(d, value_col, group_col, alpha=0.05):
    dd = d[[value_col, group_col]].dropna().copy()
    dd[value_col] = pd.to_numeric(dd[value_col], errors="coerce")
    dd = dd.dropna()
    if dd[group_col].nunique() < 2:
        return pd.DataFrame()
    res = pairwise_tukeyhsd(endog=dd[value_col].astype(float), groups=dd[group_col].astype(str), alpha=alpha)
    tbl = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return compact_numeric_df(tbl, 4)

def dunn_posthoc_table(d, value_col, group_col, alpha=0.05):
    dd = d[[value_col, group_col]].dropna().copy()
    dd[value_col] = pd.to_numeric(dd[value_col], errors="coerce")
    dd[group_col] = dd[group_col].astype(str)
    dd = dd.dropna()
    groups = sorted(dd[group_col].unique())
    if len(groups) < 2:
        return pd.DataFrame()
    ranks = stats.rankdata(dd[value_col].values)
    dd = dd.assign(_rank=ranks)
    n = len(dd)
    tie_counts = pd.Series(dd[value_col]).value_counts().values
    tie_corr = 1 - np.sum(tie_counts**3 - tie_counts) / (n**3 - n) if n > 1 else 1
    rows = []
    m = len(groups) * (len(groups)-1) / 2
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1, g2 = groups[i], groups[j]
            r1 = dd.loc[dd[group_col] == g1, "_rank"]
            r2 = dd.loc[dd[group_col] == g2, "_rank"]
            se = math.sqrt((n*(n+1)/12) * (1/len(r1) + 1/len(r2)) * tie_corr)
            z = (r1.mean() - r2.mean()) / se if se > 0 else np.nan
            p_raw = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
            p_adj = min(1.0, p_raw * m) if not np.isnan(p_raw) else np.nan
            rows.append([g1, g2, z, format_p_value(p_raw), format_p_value(p_adj), "Yes" if p_adj < alpha else "No"])
    return compact_numeric_df(pd.DataFrame(rows, columns=["Group 1", "Group 2", "Z", "Sig.", "Bonferroni Sig.", "Significant"]), 4)

def pairwise_wilcoxon_related(wide, alpha=0.05):
    cols = list(wide.columns)
    rows = []
    m = len(cols) * (len(cols)-1) / 2
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a = wide[cols[i]].values
            b = wide[cols[j]].values
            stat, pval = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            p_adj = min(1.0, float(pval) * m)
            rows.append([str(cols[i]), str(cols[j]), float(stat), format_p_value(float(pval)), format_p_value(p_adj), "Yes" if p_adj < alpha else "No"])
    return compact_numeric_df(pd.DataFrame(rows, columns=["Condition 1", "Condition 2", "Statistic", "Sig.", "Bonferroni Sig.", "Significant"]), 4)

def proportion_ci_methods(x, n, conf_level=0.95):
    from statsmodels.stats.proportion import proportion_confint
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
    return compact_numeric_df(out, 4), wald_ok

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
    ], columns=["Measure", "Percent"]), 4)
    table = compact_numeric_df(pd.DataFrame([
        ["Test Positive", tp, fp, tp+fp],
        ["Test Negative", fn, tn, fn+tn],
        ["Total", disease, no_disease, population],
    ], columns=["Result", "Disease Present", "Disease Absent", "Total"]), 4)
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
    ], columns=["Calculation", "Value"]), 4)
    return summary, table, calc

# =========================================================
# Navigation state
# =========================================================
if "section" not in st.session_state:
    st.session_state.section = "Home"
if "sub" not in st.session_state:
    st.session_state.sub = "Overview"

def set_nav(section: str, sub: str):
    st.session_state.section = section
    st.session_state.sub = sub


# =========================================================
# Sidebar navigation
# =========================================================
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
        if st.button("Mean & Variance CI", key="ci_1", use_container_width=True):
            set_nav("Confidence Intervals", "Mean & Variance")
        if st.button("Proportion CI", key="ci_prop", use_container_width=True):
            set_nav("Confidence Intervals", "Proportion")

    with st.expander("Diagnostic Probability", expanded=(st.session_state.section == "Diagnostic Probability")):
        if st.button("Predictive Values", key="diag_prob", use_container_width=True):
            set_nav("Diagnostic Probability", "Predictive Values")


# =========================================================
# Pages
# =========================================================
section = st.session_state.section
sub = st.session_state.sub

# -----------------------------
# HOME
# -----------------------------
if section == "Home":
    st.markdown("## Overview")
    st.info(
        "Select a module from the sidebar. Each analysis page includes its own template download and file upload panel."
    )

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

    # Template used everywhere in logistic pages
    logistic_template = pd.DataFrame({
        "target_binary": [0, 1, 0, 1],
        "CRP": [10.2, 35.1, 8.7, 22.4],
        "WBC": [7.1, 12.5, 6.8, 10.3],
    })

    if sub == "Data":
        st.markdown("## Logistic Regression — Data")
        data_input_panel(
            template_df=logistic_template,
            template_name="logistic_template",
            store_key=LOGISTIC_KEY,
            store_name_key="df_logistic_name",
            help_text="Upload CSV/XLSX • Keep column names consistent with the template."
        )

    elif sub == "EDA":
        st.markdown("## Logistic Regression — EDA")

        data_input_panel(
            template_df=logistic_template,
            template_name="logistic_template",
            store_key=LOGISTIC_KEY,
            store_name_key="df_logistic_name",
            help_text="Upload here anytime (you do not need to return to the Data page)."
        )

        df = require_df(LOGISTIC_KEY)

        st.markdown("### Missing values")
        miss = df.isna().sum().reset_index()
        miss.columns = ["Variable", "Missing"]
        show_table(miss, "Missing Values")
        download_table_block(miss, "logistic_missing_values", "Missing Values")

        st.markdown("### Descriptive statistics (numeric)")
        desc = df.describe(include=[np.number]).T.reset_index().rename(columns={"index": "Variable"})
        desc = compact_numeric_df(desc, decimals=4)
        show_table(desc, "Descriptives")
        download_table_block(desc, "logistic_descriptives", "Descriptives")

    elif sub == "Modeling":
        st.markdown("## Logistic Regression — Modeling (OR, ROC)")

        data_input_panel(
            template_df=logistic_template,
            template_name="logistic_template",
            store_key=LOGISTIC_KEY,
            store_name_key="df_logistic_name",
            help_text="Upload here anytime."
        )

        df = require_df(LOGISTIC_KEY)
        cols = list(df.columns)

        target = st.selectbox("Target column (binary 0/1)", options=cols)
        features = st.multiselect("Predictors (numeric)", options=[c for c in cols if c != target])
        cutoff = st.slider("Classification cutoff", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

        if st.button("Run Logistic Regression", type="primary", use_container_width=True):
            try:
                res = run_logistic_statsmodels(df, target, features, cutoff=cutoff)

                show_table(res["case_summary"], "Case Processing Summary")
                download_table_block(res["case_summary"], "logistic_case_processing", "Case Processing Summary")

                show_table(res["encoding"], "Dependent Variable Encoding")
                download_table_block(res["encoding"], "logistic_dependent_variable_encoding", "Dependent Variable Encoding")

                show_table(res["omnibus"], "Omnibus Tests of Model Coefficients")
                download_table_block(res["omnibus"], "logistic_omnibus_tests", "Omnibus Tests")

                show_table(res["model_summary"], "Model Summary")
                download_table_block(res["model_summary"], "logistic_model_summary", "Model Summary")

                show_table(res["hosmer"], "Hosmer and Lemeshow Test")
                download_table_block(res["hosmer"], "logistic_hosmer_lemeshow", "Hosmer and Lemeshow Test")

                show_table(res["hosmer_detail"], "Contingency Table for Hosmer and Lemeshow Test")
                download_table_block(res["hosmer_detail"], "logistic_hosmer_detail", "Hosmer and Lemeshow Detail")

                show_table(res["classification_cutoff"], "Classification Table Note")
                show_table(res["classification"], "Classification Table")
                download_table_block(res["classification"], "logistic_classification_table", "Classification Table")

                show_table(res["table"], "Variables in the Equation")
                download_table_block(res["table"], "logistic_variables_in_equation", "Variables in the Equation")

                auc_tbl, roc_tbl, cutoff_tbl, fig = roc_outputs(res["model"], res["y"], res["X"])

                show_table(auc_tbl, "ROC — Area Under the Curve")
                download_table_block(auc_tbl, "logistic_auc", "AUC")

                show_table(cutoff_tbl, "ROC — Optimal Cutoff by Youden Index")
                download_table_block(cutoff_tbl, "logistic_roc_optimal_cutoff", "Optimal Cutoff")

                show_table(roc_tbl.head(50), "ROC Coordinates (first 50 rows)")
                download_table_block(roc_tbl.head(60), "logistic_roc_coordinates", "ROC Coordinates")

                st.pyplot(fig)
                download_figure_block(fig, "logistic_roc_curve")
                plt.close(fig)

            except Exception as e:
                st.error(f"Modeling failed: {e}")

    elif sub == "Export":
        st.markdown("## Logistic Regression — Export")
        st.info("Each table/figure includes download buttons (Excel/PNG).")


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
        data_input_panel(
            template_df=linear_template,
            template_name="linear_template",
            store_key=LINEAR_KEY,
            store_name_key="df_linear_name",
            help_text="Upload CSV/XLSX • Keep column names consistent with the template."
        )

    elif sub == "Diagnostics":
        st.markdown("## Linear Regression — Assumptions & Diagnostics")

        data_input_panel(
            template_df=linear_template,
            template_name="linear_template",
            store_key=LINEAR_KEY,
            store_name_key="df_linear_name",
            help_text="Upload here anytime."
        )

        df = require_df(LINEAR_KEY)
        cols = list(df.columns)

        y_col = st.selectbox("Dependent variable (Y)", options=cols)
        x_cols = st.multiselect("Independent variables (X)", options=[c for c in cols if c != y_col])

        if st.button("Run diagnostics", type="primary", use_container_width=True):
            try:
                model, data_used = fit_linear_ols(df, y_col, x_cols)
                resid = np.asarray(model.resid, dtype=float)
                fitted = np.asarray(model.fittedvalues, dtype=float)

                # Normality (Shapiro)
                sh_stat, sh_p = stats.shapiro(resid) if 3 <= len(resid) <= 5000 else (np.nan, np.nan)
                norm_tbl = pd.DataFrame([["Shapiro-Wilk", sh_stat, format_p_value(sh_p)]],
                                        columns=["Test", "Statistic", "Sig."])
                norm_tbl = compact_numeric_df(norm_tbl, decimals=4)
                show_table(norm_tbl, "Tests of Normality (Residuals)")
                download_table_block(norm_tbl, "linear_residual_normality", "Normality")

                # Homoscedasticity (Breusch–Pagan)
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(model.resid, model.model.exog)
                bp_tbl = pd.DataFrame([["Breusch-Pagan", bp_lm, format_p_value(bp_p)]],
                                      columns=["Test", "LM", "Sig."])
                bp_tbl = compact_numeric_df(bp_tbl, decimals=4)
                show_table(bp_tbl, "Homoscedasticity")
                download_table_block(bp_tbl, "linear_homoscedasticity", "Homoscedasticity")

                # Multicollinearity (VIF)
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                X_exog = model.model.exog
                names = model.model.exog_names
                vif_rows = []
                for i in range(len(names)):
                    if names[i] == "Intercept":
                        continue
                    vif_rows.append([clean_term_name(names[i]), variance_inflation_factor(X_exog, i)])
                vif_df = pd.DataFrame(vif_rows, columns=["Predictor", "VIF"])
                vif_df = compact_numeric_df(vif_df, decimals=4)
                show_table(vif_df, "Collinearity Statistics (VIF)")
                download_table_block(vif_df, "linear_vif", "VIF")

                # Plots
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.scatter(fitted, resid)
                ax1.axhline(0, linestyle="--")
                ax1.set_xlabel("Fitted values")
                ax1.set_ylabel("Residuals")
                ax1.set_title("Residuals vs Fitted")
                st.pyplot(fig1)
                download_figure_block(fig1, "linear_residuals_vs_fitted")
                plt.close(fig1)

            except Exception as e:
                st.error(f"Diagnostics failed: {e}")

    elif sub == "Modeling":
        st.markdown("## Linear Regression — Modeling (Coefficients, ANOVA)")

        data_input_panel(
            template_df=linear_template,
            template_name="linear_template",
            store_key=LINEAR_KEY,
            store_name_key="df_linear_name",
            help_text="Upload here anytime."
        )

        df = require_df(LINEAR_KEY)
        cols = list(df.columns)

        y_col = st.selectbox("Dependent variable (Y)", options=cols)
        x_cols = st.multiselect("Independent variables (X)", options=[c for c in cols if c != y_col])

        if st.button("Run linear regression", type="primary", use_container_width=True):
            try:
                model, data_used = fit_linear_ols(df, y_col, x_cols)

                # ANOVA
                a = anova_lm(model, typ=1).reset_index().rename(columns={"index": "Source"})
                a["Source"] = a["Source"].apply(clean_term_name)
                a = a.rename(columns={"df": "df", "sum_sq": "Sum Sq", "mean_sq": "Mean Sq", "F": "F", "PR(>F)": "Sig."})
                a["Sig."] = a["Sig."].apply(format_p_value)
                for col in ["Sum Sq", "Mean Sq", "F"]:
                    if col in a.columns:
                        a[col] = pd.to_numeric(a[col], errors="coerce").round(4)
                a = a.apply(lambda col: col.map(clean_cell))

                show_table(a, "ANOVA")
                download_table_block(a, "linear_anova", "ANOVA")

                # Coefficients
                b = model.summary2().tables[1].reset_index().rename(columns={"index": "Term"})
                b["Term"] = b["Term"].apply(clean_term_name)

                b = b.rename(columns={
                    "Coef.": "B",
                    "Std.Err.": "S.E.",
                    "t": "t",
                    "P>|t|": "Sig.",
                    "[0.025": "CI 2.5%",
                    "0.975]": "CI 97.5%"
                })
                b["Sig."] = b["Sig."].apply(format_p_value)

                def sig_yesno(s: str) -> str:
                    if s == "< 0.001":
                        return "Yes"
                    if not s:
                        return "No"
                    try:
                        return "Yes" if float(s) < 0.05 else "No"
                    except Exception:
                        return "No"

                b["Significant (p<0.05)"] = b["Sig."].apply(sig_yesno)

                # Move Significant to last
                sig_col = b.pop("Significant (p<0.05)")
                b["Significant (p<0.05)"] = sig_col

                for col in ["B", "S.E.", "t", "CI 2.5%", "CI 97.5%"]:
                    if col in b.columns:
                        b[col] = pd.to_numeric(b[col], errors="coerce").round(4)
                b = b.apply(lambda col: col.map(clean_cell))

                show_table(b, "Coefficients")
                download_table_block(b, "linear_coefficients", "Coefficients")

            except Exception as e:
                st.error(f"Modeling failed: {e}")


# -----------------------------
# CATEGORICAL TESTS
# -----------------------------
elif section == "Categorical Tests":

    if sub == "Chi-square r×c":
        st.markdown("## Categorical Tests — Contingency Table (r×c) / Chi-square")

        counts_df, observed_df = rc_contingency_ui(key="chisq", default_r=2, default_c=2)
        show_table(counts_df, "Observed Frequencies (with Totals)")

        if st.button("Run Chi-square", type="primary", use_container_width=True):
            try:
                obs = get_observed_matrix(observed_df)

                chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)

                expected_assumption = chi_square_expected_assumption_table(expected)
                show_table(expected_assumption, "Expected Count Assumption")
                guidance = chi_square_guidance(obs, expected)
                show_table(recommendation_table(guidance), "Recommendation")
                expected_ok = bool((np.asarray(expected, dtype=float) >= 5).all())
                if not expected_ok:
                    st.warning(guidance)

                chi_tbl = pd.DataFrame([[
                    "Pearson Chi-Square", chi2, dof, format_p_value(p),
                    "Yes" if p < 0.05 else "No", conclusion_text(p)
                ]], columns=["Test", "Value", "df", "Asymp. Sig. (2-sided)", "Significant (p<0.05)", "Conclusion"])
                chi_tbl["Value"] = pd.to_numeric(chi_tbl["Value"], errors="coerce").round(6)
                chi_tbl = chi_tbl.apply(lambda col: col.map(clean_cell))

                show_table(chi_tbl, "Chi-Square Tests")
                download_table_block(chi_tbl, "chisq_tests", "Chi-Square Tests")

                cv_tbl = chi_square_effect_table(obs)
                show_table(cv_tbl, "Effect Size")
                download_table_block(cv_tbl, "chisq_effect_size", "Effect Size")

                if not expected_ok:
                    alt_tbl = chi_square_alternative_test_table(obs, expected)
                    show_table(alt_tbl, "Alternative Test")
                    download_table_block(alt_tbl, "chisq_alternative_test", "Alternative Test")

                # Expected table (with totals)
                # Use current group labels from editor state
                group_labels = st.session_state.get("ct_chisq", pd.DataFrame()).get("Group", pd.Series([""]*obs.shape[0])).tolist()
                exp_df = pd.DataFrame(expected, columns=observed_df.columns)
                exp_df.insert(0, "Group", group_labels[:exp_df.shape[0]])
                exp_df["Total"] = exp_df[observed_df.columns].sum(axis=1)

                total_row = {"Group": "Total"}
                for c in observed_df.columns:
                    total_row[c] = float(exp_df[c].sum())
                total_row["Total"] = float(exp_df["Total"].sum())
                exp_df = pd.concat([exp_df, pd.DataFrame([total_row])], ignore_index=True)

                for c in observed_df.columns.tolist() + ["Total"]:
                    exp_df[c] = pd.to_numeric(exp_df[c], errors="coerce").round(4)
                exp_df = exp_df.apply(lambda col: col.map(clean_cell))

                show_table(exp_df, "Expected Frequencies")
                download_table_block(exp_df, "chisq_expected", "Expected Frequencies")

                # If 2×2, show OR/RR/VE + accuracy
                if obs.shape == (2, 2):
                    meas = two_by_two_measures(obs, alpha=0.05)
                    show_table(meas, "2×2 Measures (OR, RR, VE, Diagnostic Accuracy)")
                    download_table_block(meas, "chisq_2x2_measures", "2×2 Measures")

            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Fisher 2×2":
        st.markdown("## Categorical Tests — Fisher's Exact Test (2×2)")

        counts_df, observed_df = contingency_editor(
            key="fisher",
            default_rows=["Group 1", "Group 2"],
            default_cols=["Outcome +", "Outcome -"],
            default_counts=np.array([[10, 30], [20, 15]], dtype=int),
        )
        show_table(counts_df, "Observed Frequencies (with Totals)")

        if st.button("Run Fisher's Exact", type="primary", use_container_width=True):
            try:
                obs = require_2x2(observed_df)
                oddsratio, p = stats.fisher_exact(obs, alternative="two-sided")

                tbl = pd.DataFrame([[
                    "Fisher's Exact Test", oddsratio, format_p_value(p),
                    "Yes" if p < 0.05 else "No"
                ]], columns=["Test", "Odds Ratio", "Exact Sig. (2-sided)", "Significant (p<0.05)"])
                tbl["Odds Ratio"] = pd.to_numeric(tbl["Odds Ratio"], errors="coerce").round(4)
                tbl = tbl.apply(lambda col: col.map(clean_cell))

                show_table(tbl, "Fisher's Exact Test")
                download_table_block(tbl, "fisher_exact", "Fisher's Exact Test")

                meas = two_by_two_measures(obs, alpha=0.05)
                show_table(meas, "2×2 Measures (OR, RR, VE, Diagnostic Accuracy)")
                download_table_block(meas, "fisher_2x2_measures", "2×2 Measures")

            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Goodness-of-fit":
        st.markdown("## Categorical Tests — Goodness-of-fit (Chi-square)")

        template = pd.DataFrame({
            "Category": ["A", "B", "C"],
            "Observed": [30, 50, 20]
        })

        st.download_button(
            "Download Excel template",
            data=df_to_excel_bytes({"gof_template": template}),
            file_name="gof_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )

        up = st.file_uploader("Upload GOF template (XLSX/CSV)", type=["xlsx", "csv"], key="gof_upload")
        if up is not None:
            try:
                df = load_uploaded_file(up)
                st.dataframe(df, use_container_width=True)

                if "Observed" not in df.columns:
                    raise ValueError("Missing 'Observed' column.")

                obs_s = pd.to_numeric(df["Observed"], errors="coerce")
                if obs_s.isna().any():
                    raise ValueError("Observed counts must be numeric.")
                if (obs_s < 0).any():
                    raise ValueError("Observed counts must be non-negative.")

                obs = obs_s.astype(float).values
                if np.sum(obs) <= 0:
                    raise ValueError("Observed counts must have total > 0.")

                if "Category" in df.columns:
                    categories = df["Category"].astype(str).tolist()
                else:
                    categories = [f"Category {i+1}" for i in range(len(obs))]

                expected_method = st.radio(
                    "Expected distribution",
                    ["Equal proportions", "Custom proportions / weights"],
                    horizontal=True
                )

                expected_input = None
                if expected_method == "Custom proportions / weights":
                    default_weights = np.ones(len(obs), dtype=float)
                    custom_df = pd.DataFrame({
                        "Category": categories,
                        "Expected proportion / weight": default_weights
                    })
                    expected_input = st.data_editor(
                        custom_df,
                        key="gof_expected_weights_editor",
                        use_container_width=True,
                        num_rows="fixed",
                        column_config={
                            "Category": st.column_config.TextColumn("Category", disabled=True),
                            "Expected proportion / weight": st.column_config.NumberColumn(
                                "Expected proportion / weight",
                                min_value=0.0,
                                step=0.01,
                                format="%.6f"
                            )
                        }
                    )

                if st.button("Run Goodness-of-fit", type="primary", use_container_width=True):
                    if expected_method == "Equal proportions":
                        weights = np.ones(len(obs), dtype=float)
                    else:
                        weights = pd.to_numeric(expected_input["Expected proportion / weight"], errors="coerce").fillna(0).astype(float).values
                        if len(weights) != len(obs):
                            raise ValueError("Expected proportions must have the same number of rows as Observed.")
                        if np.any(weights < 0):
                            raise ValueError("Expected proportions must be non-negative.")
                        if np.sum(weights) <= 0:
                            raise ValueError("Expected proportions must have total > 0.")

                    exp = weights / np.sum(weights) * np.sum(obs)
                    stat, p = stats.chisquare(f_obs=obs, f_exp=exp)
                    gof_assumption = chi_square_expected_assumption_table(exp)
                    show_table(gof_assumption, "Expected Count Assumption")
                    if not (np.asarray(exp, dtype=float) >= 5).all():
                        st.warning("Some expected counts are below 5. Consider combining sparse categories or using an exact/Monte Carlo goodness-of-fit approach.")

                    expected_tbl = pd.DataFrame({
                        "Category": categories,
                        "Observed": obs,
                        "Expected": exp,
                        "Expected proportion": exp / np.sum(exp)
                    })
                    expected_tbl = compact_numeric_df(expected_tbl, decimals=4)

                    tbl = pd.DataFrame([[
                        "Chi-square", stat, len(obs)-1, format_p_value(p),
                        "Yes" if p < 0.05 else "No", conclusion_text(p)
                    ]], columns=["Test", "Value", "df", "Asymp. Sig. (2-sided)", "Significant (p<0.05)", "Conclusion"])
                    tbl["Value"] = pd.to_numeric(tbl["Value"], errors="coerce").round(6)
                    tbl = tbl.apply(lambda col: col.map(clean_cell))

                    show_table(expected_tbl, "Observed and Expected Frequencies")
                    download_table_block(expected_tbl, "gof_observed_expected", "Observed and Expected Frequencies")

                    show_table(tbl, "Chi-Square Tests")
                    download_table_block(tbl, "gof_chisq", "Goodness-of-fit")

            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Mantel–Haenszel":
        st.markdown("## Categorical Tests — Mantel–Haenszel (Stratified 2×2)")

        st.caption(
            "Upload long-format data with columns: Stratum, a, b, c, d (each row = one stratum 2×2 table). "
            "a=Exposed&Outcome+, b=Exposed&Outcome-, c=Unexposed&Outcome+, d=Unexposed&Outcome-."
        )

        template = pd.DataFrame({
            "Stratum": ["S1", "S2"],
            "a": [5, 8],
            "b": [10, 12],
            "c": [7, 6],
            "d": [20, 18],
        })

        st.download_button(
            "Download Excel template",
            data=df_to_excel_bytes({"mh_template": template}),
            file_name="mh_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )

        up = st.file_uploader("Upload MH template (XLSX/CSV)", type=["xlsx", "csv"], key="mh_upload")
        if up is not None:
            try:
                df = load_uploaded_file(up)
                st.dataframe(df, use_container_width=True)

                needed = ["Stratum", "a", "b", "c", "d"]
                for col in needed:
                    if col not in df.columns:
                        raise ValueError(f"Missing column: {col}")

                for col in ["a", "b", "c", "d"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(0).astype(int)
                    if (df[col] < 0).any():
                        raise ValueError("Counts must be non-negative integers.")

                if st.button("Run Mantel–Haenszel", type="primary", use_container_width=True):
                    tables = []
                    for _, r in df.iterrows():
                        tab = np.array([[r["a"], r["b"]], [r["c"], r["d"]]], dtype=int)
                        tables.append(tab)

                    stbl = StratifiedTable(tables)
                    mh_or = float(stbl.oddsratio_pooled)
                    mh_ci = stbl.oddsratio_pooled_confint()
                    mh_p = float(stbl.test_null_odds().pvalue)

                    out = pd.DataFrame([[
                        "Mantel-Haenszel Common Odds Ratio", mh_or, mh_ci[0], mh_ci[1], format_p_value(mh_p),
                        "Yes" if mh_p < 0.05 else "No"
                    ]], columns=["Test", "Common OR", "CI 2.5%", "CI 97.5%", "Sig.", "Significant (p<0.05)"])
                    out = compact_numeric_df(out, decimals=4)

                    show_table(out, "Mantel-Haenszel Test")
                    download_table_block(out, "mh_results", "Mantel–Haenszel")

            except Exception as e:
                st.error(f"Failed: {e}")




# -----------------------------
# QUANTITATIVE TESTS
# -----------------------------
elif section == "Quantitative Tests":
    def _show_template_downloads(kind: str):
        if kind == "ttest":
            c1, c2, c3 = st.columns(3)
            with c1:
                tpl = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [10.2, 11.1, 13.0, 12.4]})
                st.download_button("Template: one-sample / independent", data=df_to_excel_bytes({"ttest_independent": tpl}), file_name="ttest_independent_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c2:
                tpl = pd.DataFrame({"subject": [1, 2, 3, 4], "before": [10.0, 11.2, 9.8, 12.0], "after": [11.0, 12.1, 10.3, 12.9]})
                st.download_button("Template: paired", data=df_to_excel_bytes({"ttest_paired": tpl}), file_name="ttest_paired_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c3:
                tpl = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [10.2, 11.1, 13.0, 12.4], "before": [10.0, 11.2, 9.8, 12.0], "after": [11.0, 12.1, 10.3, 12.9]})
                st.download_button("Template: combined", data=df_to_excel_bytes({"ttest_combined": tpl}), file_name="ttest_combined_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        elif kind == "nonparam":
            c1, c2, c3 = st.columns(3)
            with c1:
                tpl = pd.DataFrame({"group": ["A", "A", "B", "B", "C", "C"], "value": [10.2, 11.1, 13.0, 12.4, 9.5, 9.9]})
                st.download_button("Template: independent groups", data=df_to_excel_bytes({"nonparam_independent": tpl}), file_name="nonparam_independent_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c2:
                tpl = pd.DataFrame({"subject": [1, 2, 3, 4], "before": [10.0, 11.2, 9.8, 12.0], "after": [11.0, 12.1, 10.3, 12.9]})
                st.download_button("Template: Wilcoxon paired", data=df_to_excel_bytes({"wilcoxon_paired": tpl}), file_name="wilcoxon_paired_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c3:
                tpl = pd.DataFrame({"subject": [1,1,1,2,2,2,3,3,3], "time": ["T1","T2","T3","T1","T2","T3","T1","T2","T3"], "value": [10,12,11,9,10,8,13,15,14]})
                st.download_button("Template: Friedman", data=df_to_excel_bytes({"friedman_long": tpl}), file_name="friedman_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        elif kind == "anova":
            c1, c2, c3 = st.columns(3)
            with c1:
                tpl = pd.DataFrame({"group": ["A","A","B","B","C","C"], "value": [10.2,11.1,13.0,12.4,9.5,9.9]})
                st.download_button("Template: one-way independent", data=df_to_excel_bytes({"anova_oneway": tpl}), file_name="anova_oneway_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c2:
                tpl = pd.DataFrame({"subject": [1,1,1,2,2,2,3,3,3], "time": ["T1","T2","T3","T1","T2","T3","T1","T2","T3"], "value": [10,12,11,9,10,8,13,15,14]})
                st.download_button("Template: repeated-measures", data=df_to_excel_bytes({"anova_repeated": tpl}), file_name="anova_repeated_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with c3:
                tpl = pd.DataFrame({"factor_a": ["A","A","B","B","A","A","B","B"], "factor_b": ["T1","T1","T1","T1","T2","T2","T2","T2"], "value": [10,11,13,14,12,13,15,16]})
                st.download_button("Template: two-way", data=df_to_excel_bytes({"anova_twoway": tpl}), file_name="anova_twoway_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    def _long_wide_repeated(df, subject_col, within_col, value_col):
        d = df[[subject_col, within_col, value_col]].copy()
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        d = d.dropna()
        wide = d.pivot_table(index=subject_col, columns=within_col, values=value_col, aggfunc="mean")
        wide_complete = wide.dropna()
        return d, wide_complete

    if sub == "t Tests":
        st.markdown("## Quantitative Tests — t Tests")
        _show_template_downloads("ttest")
        up = st.file_uploader("Upload t-test data (XLSX/CSV)", type=["xlsx", "csv"], key="ttest_upload")
        if up is not None:
            df = load_uploaded_file(up)
            st.dataframe(df.head(50), use_container_width=True)
            numeric_cols = numeric_candidate_cols(df)
            if not numeric_cols:
                st.error("No numeric test variables found.")
                st.stop()
            test_type = st.radio("Test type", ["One-sample t test", "Independent-samples t test", "Paired-samples t test"], horizontal=False)
            alpha = 1 - st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="tt_alpha_auto")
            try:
                if test_type == "One-sample t test":
                    value_col = selectbox_default("Test variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="tt_auto_one_value")
                    group_candidates = categorical_candidate_cols(df, exclude=[value_col])
                    group_default = default_group_col(df, exclude=[value_col])
                    group_choice = selectbox_default("Grouping variable (optional)", ["None"] + group_candidates, group_default if group_default else "None", key="tt_auto_one_group")
                    selected_level = None
                    levels = []
                    if group_choice != "None":
                        levels = sorted([str(x) for x in df[group_choice].dropna().astype(str).unique()])
                        selected_level = selectbox_default("Group to test", ["All groups separately"] + levels, "All groups separately", key="tt_auto_one_level")
                    mu = st.number_input("Test value", value=0.0, key="tt_auto_mu")
                    if st.button("Run one-sample t test", type="primary", use_container_width=True):
                        if group_choice == "None":
                            x = numeric_series_from_df(df, value_col).astype(float).values
                            if len(x) < 2:
                                raise ValueError("At least 2 valid observations are required.")
                            show_table(descriptives_for_groups({value_col: x}), "Descriptive Statistics")
                            show_table(normality_by_group_table({value_col: x}), "Tests of Normality")
                            show_table(recommendation_table(assumption_recommendation(normality_overall_ok({value_col: x}), None, "one-sample t test", "one-sample Wilcoxon signed-rank test")), "Recommendation")
                            show_table(one_sample_ttest_table(x, mu, alpha), "One-Sample Test")
                            d_val = cohen_d_one_sample(x, mu)
                            show_table(effect_size_table([["Cohen's d", d_val, cohen_interpretation(d_val)]]), "Effect Size")
                            if np.any((x - mu) != 0):
                                w_stat, w_p = stats.wilcoxon(x - mu, zero_method="wilcox", alternative="two-sided")
                                show_table(nonparam_result_table("Wilcoxon Signed-Rank Test", float(w_stat), float(w_p)), "Nonparametric Alternative")
                        else:
                            run_levels = levels if selected_level == "All groups separately" else [selected_level]
                            desc_groups, norm_groups, t_rows, w_rows = {}, {}, [], []
                            for lev in run_levels:
                                x = pd.to_numeric(df.loc[df[group_choice].astype(str) == lev, value_col], errors="coerce").dropna().astype(float).values
                                if len(x) < 2:
                                    continue
                                desc_groups[lev] = x
                                norm_groups[lev] = x
                                t_tbl = one_sample_ttest_table(x, mu, alpha)
                                t_tbl.insert(0, "Group", lev)
                                t_rows.append(t_tbl)
                                if np.any((x - mu) != 0):
                                    w_stat, w_p = stats.wilcoxon(x - mu, zero_method="wilcox", alternative="two-sided")
                                    w_tbl = nonparam_result_table("Wilcoxon Signed-Rank Test", float(w_stat), float(w_p))
                                    w_tbl.insert(0, "Group", lev)
                                    w_rows.append(w_tbl)
                            if not t_rows:
                                raise ValueError("No selected group has at least 2 valid numeric observations.")
                            show_table(descriptives_for_groups(desc_groups), "Descriptive Statistics")
                            show_table(normality_by_group_table(norm_groups), "Tests of Normality")
                            show_table(recommendation_table(assumption_recommendation(normality_overall_ok(norm_groups), None, "one-sample t test", "one-sample Wilcoxon signed-rank test")), "Recommendation")
                            show_table(compact_numeric_df(pd.concat(t_rows, ignore_index=True), 4), "One-Sample Test")
                            es_rows = []
                            for lev in run_levels:
                                x_es = pd.to_numeric(df.loc[df[group_choice].astype(str) == lev, value_col], errors="coerce").dropna().astype(float).values
                                if len(x_es) >= 2:
                                    d_val = cohen_d_one_sample(x_es, mu)
                                    es_rows.append([f"Cohen's d ({lev})", d_val, cohen_interpretation(d_val)])
                            if es_rows:
                                show_table(effect_size_table(es_rows), "Effect Size")
                            if w_rows:
                                show_table(compact_numeric_df(pd.concat(w_rows, ignore_index=True), 4), "Nonparametric Alternative")

                elif test_type == "Independent-samples t test":
                    value_col = selectbox_default("Test variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="tt_auto_ind_value")
                    group_candidates = categorical_candidate_cols(df, exclude=[value_col])
                    group_col = selectbox_default("Grouping variable", group_candidates, default_group_col(df, exclude=[value_col]), key="tt_auto_ind_group")
                    g1, g2 = level_selector(df, group_col, "tt_auto_ind")
                    if st.button("Run independent-samples t test", type="primary", use_container_width=True):
                        d0 = df[df[group_col].astype(str).isin([g1, g2])].copy()
                        d = long_numeric_group_data(d0, value_col, group_col)
                        groups, lev_tbl, t_tbl = independent_ttest_tables(d, value_col, group_col, alpha)
                        show_table(descriptives_for_groups(groups), "Group Statistics")
                        show_table(normality_by_group_table(groups), "Tests of Normality")
                        show_table(lev_tbl, "Test of Homogeneity of Variance")
                        lev_ok = True
                        try:
                            lev_ok = str(lev_tbl.loc[0, "Decision"]).startswith("Equal variances assumed")
                        except Exception:
                            lev_ok = None
                        show_table(recommendation_table(assumption_recommendation(normality_overall_ok(groups), lev_ok, "independent-samples t test", "Mann-Whitney U test")), "Recommendation")
                        show_table(t_tbl, "Independent Samples Test")
                        levels2 = list(groups.keys())
                        u_stat, u_p = stats.mannwhitneyu(groups[levels2[0]], groups[levels2[1]], alternative="two-sided")
                        show_table(nonparam_result_table("Mann-Whitney U Test", float(u_stat), float(u_p)), "Nonparametric Alternative")

                else:
                    before_default = first_existing(numeric_cols, ["before", "pre", "baseline", "time1", "t1"], numeric_cols[0])
                    before_col = selectbox_default("Variable 1", numeric_cols, before_default, key="tt_auto_pair_before")
                    after_options = [c for c in numeric_cols if c != before_col]
                    after_default = first_existing(after_options, ["after", "post", "followup", "time2", "t2"], after_options[0] if after_options else None)
                    after_col = selectbox_default("Variable 2", after_options, after_default, key="tt_auto_pair_after")
                    if st.button("Run paired-samples t test", type="primary", use_container_width=True):
                        d = paired_numeric_data(df, before_col, after_col)
                        groups, t_tbl = paired_ttest_table(d, before_col, after_col, alpha)
                        show_table(descriptives_for_groups({before_col: d[before_col].values, after_col: d[after_col].values}), "Paired Samples Statistics")
                        show_table(normality_by_group_table(groups), "Tests of Normality for Paired Difference")
                        show_table(recommendation_table(assumption_recommendation(normality_overall_ok(groups), None, "paired-samples t test", "Wilcoxon signed-rank test")), "Recommendation")
                        show_table(t_tbl, "Paired Samples Test")
                        w_stat, w_p = stats.wilcoxon(d[before_col].astype(float).values, d[after_col].astype(float).values, zero_method="wilcox", alternative="two-sided")
                        show_table(nonparam_result_table("Wilcoxon Signed-Rank Test", float(w_stat), float(w_p)), "Nonparametric Alternative")
            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Nonparametric Tests":
        st.markdown("## Quantitative Tests — Nonparametric Tests")
        _show_template_downloads("nonparam")
        up = st.file_uploader("Upload nonparametric-test data (XLSX/CSV)", type=["xlsx", "csv"], key="np_upload_auto")
        if up is not None:
            df = load_uploaded_file(up)
            st.dataframe(df.head(50), use_container_width=True)
            numeric_cols = numeric_candidate_cols(df)
            if not numeric_cols:
                st.error("No numeric test variables found.")
                st.stop()
            test_type = st.radio("Test", ["Mann-Whitney U (2 independent groups)", "Wilcoxon signed-rank (paired)", "Kruskal-Wallis (k independent groups)", "Friedman (k related samples)", "One-sample Wilcoxon"], horizontal=False)
            try:
                if test_type == "Mann-Whitney U (2 independent groups)":
                    value_col = selectbox_default("Test variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="np_auto_mw_value")
                    group_candidates = categorical_candidate_cols(df, exclude=[value_col])
                    group_col = selectbox_default("Grouping variable", group_candidates, default_group_col(df, exclude=[value_col]), key="np_auto_mw_group")
                    g1, g2 = level_selector(df, group_col, "np_auto_mw")
                    if st.button("Run Mann-Whitney U", type="primary", use_container_width=True):
                        d = df[df[group_col].astype(str).isin([g1, g2])].copy()
                        d = long_numeric_group_data(d, value_col, group_col)
                        groups = {str(k): v[value_col].astype(float).values for k, v in d.groupby(group_col)}
                        if len(groups) != 2:
                            raise ValueError("Mann-Whitney U requires exactly 2 selected groups.")
                        stat, pval = stats.mannwhitneyu(*groups.values(), alternative="two-sided")
                        show_table(descriptives_for_groups(groups), "Group Summary")
                        show_table(nonparam_result_table("Mann-Whitney U Test", float(stat), float(pval)), "Test Statistics")

                elif test_type == "Wilcoxon signed-rank (paired)":
                    before_default = first_existing(numeric_cols, ["before", "pre", "baseline", "time1", "t1"], numeric_cols[0])
                    before_col = selectbox_default("Variable 1", numeric_cols, before_default, key="np_auto_wx_before")
                    after_options = [c for c in numeric_cols if c != before_col]
                    after_default = first_existing(after_options, ["after", "post", "followup", "time2", "t2"], after_options[0] if after_options else None)
                    after_col = selectbox_default("Variable 2", after_options, after_default, key="np_auto_wx_after")
                    if st.button("Run Wilcoxon signed-rank", type="primary", use_container_width=True):
                        d = paired_numeric_data(df, before_col, after_col)
                        stat, pval = stats.wilcoxon(d[before_col].astype(float).values, d[after_col].astype(float).values, zero_method="wilcox", alternative="two-sided")
                        show_table(descriptives_for_groups({before_col: d[before_col].values, after_col: d[after_col].values}), "Paired Summary")
                        show_table(nonparam_result_table("Wilcoxon Signed-Rank Test", float(stat), float(pval)), "Test Statistics")

                elif test_type == "Kruskal-Wallis (k independent groups)":
                    value_col = selectbox_default("Test variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="np_auto_kw_value")
                    group_candidates = categorical_candidate_cols(df, exclude=[value_col])
                    group_col = selectbox_default("Grouping variable", group_candidates, default_group_col(df, exclude=[value_col]), key="np_auto_kw_group")
                    if st.button("Run Kruskal-Wallis", type="primary", use_container_width=True):
                        d = long_numeric_group_data(df, value_col, group_col)
                        groups = {str(k): v[value_col].astype(float).values for k, v in d.groupby(group_col)}
                        if len(groups) < 2:
                            raise ValueError("Kruskal-Wallis requires at least 2 groups.")
                        stat, pval = stats.kruskal(*groups.values())
                        show_table(descriptives_for_groups(groups), "Group Summary")
                        show_table(nonparam_result_table("Kruskal-Wallis Test", float(stat), float(pval)), "Test Statistics")

                elif test_type == "Friedman (k related samples)":
                    subject_col = selectbox_default("Subject ID", list(df.columns), default_subject_col(df), key="np_auto_fr_subject")
                    value_col = selectbox_default("Test variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="np_auto_fr_value")
                    within_candidates = categorical_candidate_cols(df, exclude=[subject_col, value_col])
                    within_col = selectbox_default("Within-subject factor", within_candidates, default_within_col(df, exclude=[subject_col, value_col]), key="np_auto_fr_within")
                    d, wide = _long_wide_repeated(df, subject_col, within_col, value_col)
                    n_levels = int(d[within_col].dropna().nunique()) if within_col in d.columns else 0
                    n_complete = int(wide.shape[0])
                    if n_levels < 3:
                        st.warning("Friedman requires at least 3 related conditions. With 2 related conditions, use Wilcoxon signed-rank instead.")
                    if n_complete < 2:
                        st.warning("Friedman requires at least 2 complete subjects. Use the Friedman template: each subject must have T1, T2 and T3 values.")
                    can_run_friedman = (n_levels >= 3 and n_complete >= 2)
                    if st.button("Run Friedman", type="primary", use_container_width=True, disabled=(not can_run_friedman)):
                        stat, pval = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
                        show_table(compact_numeric_df(wide.reset_index(), 4), "Complete Repeated-Measures Data")
                        show_table(nonparam_result_table("Friedman Test", float(stat), float(pval)), "Test Statistics")
                        show_table(pairwise_wilcoxon_related(wide, alpha=0.05), "Post-hoc Pairwise Wilcoxon-Bonferroni")

                else:
                    value_col = selectbox_default("Test variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="np_auto_one_value")
                    median0 = st.number_input("Test median", value=0.0, key="np_auto_median")
                    if st.button("Run one-sample Wilcoxon", type="primary", use_container_width=True):
                        x = numeric_series_from_df(df, value_col).astype(float).values
                        if len(x) < 2:
                            raise ValueError("At least 2 valid observations are required.")
                        stat, pval = stats.wilcoxon(x - median0, zero_method="wilcox", alternative="two-sided")
                        show_table(descriptives_for_groups({value_col: x}), "Descriptive Statistics")
                        show_table(nonparam_result_table("One-Sample Wilcoxon Signed-Rank Test", float(stat), float(pval)), "Test Statistics")
            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "ANOVA":
        st.markdown("## Quantitative Tests — ANOVA")
        _show_template_downloads("anova")
        up = st.file_uploader("Upload ANOVA data (XLSX/CSV)", type=["xlsx", "csv"], key="anova_upload_auto")
        if up is not None:
            df = load_uploaded_file(up)
            st.dataframe(df.head(50), use_container_width=True)
            numeric_cols = numeric_candidate_cols(df)
            if not numeric_cols:
                st.error("No numeric dependent variable found.")
                st.stop()
            anova_type = st.radio("ANOVA type", ["One-way ANOVA (independent)", "One-way repeated-measures ANOVA", "Two-way ANOVA"], horizontal=False)
            try:
                if anova_type == "One-way ANOVA (independent)":
                    value_col = selectbox_default("Dependent variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="anova_auto1_value")
                    factor_candidates = categorical_candidate_cols(df, exclude=[value_col])
                    factor_col = selectbox_default("Factor", factor_candidates, default_group_col(df, exclude=[value_col]), key="anova_auto1_factor")
                    if st.button("Run one-way ANOVA", type="primary", use_container_width=True):
                        d = long_numeric_group_data(df, value_col, factor_col)
                        groups = {str(k): v[value_col].astype(float).values for k, v in d.groupby(factor_col)}
                        if len(groups) < 2:
                            raise ValueError("One-way ANOVA requires at least 2 groups.")
                        show_table(descriptives_for_groups(groups), "Descriptives")
                        show_table(normality_by_group_table(groups), "Tests of Normality")
                        lev_stat, lev_p = stats.levene(*groups.values(), center="mean")
                        show_table(compact_numeric_df(pd.DataFrame([["Levene's Test", lev_stat, format_p_value(lev_p), "Yes" if lev_p >= 0.05 else "No"]], columns=["Test", "Statistic", "Sig.", "Equal variances assumption"]), 4), "Test of Homogeneity of Variances")
                        show_table(recommendation_table(assumption_recommendation(normality_overall_ok(groups), bool(lev_p >= 0.05), "one-way ANOVA", "Kruskal-Wallis test")), "Recommendation")
                        model = smf.ols(f'Q("{value_col}") ~ C(Q("{factor_col}"))', data=d).fit()
                        anova_tbl = anova_summary_table(model, typ=2)
                        show_table(anova_tbl, "ANOVA")
                        eta_tbl = eta_squared_from_anova_table(anova_tbl)
                        if not eta_tbl.empty:
                            show_table(eta_tbl, "Effect Size")
                        if len(groups) >= 3:
                            show_table(tukey_posthoc_table(d, value_col, factor_col, alpha=0.05), "Post-hoc Multiple Comparisons — Tukey HSD")
                        kw_stat, kw_p = stats.kruskal(*groups.values())
                        show_table(nonparam_result_table("Kruskal-Wallis Test", float(kw_stat), float(kw_p)), "Nonparametric Alternative")
                        if len(groups) >= 3:
                            show_table(dunn_posthoc_table(d, value_col, factor_col, alpha=0.05), "Nonparametric Post-hoc — Dunn-Bonferroni")

                elif anova_type == "One-way repeated-measures ANOVA":
                    subject_col = selectbox_default("Subject ID", list(df.columns), default_subject_col(df), key="rm_auto_subject")
                    value_col = selectbox_default("Dependent variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="rm_auto_value")
                    within_candidates = categorical_candidate_cols(df, exclude=[subject_col, value_col])
                    within_col = selectbox_default("Within-subject factor", within_candidates, default_within_col(df, exclude=[subject_col, value_col]), key="rm_auto_within")
                    if st.button("Run repeated-measures ANOVA", type="primary", use_container_width=True):
                        d, wide = _long_wide_repeated(df, subject_col, within_col, value_col)
                        if d[within_col].nunique() < 2:
                            raise ValueError("Repeated-measures ANOVA requires at least 2 levels of the within-subject factor.")
                        if wide.shape[0] < 2:
                            raise ValueError("Repeated-measures ANOVA requires at least 2 complete subjects.")
                        rm = sm.stats.AnovaRM(d, depvar=value_col, subject=subject_col, within=[within_col]).fit()
                        out = rm.anova_table.reset_index().rename(columns={"index": "Source", "F Value": "F", "Num DF": "df1", "Den DF": "df2", "Pr > F": "Sig."})
                        out["Sig."] = out["Sig."].apply(format_p_value)
                        show_table(compact_numeric_df(out, 4), "Tests of Within-Subjects Effects")
                        if wide.shape[1] >= 2:
                            show_table(pairwise_wilcoxon_related(wide, alpha=0.05), "Pairwise Related-Samples Comparisons")
                        if wide.shape[1] >= 3:
                            fr_stat, fr_p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
                            show_table(nonparam_result_table("Friedman Test", float(fr_stat), float(fr_p)), "Nonparametric Alternative")
                        elif wide.shape[1] == 2:
                            w_stat, w_p = stats.wilcoxon(wide.iloc[:, 0].values, wide.iloc[:, 1].values, zero_method="wilcox", alternative="two-sided")
                            show_table(nonparam_result_table("Wilcoxon Signed-Rank Test", float(w_stat), float(w_p)), "Nonparametric Alternative")

                else:
                    value_col = selectbox_default("Dependent variable", numeric_cols, first_existing(numeric_cols, ["value", "score", "measurement"], default_numeric_col(df)), key="anova_auto2_value")
                    factor_candidates = categorical_candidate_cols(df, exclude=[value_col])
                    factor_a = selectbox_default("Factor A", factor_candidates, first_existing(factor_candidates, ["factor_a", "group", "treatment"], factor_candidates[0] if factor_candidates else None), key="anova_auto2_a")
                    factor_b_options = [c for c in factor_candidates if c != factor_a]
                    factor_b = selectbox_default("Factor B", factor_b_options, first_existing(factor_b_options, ["factor_b", "time", "condition"], factor_b_options[0] if factor_b_options else None), key="anova_auto2_b")
                    if st.button("Run two-way ANOVA", type="primary", use_container_width=True):
                        d = df[[value_col, factor_a, factor_b]].copy()
                        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
                        d[factor_a] = d[factor_a].astype(str)
                        d[factor_b] = d[factor_b].astype(str)
                        d = d.dropna()
                        if d[factor_a].nunique() < 2 or d[factor_b].nunique() < 2:
                            raise ValueError("Two-way ANOVA requires at least 2 levels in each factor.")
                        formula = f'Q("{value_col}") ~ C(Q("{factor_a}")) * C(Q("{factor_b}"))'
                        model = smf.ols(formula=formula, data=d).fit()
                        anova_tbl = anova_summary_table(model, typ=2)
                        show_table(anova_tbl, "Tests of Between-Subjects Effects")
                        eta_tbl = eta_squared_from_anova_table(anova_tbl)
                        if not eta_tbl.empty:
                            show_table(eta_tbl, "Effect Size")
            except Exception as e:
                st.error(f"Failed: {e}")


# -----------------------------
# DIAGNOSTIC PROBABILITY
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
        show_table(summary, "Predictive Values")
        download_table_block(summary, "diagnostic_predictive_values", "Predictive Values")
        show_table(table, f"Expected Results per {int(population)} People")
        download_table_block(table, "diagnostic_expected_results", "Expected Results")
        show_table(calc, "Calculations")
        download_table_block(calc, "diagnostic_calculations", "Calculations")

# -----------------------------
# CONFIDENCE INTERVALS — PROPORTION
# -----------------------------
elif section == "Confidence Intervals" and sub == "Proportion":
    st.markdown("## Confidence Intervals — Proportion")
    st.write("Estimate a proportion and confidence intervals. Wald is used only when np and n(1-p) are both at least 5; otherwise Wilson and exact-style intervals are recommended.")
    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.number_input("Number with event / success", min_value=0, value=50, step=1)
    with c2:
        n = st.number_input("Total sample size", min_value=1, value=100, step=1)
    with c3:
        conf_level = st.slider("Confidence level", min_value=0.80, max_value=0.99, value=0.95, step=0.01, key="prop_ci_level")
    if int(x) > int(n):
        st.error("Number with event cannot be greater than total sample size.")
    else:
        if st.button("Compute proportion CI", type="primary", use_container_width=True):
            tbl, wald_ok = proportion_ci_methods(int(x), int(n), float(conf_level))
            if not wald_ok:
                st.warning("Wald condition is not satisfied because np or n(1-p) is below 5. Prefer Wilson, exact (Clopper-Pearson), Agresti-Coull, or Jeffreys intervals.")
            else:
                st.success("Wald condition is satisfied. Wald CI can be reported, but Wilson is also commonly recommended.")
            show_table(tbl, "Proportion Confidence Intervals (%)")
            download_table_block(tbl, "proportion_confidence_intervals", "Proportion CI")

# -----------------------------
# CONFIDENCE INTERVALS
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
        txt = st.text_area("Paste numeric values (comma/space/newline separated)", height=120)
        if txt.strip():
            parts = [p for p in txt.replace(",", " ").split() if p.strip()]
            vals = pd.to_numeric(pd.Series(parts), errors="coerce").dropna()
            x = vals.values

    conf_level = st.slider("Confidence level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    alpha_tail = 1 - conf_level
    force_boot = st.checkbox("Force bootstrap (recommended if non-normal)", value=False)
    n_boot = st.number_input("Bootstrap resamples", min_value=1000, max_value=20000, value=5000, step=500)

    # Normality check
    decision = "Unknown"
    if x is not None and len(x) >= 3:
        sh_stat, sh_p = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)
        norm_tbl = pd.DataFrame([["Shapiro-Wilk", sh_stat, format_p_value(sh_p)]], columns=["Test", "Statistic", "Sig."])
        norm_tbl = compact_numeric_df(norm_tbl, decimals=4)
        show_table(norm_tbl, "Tests of Normality")
        download_table_block(norm_tbl, "ci_normality", "Normality")
        if isinstance(sh_p, float) and not np.isnan(sh_p) and sh_p < 0.05:
            decision = "Non-normal"
        else:
            decision = "Normal"

    st.markdown("### Confidence Interval Results")
    if st.button("Compute CI", type="primary", use_container_width=True):
        if x is None or len(x) < 2:
            st.info("Provide at least 2 numeric values.")
        else:
            try:
                use_boot = force_boot or (decision == "Non-normal")

                if not use_boot:
                    t1 = ci_mean(x, alpha=alpha_tail)
                    t2 = ci_variance(x, alpha=alpha_tail)
                    show_table(t1, "Confidence Interval for Mean")
                    download_table_block(t1, "ci_mean", "CI Mean")
                    show_table(t2, "Confidence Interval for Variance")
                    download_table_block(t2, "ci_variance", "CI Variance")
                else:
                    mean_est = float(np.mean(x))
                    var_est = float(np.var(x, ddof=1))
                    m_lo, m_hi = bootstrap_ci(x, lambda a: float(np.mean(a)), n_boot=int(n_boot), alpha=alpha_tail)
                    v_lo, v_hi = bootstrap_ci(x, lambda a: float(np.var(a, ddof=1)), n_boot=int(n_boot), alpha=alpha_tail)

                    t1 = pd.DataFrame([["Mean (bootstrap)", mean_est, m_lo, m_hi]],
                                      columns=["Parameter", "Estimate", "CI 2.5%", "CI 97.5%"])
                    t2 = pd.DataFrame([["Variance (bootstrap)", var_est, v_lo, v_hi]],
                                      columns=["Parameter", "Estimate", "CI 2.5%", "CI 97.5%"])

                    t1 = compact_numeric_df(t1, decimals=4)
                    t2 = compact_numeric_df(t2, decimals=4)

                    show_table(t1, "Confidence Interval for Mean (Bootstrap)")
                    download_table_block(t1, "ci_mean_bootstrap", "CI Mean Bootstrap")
                    show_table(t2, "Confidence Interval for Variance (Bootstrap)")
                    download_table_block(t2, "ci_variance_bootstrap", "CI Variance Bootstrap")

            except Exception as e:
                st.error(f"Failed: {e}")
