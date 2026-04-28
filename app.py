# app.py
# =========================================================
# Regression Applications in Health Sciences
# Single-file Streamlit app (stable: expanders + buttons)
# =========================================================

import io
import math
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Regression Applications in Health Sciences",
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

/* Slightly compact DataFrame font */
div[data-testid="stDataFrame"]{
  font-size: 13px;
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
  <h1>Regression Applications in Health Sciences</h1>
  <p>Regression modeling + hypothesis testing, with diagnostics and SPSS-like outputs.</p>
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
    df2 = df.copy().fillna("")
    fig_w = min(18, max(7, df2.shape[1] * 1.5))
    fig_h = min(18, max(2.2, (df2.shape[0] + 1) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    tbl = ax.table(cellText=df2.values, colLabels=df2.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.15)
    png = fig_to_png_bytes(fig, dpi=dpi)
    plt.close(fig)
    return png

def download_table_block(df: pd.DataFrame, base_name: str, title: str = ""):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Download Excel",
            data=df_to_excel_bytes({base_name: df}),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "Download PNG",
            data=df_to_png_bytes(df, title=title),
            file_name=f"{base_name}.png",
            mime="image/png",
            use_container_width=True
        )

def download_figure_block(fig: plt.Figure, base_name: str):
    st.download_button(
        "Download PNG",
        data=fig_to_png_bytes(fig),
        file_name=f"{base_name}.png",
        mime="image/png",
        use_container_width=False
    )

def show_table(df: pd.DataFrame, title: str):
    st.markdown(f"### {title}")
    st.dataframe(df.fillna(""), use_container_width=True)

# =========================================================
# SPSS-like formatting
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

    # pandas >= 2.1 supports DataFrame.map; older versions use applymap
    if hasattr(out, "map"):
        return out.map(clean_cell)
    return out.applymap(clean_cell)

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
# Logistic regression (statsmodels): SPSS-like "Variables in the Equation"
# =========================================================
def hosmer_lemeshow_table(y_true, y_prob, g=10):
    """
    SPSS-like Hosmer-Lemeshow test table.
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
    Fit binary logistic regression and return SPSS-like output tables:
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

from sklearn.metrics import auc

def auc_from_roc(fpr, tpr):
    return float(auc(fpr, tpr))
def roc_outputs(model, y: pd.Series, X: pd.DataFrame):
    X_sm = sm.add_constant(X)
    p = model.predict(X_sm)

    fpr, tpr, thr = statsmodels_roc(y.values, p.values)
    auc_val = auc_from_roc(fpr, tpr)

    auc_tbl = pd.DataFrame([["Area Under the Curve", auc_val]], columns=["Measure", "Value"])
    auc_tbl["Value"] = pd.to_numeric(auc_tbl["Value"], errors="coerce").round(4)
    auc_tbl = auc_tbl.applymap(clean_cell)

    roc_tbl = pd.DataFrame({
        "Threshold": thr,
        "Sensitivity (TPR)": tpr,
        "1 - Specificity (FPR)": fpr
    })
    roc_tbl = compact_numeric_df(roc_tbl, decimals=4)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"Logistic (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    return auc_tbl, roc_tbl, fig


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

    with st.expander("Confidence Intervals", expanded=(st.session_state.section == "Confidence Intervals")):
        if st.button("Mean & Variance CI", key="ci_1", use_container_width=True):
            set_nav("Confidence Intervals", "Mean & Variance")


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

                auc_tbl, roc_tbl, fig = roc_outputs(res["model"], res["y"], res["X"])

                show_table(auc_tbl, "ROC — Area Under the Curve")
                download_table_block(auc_tbl, "logistic_auc", "AUC")

                show_table(roc_tbl.head(50), "ROC Coordinates (first 50 rows)")
                download_table_block(roc_tbl, "logistic_roc_coordinates", "ROC Coordinates")

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

                # ANOVA (SPSS-like)
                a = anova_lm(model, typ=1).reset_index().rename(columns={"index": "Source"})
                a["Source"] = a["Source"].apply(clean_term_name)
                a = a.rename(columns={"df": "df", "sum_sq": "Sum Sq", "mean_sq": "Mean Sq", "F": "F", "PR(>F)": "Sig."})
                a["Sig."] = a["Sig."].apply(format_p_value)
                for col in ["Sum Sq", "Mean Sq", "F"]:
                    if col in a.columns:
                        a[col] = pd.to_numeric(a[col], errors="coerce").round(4)
                a = a.applymap(clean_cell)

                show_table(a, "ANOVA")
                download_table_block(a, "linear_anova", "ANOVA")

                # Coefficients (SPSS-like)
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
                b = b.applymap(clean_cell)

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

                chi_tbl = pd.DataFrame([[
                    "Pearson Chi-Square", chi2, dof, format_p_value(p),
                    "Yes" if p < 0.05 else "No"
                ]], columns=["Test", "Value", "df", "Asymp. Sig. (2-sided)", "Significant (p<0.05)"])
                chi_tbl["Value"] = pd.to_numeric(chi_tbl["Value"], errors="coerce").round(6)
                chi_tbl = chi_tbl.applymap(clean_cell)

                show_table(chi_tbl, "Chi-Square Tests")
                download_table_block(chi_tbl, "chisq_tests", "Chi-Square Tests")

                # Expected table (SPSS-like with totals)
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
                exp_df = exp_df.applymap(clean_cell)

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
                tbl = tbl.applymap(clean_cell)

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
            "Observed": [30, 50, 20],
            "Expected(optional)": [33.3, 33.3, 33.3]
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

                obs = pd.to_numeric(df["Observed"], errors="coerce")
                if obs.isna().any():
                    raise ValueError("Observed counts must be numeric.")
                obs = obs.astype(float).values

                exp = None
                if "Expected(optional)" in df.columns:
                    exp_s = pd.to_numeric(df["Expected(optional)"], errors="coerce")
                    if not exp_s.isna().all():
                        exp = exp_s.fillna(0).astype(float).values

                if st.button("Run Goodness-of-fit", type="primary", use_container_width=True):
                    stat, p = stats.chisquare(f_obs=obs, f_exp=exp)
                    tbl = pd.DataFrame([[
                        "Chi-square", stat, len(obs)-1, format_p_value(p),
                        "Yes" if p < 0.05 else "No"
                    ]], columns=["Test", "Value", "df", "Asymp. Sig. (2-sided)", "Significant (p<0.05)"])
                    tbl["Value"] = pd.to_numeric(tbl["Value"], errors="coerce").round(6)
                    tbl = tbl.applymap(clean_cell)

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
