# app.py
# =========================================================
# Regression Applications in Health Sciences
# Streamlit app (single-file, robust navigation with expanders + buttons)
# Notes: All UI text is English. Outputs aim to be SPSS-like tables.
# =========================================================

import io
import math
from dataclasses import dataclass
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
# CSS (blue sidebar + button-like nav + compact tables)
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

/* Make ALL sidebar buttons consistent */
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stDownloadButton button{
  width: 100% !important;
  background: rgba(255,255,255,0.10) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.30) !important;
  border-radius: 14px !important;
  font-weight: 600 !important;
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
  font-weight: 700 !important;
}

/* File uploader - reduce "white card" look */
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
  font-weight: 700 !important;
}

/* Main header banner */
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

/* Compact dataframe: a bit smaller */
div[data-testid="stDataFrame"]{
  font-size: 13px;
}
</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# Helpers: download bytes
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
        # Render table -> image (simple)
        fig, ax = plt.subplots(figsize=(min(18, max(7, df.shape[1] * 1.5)),
                                        min(18, max(2.2, (df.shape[0] + 1) * 0.45))))
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12, pad=8)
        disp = df.copy()
        disp = disp.fillna("")
        tbl = ax.table(cellText=disp.values, colLabels=disp.columns, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.15)
        png = fig_to_png_bytes(fig)
        plt.close(fig)
        st.download_button(
            "Download PNG",
            data=png,
            file_name=f"{base_name}.png",
            mime="image/png",
            use_container_width=True
        )

def show_table(df: pd.DataFrame, title: str):
    st.markdown(f"### {title}")
    st.dataframe(df.fillna(""), use_container_width=True)

# =========================================================
# Formatting: p-values (SPSS-like)
# =========================================================
def format_p_value(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return ""
    try:
        p = float(p)
    except Exception:
        return ""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"

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
# Header banner (single, no extra big title below)
# =========================================================
st.markdown(
    """
<div class="header-banner">
  <h1>Regression Applications in Health Sciences</h1>
  <p>Unified platform for regression modeling and hypothesis testing, with diagnostics and exports.</p>
</div>
""",
    unsafe_allow_html=True
)

st.write("")  # small gap

# =========================================================
# Sidebar navigation (expanders + buttons)
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
        if st.button("Model Comparison", key="log_cmp", use_container_width=True):
            set_nav("Logistic Regression", "Model Comparison")
        if st.button("Export", key="log_export", use_container_width=True):
            set_nav("Logistic Regression", "Export")

    with st.expander("Linear Regression (Multivariable)", expanded=(st.session_state.section == "Linear Regression")):
        if st.button("Data (Upload & Template)", key="lin_data", use_container_width=True):
            set_nav("Linear Regression", "Data")
        if st.button("Assumptions & Diagnostics", key="lin_diag", use_container_width=True):
            set_nav("Linear Regression", "Diagnostics")
        if st.button("Modeling (Coefficients, ANOVA)", key="lin_model", use_container_width=True):
            set_nav("Linear Regression", "Modeling")

    with st.expander("ANOVA", expanded=(st.session_state.section == "ANOVA")):
        if st.button("One-way (Between-subjects)", key="a_1", use_container_width=True):
            set_nav("ANOVA", "One-way Between")
        if st.button("One-way (Repeated measures)", key="a_2", use_container_width=True):
            set_nav("ANOVA", "One-way Repeated")
        if st.button("Two-way (Repeated measures)", key="a_3", use_container_width=True):
            set_nav("ANOVA", "Two-way Repeated")

    with st.expander("t-test", expanded=(st.session_state.section == "t-test")):
        if st.button("One-sample", key="t_1", use_container_width=True):
            set_nav("t-test", "One-sample")
        if st.button("Independent samples", key="t_2", use_container_width=True):
            set_nav("t-test", "Independent")
        if st.button("Paired samples", key="t_3", use_container_width=True):
            set_nav("t-test", "Paired")

    with st.expander("Categorical Tests", expanded=(st.session_state.section == "Categorical Tests")):
        if st.button("Contingency Table (r×c) / Chi-square", key="c_1", use_container_width=True):
            set_nav("Categorical Tests", "Chi-square r×c")
        if st.button("Fisher's Exact (2×2)", key="c_2", use_container_width=True):
            set_nav("Categorical Tests", "Fisher 2×2")
        if st.button("Goodness-of-fit", key="c_3", use_container_width=True):
            set_nav("Categorical Tests", "Goodness-of-fit")
        if st.button("Mantel–Haenszel (Stratified 2×2)", key="c_4", use_container_width=True):
            set_nav("Categorical Tests", "Mantel–Haenszel")
        if st.button("Cochran's Q (+ McNemar post-hoc)", key="c_5", use_container_width=True):
            set_nav("Categorical Tests", "Cochran Q")

    with st.expander("Confidence Intervals", expanded=(st.session_state.section == "Confidence Intervals")):
        if st.button("Mean & Variance CI", key="ci_1", use_container_width=True):
            set_nav("Confidence Intervals", "Mean & Variance")


# =========================================================
# Data storage in session (shared)
# =========================================================
if "df_data" not in st.session_state:
    st.session_state.df_data = None
if "df_name" not in st.session_state:
    st.session_state.df_name = ""


# =========================================================
# Shared: data upload utilities
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

def sidebar_dataset_uploader(template_df: pd.DataFrame, template_name: str):
    """Sidebar widget: download template + upload dataset."""
    with st.sidebar:
        st.markdown("---")
        st.download_button(
            "Download Excel template",
            data=df_to_excel_bytes({template_name: template_df}),
            file_name=f"{template_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.caption("Upload CSV/XLSX • Use the template to keep column names consistent.")
        up = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"], key=f"uploader_{template_name}")
        if up is not None:
            df = load_uploaded_file(up)
            st.session_state.df_data = df
            st.session_state.df_name = up.name

def require_dataset() -> pd.DataFrame:
    df = st.session_state.df_data
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("No dataset loaded yet. Please upload a CSV/XLSX file in the Data section.")
        raise RuntimeError("No dataset")
    return df

# =========================================================
# SPSS-like numeric display helpers
# =========================================================
def clean_float_cell(x):
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

def compact_numeric_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)
    out = out.applymap(clean_float_cell)
    return out


# =========================================================
# Contingency editor (labels editable + Total)
# =========================================================
def contingency_editor(
    key: str,
    default_rows: List[str],
    default_cols: List[str],
    default_counts: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      counts_df: display table with Group + Total row/col (SPSS-like)
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

        # clean unique
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

    # SPSS-like totals
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
# Categorical pipeline (IMPORTANT)
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
# 2×2 Measures: OR, RR, VE, diagnostic accuracy + CI
# =========================================================
def _safe_div(a, b):
    return np.nan if b == 0 else a / b

def wilson_ci(x, n, alpha=0.05) -> Tuple[float, float]:
    # Wilson score interval for proportion
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
    # Layout: [[a, b],
    #          [c, d]]
    # Here we interpret:
    # a = Exposed & Outcome+
    # b = Exposed & Outcome-
    # c = Unexposed & Outcome+
    # d = Unexposed & Outcome-
    a, b, c, d = obs2x2[0,0], obs2x2[0,1], obs2x2[1,0], obs2x2[1,1]

    # Add 0.5 correction for OR/RR if any zero
    cc = 0.5 if min(a,b,c,d) == 0 else 0.0
    a2, b2, c2, d2 = a+cc, b+cc, c+cc, d+cc

    # OR and CI (Woolf)
    OR = (a2*d2) / (b2*c2)
    se_log_or = math.sqrt(1/a2 + 1/b2 + 1/c2 + 1/d2)
    or_lo, or_hi = log_ci_ratio(OR, se_log_or, alpha)

    # RR and CI (log method)
    risk_e = _safe_div(a2, (a2+b2))
    risk_u = _safe_div(c2, (c2+d2))
    RR = _safe_div(risk_e, risk_u)
    se_log_rr = math.sqrt((1/a2) - (1/(a2+b2)) + (1/c2) - (1/(c2+d2)))
    rr_lo, rr_hi = log_ci_ratio(RR, se_log_rr, alpha)

    # VE (vaccine effectiveness) = 1 - RR
    VE = 1 - RR
    # CI for VE from RR CI
    ve_lo, ve_hi = 1 - rr_hi, 1 - rr_lo

    # Diagnostic accuracy: treat "Exposed" row as test+? Commonly:
    # Let's define:
    # TP=a, FP=b, FN=c, TN=d
    TP, FP, FN, TN = a, b, c, d
    sens = _safe_div(TP, TP+FN)
    spec = _safe_div(TN, TN+FP)
    fpr  = _safe_div(FP, FP+TN)
    fnr  = _safe_div(FN, FN+TP)
    ppv  = _safe_div(TP, TP+FP)
    npv  = _safe_div(TN, TN+FN)
    lr_p = _safe_div(sens, 1-spec) if (1-spec) not in [0, np.nan] else np.nan
    lr_n = _safe_div(1-sens, spec) if spec not in [0, np.nan] else np.nan

    # CI for proportions using Wilson
    sens_lo, sens_hi = wilson_ci(TP, TP+FN, alpha) if (TP+FN)>0 else (np.nan, np.nan)
    spec_lo, spec_hi = wilson_ci(TN, TN+FP, alpha) if (TN+FP)>0 else (np.nan, np.nan)
    ppv_lo, ppv_hi = wilson_ci(TP, TP+FP, alpha) if (TP+FP)>0 else (np.nan, np.nan)
    npv_lo, npv_hi = wilson_ci(TN, TN+FN, alpha) if (TN+FN)>0 else (np.nan, np.nan)

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
    df = compact_numeric_df(df, decimals=4)
    return df


# =========================================================
# Logistic regression (statsmodels): OR table + ROC
# =========================================================
def run_logistic_statsmodels(df: pd.DataFrame, target: str, features: List[str]) -> Dict[str, object]:
    data = df[[target] + features].dropna().copy()
    if data[target].nunique() != 2:
        raise ValueError("Target must have exactly 2 unique values (e.g., 0/1).")

    y = data[target].astype(int)
    X = data[features]
    X_sm = sm.add_constant(X)

    model = sm.Logit(y, X_sm).fit(disp=False)
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues

    # Table similar to SPSS "Variables in the Equation"
    # Keep raw p for significance; display formatted p
    tbl = pd.DataFrame({
        "Term": params.index,
        "B": params.values,
        "S.E.": model.bse.values,
        "Wald": (params.values / model.bse.values) ** 2,
        "df": 1,
        "Sig.": [format_p_value(p) for p in pvals.values],
        "Exp(B)": np.exp(params.values),
        "95% CI for Exp(B) Lower": np.exp(conf[0].values),
        "95% CI for Exp(B) Upper": np.exp(conf[1].values),
        "_praw": pvals.values
    })

    # Hide Exp(B) and CI for constant/intercept (SPSS style)
    is_const = tbl["Term"].isin(["const", "Intercept"])
    tbl.loc[is_const, ["Exp(B)", "95% CI for Exp(B) Lower", "95% CI for Exp(B) Upper"]] = ""

    # Round numeric columns; keep Sig. as text
    for c in ["B", "S.E.", "Wald", "Exp(B)", "95% CI for Exp(B) Lower", "95% CI for Exp(B) Upper"]:
        tbl[c] = pd.to_numeric(tbl[c], errors="coerce").round(4)
    tbl = tbl.drop(columns=["_praw"])
    tbl = tbl.applymap(clean_float_cell)

    return {"model": model, "table": tbl, "y": y, "X": X}


def roc_curve_table(model, y: pd.Series, X: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    # Predict probabilities
    X_sm = sm.add_constant(X)
    p = model.predict(X_sm)

    fpr, tpr, thr = statsmodels_roc(y.values, p.values)
    auc_val = auc_from_roc(fpr, tpr)

    roc_df = pd.DataFrame({
        "Threshold": thr,
        "Sensitivity (TPR)": tpr,
        "1 - Specificity (FPR)": fpr
    })
    roc_df = compact_numeric_df(roc_df, decimals=4)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"Logistic (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    auc_tbl = pd.DataFrame([["Area Under the Curve", auc_val]], columns=["Measure", "Value"])
    auc_tbl["Value"] = auc_tbl["Value"].round(4)
    return auc_tbl, roc_df, fig

def statsmodels_roc(y_true: np.ndarray, y_prob: np.ndarray):
    # Manual ROC to avoid sklearn dependency
    # Sort by descending prob
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    thresholds = np.r_[np.inf, np.unique(y_prob)]
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    tpr = []
    fpr = []
    thr_list = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        tpr.append(TP / P if P > 0 else 0.0)
        fpr.append(FP / N if N > 0 else 0.0)
        thr_list.append(t if np.isfinite(t) else 1.0)
    return np.array(fpr), np.array(tpr), np.array(thr_list)

def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    # trapezoid
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


# =========================================================
# Linear regression: formula-based (fixes design_info issue)
# =========================================================
def fit_linear_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]):
    data = df[[y_col] + x_cols].dropna().copy()
    if len(data) < 3:
        raise ValueError("Not enough rows after dropping missing values.")

    # Use formula to avoid 'design_info' errors and get clean ANOVA terms
    # Quote variable names safely
    def q(name: str) -> str:
        return f'Q("{name}")'

    formula = q(y_col) + " ~ " + " + ".join(q(c) for c in x_cols)
    model = smf.ols(formula=formula, data=data).fit()
    return model, data


def clean_term_name(s: str) -> str:
    # Remove Q("...") wrappers
    if isinstance(s, str) and s.startswith('Q("') and s.endswith('")'):
        return s[3:-2]
    return s


# =========================================================
# Confidence intervals: mean & variance (parametric) + bootstrap option
# =========================================================
def ci_mean(x: np.ndarray, alpha=0.05) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    sd = x.std(ddof=1)
    tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
    se = sd / math.sqrt(n)
    lo = mean - tcrit * se
    hi = mean + tcrit * se
    df = pd.DataFrame([["Mean", mean, lo, hi]], columns=["Parameter", "Estimate", "CI 2.5%", "CI 97.5%"])
    return compact_numeric_df(df, decimals=4)

def ci_variance(x: np.ndarray, alpha=0.05) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    n = len(x)
    s2 = x.var(ddof=1)
    chi2_lo = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_hi = stats.chi2.ppf(1 - alpha/2, df=n-1)
    lo = (n-1)*s2 / chi2_hi
    hi = (n-1)*s2 / chi2_lo
    df = pd.DataFrame([["Variance", s2, lo, hi]], columns=["Parameter", "Estimate", "CI 2.5%", "CI 97.5%"])
    return compact_numeric_df(df, decimals=4)

def bootstrap_ci(x: np.ndarray, func, n_boot=5000, alpha=0.05, seed=123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    stats_ = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        stats_.append(func(samp))
    stats_ = np.sort(stats_)
    lo = np.quantile(stats_, alpha/2)
    hi = np.quantile(stats_, 1 - alpha/2)
    return float(lo), float(hi)


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
        "Use the navigation sidebar to select a module. Upload data via the relevant Data page, "
        "then run analyses and export SPSS-like tables."
    )
    if st.session_state.df_data is not None:
        st.success(f"Loaded dataset: {st.session_state.df_name} • shape={st.session_state.df_data.shape}")
        st.dataframe(st.session_state.df_data.head(20), use_container_width=True)

# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------
elif section == "Logistic Regression":
    if sub == "Data":
        st.markdown("## Logistic Regression — Data")
        template = pd.DataFrame({
            "target_binary": [0, 1, 0],
            "feature1": [1.2, 0.7, 1.1],
            "feature2": [10, 12, 9]
        })
        sidebar_dataset_uploader(template, "logistic_template")
        if st.session_state.df_data is not None:
            st.success(f"Loaded dataset: {st.session_state.df_name} • shape={st.session_state.df_data.shape}")
            st.dataframe(st.session_state.df_data.head(50), use_container_width=True)

    elif sub == "EDA":
        st.markdown("## Logistic Regression — EDA")
        df = require_dataset()
        st.dataframe(df.head(50), use_container_width=True)
        st.caption("Tip: Use EDA to check missing values, distributions, and basic summaries.")

        st.markdown("### Missing values")
        miss = df.isna().sum().reset_index()
        miss.columns = ["Variable", "Missing"]
        show_table(miss, "Missing Values")
        download_table_block(miss, "missing_values", "Missing Values")

        st.markdown("### Descriptive statistics (numeric)")
        desc = df.describe(include=[np.number]).T.reset_index().rename(columns={"index": "Variable"})
        desc = compact_numeric_df(desc, decimals=4)
        show_table(desc, "Descriptives")
        download_table_block(desc, "descriptives", "Descriptives")

    elif sub == "Modeling":
        st.markdown("## Logistic Regression — Modeling (OR, ROC)")
        df = require_dataset()

        cols = list(df.columns)
        target = st.selectbox("Target column (binary 0/1)", options=cols)
        features = st.multiselect("Predictors (numeric)", options=[c for c in cols if c != target])

        if st.button("Run Logistic Regression", type="primary", use_container_width=True):
            try:
                res = run_logistic_statsmodels(df, target, features)

                # Variables in the Equation (SPSS name)
                show_table(res["table"], "Variables in the Equation")
                download_table_block(res["table"], "logistic_variables_in_equation", "Variables in the Equation")

                # ROC
                auc_tbl, roc_tbl, fig = roc_curve_table(res["model"], res["y"], res["X"])
                show_table(auc_tbl, "ROC — Area Under the Curve")
                download_table_block(auc_tbl, "logistic_auc", "AUC")

                show_table(roc_tbl.head(50), "ROC Coordinates (first 50 rows)")
                download_table_block(roc_tbl, "logistic_roc_coordinates", "ROC Coordinates")

                st.pyplot(fig)
                st.download_button(
                    "Download ROC PNG",
                    data=fig_to_png_bytes(fig),
                    file_name="logistic_roc.png",
                    mime="image/png",
                    use_container_width=False
                )
                plt.close(fig)

            except Exception as e:
                st.error(f"Modeling failed: {e}")

    elif sub == "Model Comparison":
        st.markdown("## Logistic Regression — Model Comparison")
        st.info(
            "This simplified build includes the core SPSS-like Logistic outputs. "
            "If you want full ML model comparison (RF/SVM/KNN + AUC/ROC per model) exactly like your old app, "
            "tell me and I will add it back in this single-file structure (robust, no indent issues)."
        )

    elif sub == "Export":
        st.markdown("## Logistic Regression — Export")
        st.info("Exports are available directly under each output table/figure.")

# -----------------------------
# LINEAR REGRESSION
# -----------------------------
elif section == "Linear Regression":
    if sub == "Data":
        st.markdown("## Linear Regression — Data")
        template = pd.DataFrame({
            "Y_outcome": [10.2, 12.1, 9.8],
            "X1": [1.1, 0.8, 1.4],
            "X2": [100, 120, 95],
        })
        sidebar_dataset_uploader(template, "linear_template")
        if st.session_state.df_data is not None:
            st.success(f"Loaded dataset: {st.session_state.df_name} • shape={st.session_state.df_data.shape}")
            st.dataframe(st.session_state.df_data.head(50), use_container_width=True)

    elif sub == "Diagnostics":
        st.markdown("## Linear Regression — Assumptions & Diagnostics")
        df = require_dataset()
        cols = list(df.columns)
        y_col = st.selectbox("Dependent variable (Y)", options=cols)
        x_cols = st.multiselect("Independent variables (X)", options=[c for c in cols if c != y_col])

        if st.button("Run diagnostics", type="primary", use_container_width=True):
            try:
                model, data_used = fit_linear_ols(df, y_col, x_cols)
                resid = model.resid
                fitted = model.fittedvalues

                # Normality (Shapiro)
                sh_stat, sh_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
                norm_tbl = pd.DataFrame([["Shapiro-Wilk", sh_stat, format_p_value(sh_p)]],
                                        columns=["Test", "Statistic", "Sig."])
                show_table(norm_tbl, "Tests of Normality (Residuals)")
                download_table_block(norm_tbl, "linear_resid_normality", "Normality")

                # Homoscedasticity (Breusch–Pagan)
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(resid, model.model.exog)
                bp_tbl = pd.DataFrame([["Breusch-Pagan", bp_lm, format_p_value(bp_p)]],
                                      columns=["Test", "LM", "Sig."])
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
                    vif_rows.append([names[i], variance_inflation_factor(X_exog, i)])
                vif_df = pd.DataFrame(vif_rows, columns=["Predictor", "VIF"])
                vif_df["Predictor"] = vif_df["Predictor"].apply(clean_term_name)
                vif_df = compact_numeric_df(vif_df, decimals=4)
                show_table(vif_df, "Collinearity Statistics (VIF)")
                download_table_block(vif_df, "linear_vif", "VIF")

                # Residual plots
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.scatter(fitted, resid)
                ax1.axhline(0, linestyle="--")
                ax1.set_xlabel("Fitted values")
                ax1.set_ylabel("Residuals")
                ax1.set_title("Residuals vs Fitted")
                st.pyplot(fig1)
                st.download_button("Download plot PNG", fig_to_png_bytes(fig1), "residuals_vs_fitted.png", "image/png")
                plt.close(fig1)

            except Exception as e:
                st.error(f"Diagnostics failed: {e}")

    elif sub == "Modeling":
        st.markdown("## Linear Regression — Modeling (Coefficients, ANOVA)")
        df = require_dataset()
        cols = list(df.columns)
        y_col = st.selectbox("Dependent variable (Y)", options=cols)
        x_cols = st.multiselect("Independent variables (X)", options=[c for c in cols if c != y_col])

        if st.button("Run linear regression", type="primary", use_container_width=True):
            try:
                model, data_used = fit_linear_ols(df, y_col, x_cols)

                # ANOVA table (Type I) — SPSS-like
                a = anova_lm(model, typ=1).reset_index().rename(columns={"index": "Source"})
                a["Source"] = a["Source"].apply(clean_term_name)
                a = a.rename(columns={
                    "df": "df",
                    "sum_sq": "Sum Sq",
                    "mean_sq": "Mean Sq",
                    "F": "F",
                    "PR(>F)": "Sig."
                })
                a["Sig."] = a["Sig."].apply(format_p_value)
                for col in ["Sum Sq", "Mean Sq", "F"]:
                    if col in a.columns:
                        a[col] = pd.to_numeric(a[col], errors="coerce").round(4)
                a = a.applymap(clean_float_cell)

                show_table(a, "ANOVA")
                download_table_block(a, "linear_anova", "ANOVA")

                # Coefficients — SPSS-like
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
                b["Significant (p<0.05)"] = b["Sig."].apply(lambda s: "Yes" if (s and not s.startswith("<") and float(s) < 0.05) else ("Yes" if s == "< 0.001" else "No"))
                # Move Significant to last
                sig_col = b.pop("Significant (p<0.05)")
                b["Significant (p<0.05)"] = sig_col

                for col in ["B", "S.E.", "t", "CI 2.5%", "CI 97.5%"]:
                    if col in b.columns:
                        b[col] = pd.to_numeric(b[col], errors="coerce").round(4)
                b = b.applymap(clean_float_cell)

                show_table(b, "Coefficients")
                download_table_block(b, "linear_coefficients", "Coefficients")

            except Exception as e:
                st.error(f"Modeling failed: {e}")

# -----------------------------
# ANOVA (basic skeleton)
# -----------------------------
elif section == "ANOVA":
    st.markdown("## ANOVA")
    st.info(
        "ANOVA modules require dedicated templates and repeated-measures structures. "
        "This build keeps ANOVA within Linear Regression output (ANOVA table). "
        "If you want full one-way/two-way repeated-measures like SPSS (with Mauchly, Greenhouse-Geisser, post-hoc), "
        "tell me and I will add them as separate pages with templates."
    )

# -----------------------------
# t-test (basic robust)
# -----------------------------
elif section == "t-test":
    st.markdown("## t-test")
    st.info(
        "This build focuses on a stable architecture. If you need the full auto-switch (normal→t-test; non-normal→Mann-Whitney/Wilcoxon), "
        "I can expand this module next."
    )

# -----------------------------
# CATEGORICAL TESTS
# -----------------------------
elif section == "Categorical Tests":

    if sub == "Chi-square r×c":
        st.markdown("## Categorical Tests — Contingency Table (r×c) / Chi-square")

        # Input table: r×c
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
                chi_tbl["Value"] = chi_tbl["Value"].round(6)
                show_table(chi_tbl, "Chi-Square Tests")
                download_table_block(chi_tbl, "chisq_tests", "Chi-Square Tests")

                exp_df = pd.DataFrame(expected, columns=observed_df.columns)
                exp_df.insert(0, "Group", st.session_state["ct_chisq"]["Group"].tolist())
                exp_df["Total"] = exp_df[observed_df.columns].sum(axis=1)
                total_row = {"Group": "Total"}
                for c in observed_df.columns:
                    total_row[c] = float(exp_df[c].sum())
                total_row["Total"] = float(exp_df["Total"].sum())
                exp_df = pd.concat([exp_df, pd.DataFrame([total_row])], ignore_index=True)
                for c in observed_df.columns.tolist() + ["Total"]:
                    exp_df[c] = pd.to_numeric(exp_df[c], errors="coerce").round(4)
                exp_df = exp_df.applymap(clean_float_cell)

                show_table(exp_df, "Expected Frequencies")
                download_table_block(exp_df, "chisq_expected", "Expected Frequencies")

                # If 2×2, also show OR/RR/VE and diagnostic metrics
                if obs.shape == (2, 2):
                    meas = two_by_two_measures(obs, alpha=0.05)
                    show_table(meas, "2×2 Measures (OR, RR, VE, Diagnostic Accuracy)")
                    download_table_block(meas, "chisq_2x2_measures", "2×2 Measures")

            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Fisher 2×2":
        st.markdown("## Categorical Tests — Fisher's Exact Test (2×2)")

        # FIXED 2×2 editor (prevents the common error)
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
                tbl = tbl.applymap(clean_float_cell)

                show_table(tbl, "Fisher's Exact Test")
                download_table_block(tbl, "fisher_exact", "Fisher's Exact Test")

                meas = two_by_two_measures(obs, alpha=0.05)
                show_table(meas, "2×2 Measures (OR, RR, VE, Diagnostic Accuracy)")
                download_table_block(meas, "fisher_2x2_measures", "2×2 Measures")

            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Goodness-of-fit":
        st.markdown("## Categorical Tests — Goodness-of-fit (Chi-square)")

        st.caption("Upload a one-column observed counts vector (and optional expected counts).")

        template = pd.DataFrame({"Category": ["A", "B", "C"], "Observed": [30, 50, 20], "Expected(optional)": [33.3, 33.3, 33.3]})
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
                    tbl["Value"] = tbl["Value"].round(6)
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
                    download_table_block(out, "mh_results", "Mantel-Haenszel")

            except Exception as e:
                st.error(f"Failed: {e}")

    elif sub == "Cochran Q":
        st.markdown("## Categorical Tests — Cochran's Q (+ McNemar post-hoc)")
        st.info(
            "Cochran's Q requires paired binary outcomes across ≥3 conditions. "
            "If you want this module fully (including McNemar pairwise + Bonferroni), tell me and I will add it."
        )

# -----------------------------
# CONFIDENCE INTERVALS
# -----------------------------
elif section == "Confidence Intervals" and sub == "Mean & Variance":
    st.markdown("## Confidence Intervals — Mean & Variance")
    st.caption("Upload a numeric column (dataset) OR paste values. If non-normal, bootstrap CI is recommended.")

    # Template for CI
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

    alpha = st.slider("Confidence level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    alpha_tail = 1 - alpha

    force_boot = st.checkbox("Force bootstrap (recommended if non-normal)", value=False)
    n_boot = st.number_input("Bootstrap resamples", min_value=1000, max_value=20000, value=5000, step=500)

    # Normality check
    if x is not None and len(x) >= 3:
        sh_stat, sh_p = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)
        norm_tbl = pd.DataFrame([["Shapiro-Wilk", sh_stat, format_p_value(sh_p)]], columns=["Test", "Statistic", "Sig."])
        show_table(norm_tbl, "Tests of Normality")
        download_table_block(norm_tbl, "ci_normality", "Normality")
        decision = "Non-normal" if (isinstance(sh_p, float) and not np.isnan(sh_p) and sh_p < 0.05) else "Normal"
    else:
        decision = "Unknown"

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
                    # Bootstrap CI for mean and variance
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
