# app.py
# -*- coding: utf-8 -*-
import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, chisquare

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable, mcnemar
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# Page + Header
# =========================================================
st.set_page_config(page_title="Health Science Statistics App", page_icon="📊", layout="wide")

HEADER_HTML = r"""
<div style="
  background: linear-gradient(180deg, #0B3A66 0%, #0A2D4E 100%);
  padding: 24px 24px 20px 24px;
  border-radius: 18px;
  color: white;
  margin-bottom: 12px;
">
  <div style="font-size: 32px; font-weight: 850; letter-spacing: -0.4px;">
    Health Science Statistics App
  </div>
  <div style="margin-top: 8px; font-size: 15px; opacity: 0.95;">
    Regression + hypothesis tests + categorical analysis + diagnostic metrics (SPSS-style tables & exports)
  </div>
</div>
"""
st.markdown(HEADER_HTML, unsafe_allow_html=True)

# =========================================================
# CSS
# =========================================================
st.markdown(
    r"""
<style>
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0B3A66 0%, #0A2D4E 100%) !important;
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

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stDownloadButton button{
  width: 100%;
  background: rgba(255,255,255,0.10) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.30) !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
  padding: 10px 12px !important;
}
section[data-testid="stSidebar"] .stButton button:hover,
section[data-testid="stSidebar"] .stDownloadButton button:hover{
  background: rgba(255,255,255,0.18) !important;
  border-color: rgba(255,255,255,0.48) !important;
}

/* Sidebar uploader: remove white card */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
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

/* main padding */
.block-container { padding-top: 0.8rem !important; }

/* dataframe corners */
div[data-testid="stDataFrame"] { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# GLOBAL: Export helpers
# =========================================================
def df_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, _df in sheets.items():
            _df.to_excel(writer, index=False, sheet_name=sheet[:31])
    bio.seek(0)
    return bio.getvalue()

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 220) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
    bio.seek(0)
    return bio.getvalue()

def df_to_png_bytes(df: pd.DataFrame, title: str = "", dpi: int = 220) -> bytes:
    df2 = df.copy()
    n_rows, n_cols = df2.shape
    fig_w = min(16, max(6.5, n_cols * 1.25))
    fig_h = min(18, max(2.2, (n_rows + 1) * 0.42))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    tbl = ax.table(cellText=df2.values, colLabels=df2.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.12)
    out = fig_to_png_bytes(fig, dpi=dpi)
    plt.close(fig)
    return out

def download_table_block(df: pd.DataFrame, base_name: str, title: str = ""):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Download Excel",
            data=df_to_excel_bytes({base_name: df}),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download PNG",
            data=df_to_png_bytes(df, title=title),
            file_name=f"{base_name}.png",
            mime="image/png",
            use_container_width=True,
        )

def download_figure_block(fig: plt.Figure, base_name: str):
    st.download_button(
        "Download PNG",
        data=fig_to_png_bytes(fig),
        file_name=f"{base_name}.png",
        mime="image/png",
        use_container_width=False,
    )

# =========================================================
# GLOBAL: SPSS-like formatting for all tables
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

def yesno_from_p(p: Optional[float], alpha: float = 0.05) -> str:
    if p is None:
        return ""
    try:
        p = float(p)
    except Exception:
        return ""
    if np.isnan(p) or np.isinf(p):
        return ""
    return "Yes" if p < alpha else "No"

def _blankify(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    # convert NaN/None to empty
    df2 = df2.replace([np.nan, None, "nan", "NaN"], "")
    return df2

def show_table(df: pd.DataFrame, title: str, center: bool = True):
    """
    ONE function for all tables in the app:
    - remove NaN/None
    - compact rounding (floats -> 3 decimals)
    - keep integers as integers (especially Total)
    - center align like SPSS
    """
    st.markdown(f"### {title}")
    d0 = _blankify(df)

    d = d0.copy()
    for col in d.columns:
        if pd.api.types.is_numeric_dtype(d[col]):
            s = pd.to_numeric(d[col], errors="coerce")
            # integer-like?
            if np.all(np.isclose((s.dropna() % 1), 0)):
                d[col] = s.astype("Int64").astype(object).where(s.notna(), "")
            else:
                d[col] = s.round(3).astype(object).where(s.notna(), "")
        else:
            # allow p-value already formatted strings
            pass

    sty = d.style
    if center:
        sty = sty.set_properties(**{"text-align": "center"})
    use_full = d.shape[1] >= 7
    st.dataframe(sty, use_container_width=use_full)

# =========================================================
# GLOBAL: Data I/O + Templates
# =========================================================
def read_dataset(uploaded) -> pd.DataFrame:
    if uploaded is None:
        raise ValueError("No file uploaded.")
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError("Only CSV/XLSX/XLS are supported.")

def make_excel_template(columns: List[str], n_rows: int = 20) -> bytes:
    df = pd.DataFrame({c: [""] * n_rows for c in columns})
    return df_to_excel_bytes({"Template": df})

# =========================================================
# GLOBAL: Contingency table editor (labels editable + Total int)
# =========================================================
def contingency_editor(key: str, default_rows: List[str], default_cols: List[str], default_counts: Optional[np.ndarray] = None):
    """
    Editable row/column names + counts.
    Returns:
      counts_df: DataFrame with index=row labels and columns=column labels (NO Total)
      observed_df: DataFrame for display with Group + counts + Total row/col (int)
    """
    state_key = f"ct_{key}"
    if state_key not in st.session_state:
        if default_counts is None:
            default_counts = np.ones((len(default_rows), len(default_cols)), dtype=int)
        df0 = pd.DataFrame(default_counts, columns=default_cols)
        df0.insert(0, "Group", default_rows)
        st.session_state[state_key] = df0

    st.markdown("### Contingency Table (Counts)")
    st.caption("Edit row/column names and counts. Totals are computed automatically (SPSS-style).")

    df_edit = st.session_state[state_key]

    # Column name editor
    col_labels = [c for c in df_edit.columns if c != "Group"]
    with st.expander("Edit column names", expanded=False):
        new_cols = []
        cols = st.columns(len(col_labels))
        for i, c in enumerate(col_labels):
            with cols[i]:
                new_cols.append(st.text_input(f"Column {i+1}", value=c, key=f"{state_key}_col_{i}"))
        if st.button("Apply column names", key=f"{state_key}_apply_cols", use_container_width=True):
            rename = {old: new for old, new in zip(col_labels, new_cols) if new and new != old}
            st.session_state[state_key] = df_edit.rename(columns=rename)
            st.rerun()

    edited = st.data_editor(
        df_edit,
        use_container_width=True,
        num_rows="fixed",
        key=f"{state_key}_editor",
    )
    st.session_state[state_key] = edited

    if "Group" not in edited.columns:
        raise ValueError('Missing required column: "Group".')

    row_labels = edited["Group"].astype(str).fillna("").tolist()
    if any(r.strip() == "" for r in row_labels):
        raise ValueError("Row labels (Group) cannot be empty.")

    counts_df = edited.drop(columns=["Group"]).copy()
    counts_df.index = row_labels
    counts_df = counts_df.apply(pd.to_numeric, errors="coerce")
    if counts_df.isna().any().any():
        raise ValueError("All count cells must be numeric.")
    if (counts_df.values < 0).any():
        raise ValueError("Counts must be non-negative.")
    counts_df = counts_df.astype(int)

    # observed with totals (Total must be integer)
    observed = counts_df.copy()
    observed["Total"] = observed.sum(axis=1).astype(int)
    total_row = observed.sum(axis=0).astype(int)
    observed.loc["Total"] = total_row
    observed = observed.reset_index().rename(columns={"index": "Group"})
    return counts_df, observed

# =========================================================
# GLOBAL: 2x2 measures (OR/RR/VE) + Diagnostic metrics + CI
# =========================================================
def _wald_ci_log(r: float, se: float, z: float = 1.96) -> Tuple[float, float]:
    lo = math.exp(math.log(r) - z * se)
    hi = math.exp(math.log(r) + z * se)
    return lo, hi

def calc_or_rr_ve_2x2(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> pd.DataFrame:
    # Exposed row: a b ; Unexposed row: c d
    z = stats.norm.ppf(1 - alpha / 2)

    aa, bb, cc, dd = float(a), float(b), float(c), float(d)
    # Haldane-Anscombe if any zero
    if min(aa, bb, cc, dd) == 0:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5

    OR = (aa * dd) / (bb * cc)
    se_log_or = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    or_lo, or_hi = _wald_ci_log(OR, se_log_or, z)

    risk_e = aa / (aa + bb)
    risk_u = cc / (cc + dd)
    RR = (risk_e / risk_u) if risk_u > 0 else np.nan
    se_log_rr = math.sqrt((1 / aa) - (1 / (aa + bb)) + (1 / cc) - (1 / (cc + dd)))
    rr_lo, rr_hi = _wald_ci_log(RR, se_log_rr, z) if np.isfinite(RR) else ("", "")

    VE = (1 - RR) * 100 if np.isfinite(RR) else np.nan

    out = pd.DataFrame(
        [
            {"Measure": "Odds Ratio (OR)", "Value": OR, "95% CI Lower": or_lo, "95% CI Upper": or_hi},
            {"Measure": "Risk Ratio (RR)", "Value": RR, "95% CI Lower": rr_lo, "95% CI Upper": rr_hi},
            {"Measure": "Effectiveness (VE%)", "Value": VE, "95% CI Lower": "", "95% CI Upper": ""},
        ]
    )
    return out

def _wilson_ci(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return ("", "")
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return max(0, center - half), min(1, center + half)

def calc_diagnostic_2x2(TP: int, FP: int, FN: int, TN: int, alpha: float = 0.05) -> pd.DataFrame:
    sens = TP / (TP + FN) if (TP + FN) else np.nan
    spec = TN / (TN + FP) if (TN + FP) else np.nan
    fpr = FP / (FP + TN) if (FP + TN) else np.nan
    fnr = FN / (FN + TP) if (FN + TP) else np.nan
    ppv = TP / (TP + FP) if (TP + FP) else np.nan
    npv = TN / (TN + FN) if (TN + FN) else np.nan

    lr_pos = sens / (1 - spec) if np.isfinite(sens) and np.isfinite(spec) and (1 - spec) else np.nan
    lr_neg = (1 - sens) / spec if np.isfinite(sens) and np.isfinite(spec) and spec else np.nan

    sens_lo, sens_hi = _wilson_ci(TP, TP + FN, alpha)
    spec_lo, spec_hi = _wilson_ci(TN, TN + FP, alpha)
    fpr_lo, fpr_hi = _wilson_ci(FP, FP + TN, alpha)
    fnr_lo, fnr_hi = _wilson_ci(FN, FN + TP, alpha)
    ppv_lo, ppv_hi = _wilson_ci(TP, TP + FP, alpha)
    npv_lo, npv_hi = _wilson_ci(TN, TN + FN, alpha)

    out = pd.DataFrame(
        [
            {"Measure": "Sensitivity", "Value": sens, "95% CI Lower": sens_lo, "95% CI Upper": sens_hi},
            {"Measure": "Specificity", "Value": spec, "95% CI Lower": spec_lo, "95% CI Upper": spec_hi},
            {"Measure": "False Positive Rate", "Value": fpr, "95% CI Lower": fpr_lo, "95% CI Upper": fpr_hi},
            {"Measure": "False Negative Rate", "Value": fnr, "95% CI Lower": fnr_lo, "95% CI Upper": fnr_hi},
            {"Measure": "PPV", "Value": ppv, "95% CI Lower": ppv_lo, "95% CI Upper": ppv_hi},
            {"Measure": "NPV", "Value": npv, "95% CI Lower": npv_lo, "95% CI Upper": npv_hi},
            {"Measure": "LR+", "Value": lr_pos, "95% CI Lower": "", "95% CI Upper": ""},
            {"Measure": "LR-", "Value": lr_neg, "95% CI Lower": "", "95% CI Upper": ""},
        ]
    )
    return out

# =========================================================
# Categorical tests bundles (SPSS-like output)
# =========================================================
def chi_square_bundle(counts_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    table = counts_df.values.astype(float)
    chi2, p, dof, expected = chi2_contingency(table, correction=False)

    chi_df = pd.DataFrame(
        [
            {
                "Test": "Pearson Chi-Square",
                "Value": float(chi2),
                "df": int(dof),
                "Asymp. Sig. (2-sided)": format_p_value(p),
                "Significant (p<0.05)": yesno_from_p(p),
            }
        ]
    )

    yates_df = None
    if table.shape == (2, 2):
        chi2y, py, dofy, _ = chi2_contingency(table, correction=True)
        yates_df = pd.DataFrame(
            [
                {
                    "Test": "Continuity Correction (Yates)",
                    "Value": float(chi2y),
                    "df": int(dofy),
                    "Asymp. Sig. (2-sided)": format_p_value(py),
                    "Significant (p<0.05)": yesno_from_p(py),
                }
            ]
        )

    exp = pd.DataFrame(expected, index=counts_df.index, columns=counts_df.columns)
    exp["Total"] = exp.sum(axis=1)
    exp.loc["Total"] = exp.sum(axis=0)
    exp = exp.reset_index().rename(columns={"index": "Group"})
    return chi_df, yates_df, exp

def fisher_bundle_2x2(counts_df: pd.DataFrame) -> pd.DataFrame:
    if counts_df.shape != (2, 2):
        raise ValueError("Fisher's Exact Test requires a 2×2 table.")
    or_hat, p = fisher_exact(counts_df.values.astype(int), alternative="two-sided")
    out = pd.DataFrame(
        [
            {
                "Test": "Fisher's Exact Test",
                "Value": float(or_hat),
                "Exact Sig. (2-sided)": format_p_value(p),
                "Significant (p<0.05)": yesno_from_p(p),
            }
        ]
    )
    return out

def gof_bundle(observed: pd.Series, expected: Optional[pd.Series] = None) -> pd.DataFrame:
    obs = pd.to_numeric(observed, errors="coerce")
    if obs.isna().any():
        raise ValueError("Observed counts must be numeric.")
    obs = obs.astype(float).values

    if expected is None:
        stat, p = chisquare(obs)
    else:
        exp = pd.to_numeric(expected, errors="coerce")
        if exp.isna().any():
            raise ValueError("Expected counts must be numeric.")
        stat, p = chisquare(obs, f_exp=exp.astype(float).values)

    out = pd.DataFrame(
        [
            {
                "Test": "Chi-Square Goodness-of-Fit",
                "Value": float(stat),
                "df": int(len(obs) - 1),
                "Asymp. Sig.": format_p_value(p),
                "Significant (p<0.05)": yesno_from_p(p),
            }
        ]
    )
    return out

def mantel_haenszel_from_long(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input long format with columns:
      Stratum, Exposed(0/1), Outcome(0/1), Count
    Returns:
      MH summary table (SPSS-like)
      Stratum 2x2 tables (flattened)
    """
    need = {"Stratum", "Exposed", "Outcome", "Count"}
    if not need.issubset(df_long.columns):
        raise ValueError(f"Missing columns: {sorted(list(need - set(df_long.columns)))}")

    df = df_long.copy()
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
    df["Exposed"] = pd.to_numeric(df["Exposed"], errors="coerce")
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")
    if df[["Count", "Exposed", "Outcome"]].isna().any().any():
        raise ValueError("Exposed/Outcome/Count must be numeric.")
    df["Count"] = df["Count"].astype(int)
    df["Exposed"] = df["Exposed"].astype(int)
    df["Outcome"] = df["Outcome"].astype(int)

    strata = []
    rows = []

    for s, g in df.groupby("Stratum"):
        # Build 2x2:
        # Exposed=1 row0, Exposed=0 row1
        # Outcome=1 col0, Outcome=0 col1
        a = int(g.loc[(g["Exposed"] == 1) & (g["Outcome"] == 1), "Count"].sum())
        b = int(g.loc[(g["Exposed"] == 1) & (g["Outcome"] == 0), "Count"].sum())
        c = int(g.loc[(g["Exposed"] == 0) & (g["Outcome"] == 1), "Count"].sum())
        d = int(g.loc[(g["Exposed"] == 0) & (g["Outcome"] == 0), "Count"].sum())

        table = np.array([[a, b], [c, d]], dtype=int)
        strata.append(table)

        rows.append({"Stratum": s, "a(TP)": a, "b(FP)": b, "c(FN)": c, "d(TN)": d})

    stt = StratifiedTable(strata)
    mh_or = float(stt.oddsratio_pooled)
    mh_ci = stt.oddsratio_pooled_confint()
    mh_test = stt.test_null_odds()
    p = float(mh_test.pvalue)

    summary = pd.DataFrame(
        [
            {
                "Test": "Mantel-Haenszel Common Odds Ratio Estimate",
                "Value": mh_or,
                "95% CI Lower": float(mh_ci[0]),
                "95% CI Upper": float(mh_ci[1]),
                "Asymp. Sig. (2-sided)": format_p_value(p),
                "Significant (p<0.05)": yesno_from_p(p),
            }
        ]
    )

    tables = pd.DataFrame(rows)
    return summary, tables

def cochran_q_from_wide(df_wide: pd.DataFrame, do_posthoc: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Wide format: each column is a treatment/condition, rows=subjects, values in {0,1}
    Cochran's Q test.
    Optional: pairwise McNemar with Bonferroni.
    """
    X = df_wide.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().any().any():
        raise ValueError("All cells must be 0/1.")
    X = X.astype(int)
    if not set(np.unique(X.values)).issubset({0, 1}):
        raise ValueError("Values must be binary (0/1).")

    # Cochran's Q
    # Q = (k-1) * (k*sum(col_sums^2) - T^2) / (k*T - sum(row_sums^2))
    k = X.shape[1]
    col_sums = X.sum(axis=0).values
    row_sums = X.sum(axis=1).values
    T = X.values.sum()
    num = (k - 1) * (k * np.sum(col_sums**2) - T**2)
    den = (k * T - np.sum(row_sums**2))
    Q = num / den if den != 0 else np.nan
    p = 1 - stats.chi2.cdf(Q, df=k - 1) if np.isfinite(Q) else np.nan

    q_table = pd.DataFrame(
        [
            {
                "Test": "Cochran's Q",
                "Value": float(Q) if np.isfinite(Q) else "",
                "df": int(k - 1),
                "Asymp. Sig.": format_p_value(p),
                "Significant (p<0.05)": yesno_from_p(p),
            }
        ]
    )

    posthoc = None
    if do_posthoc and k >= 3:
        pairs = []
        cols = list(X.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = cols[i]
                b = cols[j]
                # 2x2 for paired:
                # both1, a1b0, a0b1, both0
                tab = pd.crosstab(X[a], X[b])
                # ensure 2x2
                for r in [0, 1]:
                    if r not in tab.index:
                        tab.loc[r] = 0
                for c in [0, 1]:
                    if c not in tab.columns:
                        tab[c] = 0
                tab = tab.sort_index().sort_index(axis=1)

                res = mcnemar(tab.values, exact=False, correction=True)
                pairs.append(
                    {"Comparison": f"{a} vs {b}", "McNemar Chi-Square": float(res.statistic), "Asymp. Sig. (2-sided)": float(res.pvalue)}
                )

        ph = pd.DataFrame(pairs)
        # Bonferroni adjustment
        reject, p_adj, _, _ = multipletests(ph["Asymp. Sig. (2-sided)"].values, alpha=0.05, method="bonferroni")
        ph["Asymp. Sig. (2-sided)"] = ph["Asymp. Sig. (2-sided)"].apply(format_p_value)
        ph["Adj. Sig. (Bonferroni)"] = [format_p_value(x) for x in p_adj]
        ph["Significant (p<0.05)"] = ["Yes" if r else "No" for r in reject]
        posthoc = ph

    return q_table, posthoc

# =========================================================
# CI estimation (mean, variance)
# =========================================================
def ci_mean(x: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    se = s / math.sqrt(n)
    lo = m - tcrit * se
    hi = m + tcrit * se
    out = pd.DataFrame([{"Statistic": "Mean", "N": n, "Estimate": m, "95% CI Lower": lo, "95% CI Upper": hi}])
    return out

def ci_variance(x: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    s2 = float(np.var(x, ddof=1))
    df = n - 1
    lo = df * s2 / stats.chi2.ppf(1 - alpha / 2, df=df)
    hi = df * s2 / stats.chi2.ppf(alpha / 2, df=df)
    out = pd.DataFrame([{"Statistic": "Variance", "N": n, "Estimate": s2, "95% CI Lower": lo, "95% CI Upper": hi}])
    return out

# =========================================================
# Sidebar navigation (expanders + buttons)
# =========================================================
st.sidebar.markdown("## Navigation")

def nav_button(label: str, key: str) -> bool:
    return st.sidebar.button(label, key=key, use_container_width=True)

if "nav_main" not in st.session_state:
    st.session_state.nav_main = "Data"
if "nav_sub" not in st.session_state:
    st.session_state.nav_sub = "Upload"

# Main groups
with st.sidebar.expander("Data", expanded=True):
    if nav_button("Upload & Template", "nav_data"):
        st.session_state.nav_main = "Data"
        st.session_state.nav_sub = "Upload"

with st.sidebar.expander("Regression", expanded=False):
    if nav_button("Logistic Regression", "nav_log"):
        st.session_state.nav_main = "Regression"
        st.session_state.nav_sub = "Logistic"
    if nav_button("Linear Regression", "nav_lin"):
        st.session_state.nav_main = "Regression"
        st.session_state.nav_sub = "Linear"

with st.sidebar.expander("Hypothesis Tests", expanded=False):
    if nav_button("t-test", "nav_t"):
        st.session_state.nav_main = "Tests"
        st.session_state.nav_sub = "t-test"
    if nav_button("ANOVA", "nav_a"):
        st.session_state.nav_main = "Tests"
        st.session_state.nav_sub = "ANOVA"
    if nav_button("Categorical Tests", "nav_c"):
        st.session_state.nav_main = "Tests"
        st.session_state.nav_sub = "Categorical"
    if nav_button("Confidence Intervals", "nav_ci"):
        st.session_state.nav_main = "Tests"
        st.session_state.nav_sub = "CI"

main = st.session_state.nav_main
sub = st.session_state.nav_sub

# =========================================================
# Shared dataset in session
# =========================================================
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# =========================================================
# Page: Data
# =========================================================
if main == "Data":
    st.markdown("## Data")

    template_cols = ["Outcome(0/1)", "X1", "X2", "X3"]
    st.download_button(
        "Download Excel template (generic)",
        data=make_excel_template(template_cols, 25),
        file_name="template_generic.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False,
    )

    uploaded = st.file_uploader("Upload dataset (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df = read_dataset(uploaded)
            st.session_state.dataset = df
            show_table(df.head(30), "Data View (first 30 rows)")
            st.success(f"Loaded dataset with {df.shape[0]} rows × {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Load failed: {e}")
    else:
        st.info("Upload a dataset to use Regression modules. Categorical modules can run with built-in tables/templates.")

# =========================================================
# Page: Regression (basic, table outputs SPSS-like)
# =========================================================
if main == "Regression" and sub == "Logistic":
    st.markdown("## Logistic Regression (Basic)")
    df = st.session_state.dataset
    if df is None:
        st.warning("Please upload a dataset in the Data section first.")
    else:
        cols = list(df.columns)
        target = st.selectbox("Target (binary 0/1)", cols, index=0)
        features = st.multiselect("Features", [c for c in cols if c != target], default=[c for c in cols if c != target][:3])

        if st.button("Run Logistic Regression", type="primary", use_container_width=True):
            try:
                data = df[[target] + features].dropna().copy()
                y = pd.to_numeric(data[target], errors="coerce")
                X = data[features].apply(pd.to_numeric, errors="coerce")
                m = pd.concat([y, X], axis=1).dropna()
                y = m[target].astype(int)
                X = m[features]

                X_sm = sm.add_constant(X)
                model = sm.Logit(y, X_sm).fit(disp=False)

                params = model.params
                conf = model.conf_int()
                pvals = model.pvalues

                # SPSS-like table: Variables in the Equation
                rows = []
                for term in params.index:
                    p = float(pvals[term])
                    beta = float(params[term])
                    if term in ["const", "Intercept"]:
                        rows.append({
                            "Term": "Constant",
                            "B": beta,
                            "S.E.": float(model.bse[term]),
                            "Wald": float((beta / model.bse[term])**2) if model.bse[term] != 0 else "",
                            "df": 1,
                            "Sig.": format_p_value(p),
                            "Exp(B)": "",
                            "95% CI Lower": "",
                            "95% CI Upper": "",
                        })
                    else:
                        OR = math.exp(beta)
                        lo = math.exp(float(conf.loc[term, 0]))
                        hi = math.exp(float(conf.loc[term, 1]))
                        rows.append({
                            "Term": term,
                            "B": beta,
                            "S.E.": float(model.bse[term]),
                            "Wald": float((beta / model.bse[term])**2) if model.bse[term] != 0 else "",
                            "df": 1,
                            "Sig.": format_p_value(p),
                            "Exp(B)": OR,
                            "95% CI Lower": lo,
                            "95% CI Upper": hi,
                        })

                out = pd.DataFrame(rows)
                show_table(out, "Variables in the Equation")
                download_table_block(out, "logistic_variables", "Variables in the Equation")

                # ROC (single model)
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                clf = LogisticRegression(max_iter=2000)
                clf.fit(Xtr, ytr)
                proba = clf.predict_proba(Xte)[:, 1]
                fpr, tpr, _ = roc_curve(yte, proba)
                roc_auc = auc(fpr, tpr)

                fig = plt.figure()
                plt.plot(fpr, tpr, label=f"Logistic (AUC = {roc_auc:.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                st.pyplot(fig)
                download_figure_block(fig, "roc_logistic")

            except Exception as e:
                st.error(f"Modeling failed: {e}")

if main == "Regression" and sub == "Linear":
    st.markdown("## Multivariable Linear Regression (Basic)")
    df = st.session_state.dataset
    if df is None:
        st.warning("Please upload a dataset in the Data section first.")
    else:
        cols = list(df.columns)
        y_col = st.selectbox("Outcome (continuous)", cols, index=0)
        x_cols = st.multiselect("Predictors", [c for c in cols if c != y_col], default=[c for c in cols if c != y_col][:3])

        if st.button("Run Linear Regression", type="primary", use_container_width=True):
            try:
                data = df[[y_col] + x_cols].dropna().copy()
                y = pd.to_numeric(data[y_col], errors="coerce")
                X = data[x_cols].apply(pd.to_numeric, errors="coerce")
                m = pd.concat([y, X], axis=1).dropna()
                y = m[y_col]
                X = m[x_cols]

                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm).fit()

                # ANOVA (Type I)
                # Use formula for stable ANOVA naming
                formula = f'Q("{y_col}") ~ ' + " + ".join([f'Q("{c}")' for c in x_cols])
                modf = smf.ols(formula, data=m.rename(columns={y_col: y_col, **{c: c for c in x_cols}})).fit()
                an = anova_lm(modf, typ=1).reset_index().rename(columns={"index": "Source"})
                # Clean Q("x") -> x
                an["Source"] = an["Source"].astype(str).str.replace('Q("', "", regex=False).str.replace('")', "", regex=False)
                an = an.rename(columns={"df": "df", "sum_sq": "Sum Sq", "mean_sq": "Mean Sq", "F": "F", "PR(>F)": "Sig."})
                an["Sig."] = an["Sig."].apply(format_p_value)
                show_table(an, "ANOVA")
                download_table_block(an, "anova_linear", "ANOVA")

                # Coefficients table (SPSS-like)
                params = model.params
                bse = model.bse
                tvals = model.tvalues
                pvals = model.pvalues
                conf = model.conf_int()
                rows = []
                for term in params.index:
                    name = "Intercept" if term in ["const", "Intercept"] else term
                    p = float(pvals[term])
                    rows.append(
                        {
                            "Term": name,
                            "B": float(params[term]),
                            "Std. Error": float(bse[term]),
                            "t": float(tvals[term]),
                            "Sig.": format_p_value(p),
                            "95% CI Lower": float(conf.loc[term, 0]),
                            "95% CI Upper": float(conf.loc[term, 1]),
                            "Significant (p<0.05)": yesno_from_p(p),
                        }
                    )
                coef = pd.DataFrame(rows)
                show_table(coef, "Coefficients")
                download_table_block(coef, "coefficients_linear", "Coefficients")

                # Model summary
                summ = pd.DataFrame([{
                    "R": math.sqrt(max(0, float(model.rsquared))),
                    "R Square": float(model.rsquared),
                    "Adj. R Square": float(model.rsquared_adj),
                    "Std. Error of the Estimate": float(math.sqrt(model.mse_resid)),
                    "N": int(model.nobs),
                }])
                show_table(summ, "Model Summary")
                download_table_block(summ, "model_summary_linear", "Model Summary")

            except Exception as e:
                st.error(f"Modeling failed: {e}")

# =========================================================
# Page: Tests - Categorical
# =========================================================
if main == "Tests" and sub == "Categorical":
    st.markdown("## Categorical Tests")

    cat_choice = st.selectbox(
        "Choose a categorical procedure",
        [
            "Contingency Table (r×c) / Chi-square",
            "Fisher's Exact (2×2)",
            "Goodness-of-fit",
            "Mantel–Haenszel (stratified 2×2)",
            "Cochran's Q (+ McNemar post-hoc)",
            "2×2 Risk & Diagnostic Metrics (OR/RR/VE + Sens/Spec, etc.)",
        ],
    )

    # 1) Chi-square
    if cat_choice == "Contingency Table (r×c) / Chi-square":
        counts_df, observed_df = contingency_editor(
            key="chisq",
            default_rows=["Treatment", "Placebo"],
            default_cols=["Disease", "No disease"],
            default_counts=np.array([[10, 30], [20, 15]], dtype=int),
        )

        if st.button("Run Chi-square", type="primary", use_container_width=True):
            try:
                chi_df, yates_df, exp_df = chi_square_bundle(counts_df)

                show_table(observed_df, "Crosstabulation (Observed Counts)")
                download_table_block(observed_df, "crosstab_observed", "Observed Counts")

                show_table(chi_df, "Chi-Square Tests")
                download_table_block(chi_df, "chi_square_tests", "Chi-Square Tests")

                if yates_df is not None:
                    show_table(yates_df, "Chi-Square Tests (Continuity Correction)")
                    download_table_block(yates_df, "chi_square_yates", "Continuity Correction")

                show_table(exp_df, "Expected Counts")
                download_table_block(exp_df, "expected_counts", "Expected Counts")

            except Exception as e:
                st.error(f"Failed: {e}")

    # 2) Fisher
    elif cat_choice == "Fisher's Exact (2×2)":
        counts_df, observed_df = contingency_editor(
            key="fisher",
            default_rows=["Group 1", "Group 2"],
            default_cols=["Outcome+", "Outcome-"],
            default_counts=np.array([[8, 12], [3, 17]], dtype=int),
        )
        if st.button("Run Fisher's Exact", type="primary", use_container_width=True):
            try:
                show_table(observed_df, "Crosstabulation (Observed Counts)")
                download_table_block(observed_df, "fisher_observed", "Observed Counts")

                res = fisher_bundle_2x2(counts_df)
                show_table(res, "Fisher's Exact Test")
                download_table_block(res, "fisher_exact", "Fisher's Exact Test")

            except Exception as e:
                st.error(f"Failed: {e}")

    # 3) GOF
    elif cat_choice == "Goodness-of-fit":
        st.markdown("### Input counts")
        st.caption("Example: observed frequencies across categories. Expected can be blank (uniform).")

        if "gof_df" not in st.session_state:
            st.session_state.gof_df = pd.DataFrame({"Category": ["A", "B", "C"], "Observed": [10, 12, 8], "Expected(optional)": ["", "", ""]})

        edited = st.data_editor(st.session_state.gof_df, use_container_width=True, num_rows="dynamic")
        st.session_state.gof_df = edited

        if st.button("Run Goodness-of-fit", type="primary", use_container_width=True):
            try:
                df0 = edited.copy()
                obs = df0["Observed"]
                exp = df0["Expected(optional)"]
                exp_use = exp if exp.astype(str).str.strip().replace("", np.nan).notna().any() else None
                out = gof_bundle(obs, exp_use)
                show_table(out, "Chi-Square Goodness-of-Fit Test")
                download_table_block(out, "gof_chi_square", "Goodness-of-Fit")
            except Exception as e:
                st.error(f"Failed: {e}")

    # 4) Mantel–Haenszel
    elif cat_choice == "Mantel–Haenszel (stratified 2×2)":
        st.markdown("### Long format input")
        st.caption('Required columns: Stratum, Exposed(0/1), Outcome(0/1), Count. (This is robust and avoids row/col renaming.)')

        st.download_button(
            "Download Excel template",
            data=make_excel_template(["Stratum", "Exposed", "Outcome", "Count"], 30),
            file_name="template_mantel_haenszel.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
        )

        upl = st.file_uploader("Upload stratified 2×2 table (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="mh_upl")
        if upl is not None:
            try:
                df_long = read_dataset(upl)
                show_table(df_long.head(50), "Data Preview")
                if st.button("Run Mantel–Haenszel", type="primary", use_container_width=True):
                    summary, tables = mantel_haenszel_from_long(df_long)
                    show_table(tables, "Stratum Tables (a,b,c,d)")
                    download_table_block(tables, "mh_strata_tables", "Stratum Tables")
                    show_table(summary, "Mantel-Haenszel Test")
                    download_table_block(summary, "mantel_haenszel", "Mantel-Haenszel")
            except Exception as e:
                st.error(f"Failed: {e}")

    # 5) Cochran's Q
    elif cat_choice == "Cochran's Q (+ McNemar post-hoc)":
        st.markdown("### Wide format input")
        st.caption("Each column = condition/treatment. Rows = subjects. Values must be 0/1.")

        st.download_button(
            "Download Excel template",
            data=make_excel_template(["Cond_A", "Cond_B", "Cond_C"], 40),
            file_name="template_cochran_q.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
        )

        upl = st.file_uploader("Upload wide binary data (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="cq_upl")
        do_ph = st.checkbox("Post-hoc McNemar pairwise + Bonferroni", value=True)

        if upl is not None:
            try:
                dfw = read_dataset(upl)
                show_table(dfw.head(40), "Data Preview")
                if st.button("Run Cochran's Q", type="primary", use_container_width=True):
                    q_table, posthoc = cochran_q_from_wide(dfw, do_posthoc=do_ph)
                    show_table(q_table, "Cochran's Q Test")
                    download_table_block(q_table, "cochran_q", "Cochran's Q")

                    if posthoc is not None:
                        show_table(posthoc, "Post-hoc McNemar Pairwise (Bonferroni)")
                        download_table_block(posthoc, "mcnemar_posthoc", "McNemar Post-hoc")
            except Exception as e:
                st.error(f"Failed: {e}")

    # 6) 2x2 Risk & Diagnostic metrics
    elif cat_choice == "2×2 Risk & Diagnostic Metrics (OR/RR/VE + Sens/Spec, etc.)":
        st.markdown("### 2×2 input (counts)")
        st.caption("Interpretation template: Test+/Test- vs Disease+/Disease- OR Exposure/Outcome. The app computes OR/RR/VE and diagnostic metrics.")

        counts_df, observed_df = contingency_editor(
            key="risk2x2",
            default_rows=["Test+", "Test-"],
            default_cols=["Disease+", "Disease-"],
            default_counts=np.array([[50, 10], [8, 80]], dtype=int),
        )

        if st.button("Compute 2×2 measures", type="primary", use_container_width=True):
            try:
                show_table(observed_df, "Crosstabulation (Observed Counts)")
                download_table_block(observed_df, "risk_observed", "Observed Counts")

                if counts_df.shape != (2, 2):
                    raise ValueError("This module requires a 2×2 table.")

                a, b = int(counts_df.iloc[0, 0]), int(counts_df.iloc[0, 1])
                c, d = int(counts_df.iloc[1, 0]), int(counts_df.iloc[1, 1])

                risk = calc_or_rr_ve_2x2(a, b, c, d)
                show_table(risk, "Risk Estimate (OR / RR / VE)")
                download_table_block(risk, "risk_estimate", "Risk Estimate")

                diag = calc_diagnostic_2x2(TP=a, FP=b, FN=c, TN=d)
                show_table(diag, "Diagnostic Accuracy (with 95% CI)")
                download_table_block(diag, "diagnostic_accuracy", "Diagnostic Accuracy")

            except Exception as e:
                st.error(f"Failed: {e}")

# =========================================================
# Page: Tests - CI
# =========================================================
if main == "Tests" and sub == "CI":
    st.markdown("## Confidence Intervals")

    st.caption("Upload a numeric column or paste values; outputs are SPSS-like tables with export.")

    mode = st.radio("Input method", ["Upload dataset column", "Paste values"], horizontal=True)

    x = None
    if mode == "Upload dataset column":
        df = st.session_state.dataset
        if df is None:
            st.warning("Upload a dataset in Data first.")
        else:
            col = st.selectbox("Choose numeric column", df.columns)
            x = pd.to_numeric(df[col], errors="coerce").dropna().values
    else:
        txt = st.text_area("Paste numbers (comma/space/newline separated)", height=120, value="1.2, 2.5, 3.0, 2.8, 1.9")
        parts = [p.strip() for p in txt.replace("\n", ",").replace(" ", ",").split(",") if p.strip() != ""]
        try:
            x = np.array([float(p) for p in parts], dtype=float)
        except Exception:
            x = None

    if x is not None and len(x) >= 2:
        if st.button("Compute CI", type="primary", use_container_width=True):
            try:
                t1 = ci_mean(x)
                t2 = ci_variance(x)
                show_table(t1, "Confidence Interval for Mean")
                download_table_block(t1, "ci_mean", "CI Mean")

                show_table(t2, "Confidence Interval for Variance")
                download_table_block(t2, "ci_variance", "CI Variance")
            except Exception as e:
                st.error(f"Failed: {e}")
    else:
        st.info("Provide at least 2 numeric values.")

# =========================================================
# Page: Tests - t-test (minimal robust)
# =========================================================
if main == "Tests" and sub == "t-test":
    st.markdown("## t-test (Auto normality check → parametric / nonparametric)")
    st.caption("This module is a robust baseline. You can expand to full SPSS-style outputs later.")

    t_type = st.selectbox("Choose t-test type", ["One-sample", "Independent samples", "Paired samples"])

    df = st.session_state.dataset
    if df is None:
        st.warning("Upload a dataset in Data first (or extend this module to accept manual input).")
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
        if len(num_cols) == 0:
            st.warning("No numeric columns found.")
        else:
            def shapiro_p(xv):
                xv = pd.to_numeric(xv, errors="coerce").dropna().values
                if len(xv) < 3 or len(xv) > 5000:
                    return np.nan
                return float(stats.shapiro(xv).pvalue)

            if t_type == "One-sample":
                col = st.selectbox("Variable", num_cols)
                mu0 = st.number_input("Test value (mean0)", value=0.0)
                x = pd.to_numeric(df[col], errors="coerce").dropna().values
                if st.button("Run test", type="primary", use_container_width=True):
                    p_norm = shapiro_p(x)
                    if np.isfinite(p_norm) and p_norm < 0.05:
                        # nonparametric
                        stat, p = stats.wilcoxon(x - mu0)
                        out = pd.DataFrame([{
                            "Test": "Wilcoxon Signed-Rank (nonparametric)",
                            "N": int(len(x)),
                            "Statistic": float(stat),
                            "Sig. (2-sided)": format_p_value(p),
                            "Significant (p<0.05)": yesno_from_p(p),
                        }])
                    else:
                        stat, p = stats.ttest_1samp(x, mu0)
                        out = pd.DataFrame([{
                            "Test": "One-Sample t Test",
                            "N": int(len(x)),
                            "t": float(stat),
                            "df": int(len(x)-1),
                            "Sig. (2-sided)": format_p_value(p),
                            "Significant (p<0.05)": yesno_from_p(p),
                        }])
                    show_table(out, "Test Results")
                    download_table_block(out, "ttest_one_sample", "One-sample test")

            elif t_type == "Independent samples":
                group_col = st.selectbox("Group column (categorical)", df.columns)
                value_col = st.selectbox("Value column (numeric)", num_cols)
                g = df[[group_col, value_col]].dropna()
                groups = sorted(g[group_col].astype(str).unique())
                if len(groups) < 2:
                    st.warning("Need at least 2 groups.")
                else:
                    g1 = st.selectbox("Group 1", groups, index=0)
                    g2 = st.selectbox("Group 2", groups, index=1 if len(groups) > 1 else 0)
                    x1 = pd.to_numeric(g.loc[g[group_col].astype(str) == g1, value_col], errors="coerce").dropna().values
                    x2 = pd.to_numeric(g.loc[g[group_col].astype(str) == g2, value_col], errors="coerce").dropna().values
                    if st.button("Run test", type="primary", use_container_width=True):
                        # Normality check (rough)
                        p1 = shapiro_p(x1)
                        p2 = shapiro_p(x2)
                        if (np.isfinite(p1) and p1 < 0.05) or (np.isfinite(p2) and p2 < 0.05):
                            # nonparametric
                            stat, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
                            out = pd.DataFrame([{
                                "Test": "Mann–Whitney U (nonparametric)",
                                "Group1": g1, "Group2": g2,
                                "U": float(stat),
                                "Sig. (2-sided)": format_p_value(p),
                                "Significant (p<0.05)": yesno_from_p(p),
                            }])
                        else:
                            # Levene for equal variances
                            lev_stat, lev_p = stats.levene(x1, x2)
                            equal_var = (lev_p >= 0.05)
                            stat, p = stats.ttest_ind(x1, x2, equal_var=equal_var)
                            out = pd.DataFrame([{
                                "Test": "Independent-Samples t Test",
                                "Levene Sig.": format_p_value(lev_p),
                                "Equal variances assumed": "Yes" if equal_var else "No",
                                "t": float(stat),
                                "df": int(len(x1)+len(x2)-2) if equal_var else "",
                                "Sig. (2-sided)": format_p_value(p),
                                "Significant (p<0.05)": yesno_from_p(p),
                            }])
                        show_table(out, "Test Results")
                        download_table_block(out, "ttest_independent", "Independent samples test")

            else:
                col1 = st.selectbox("Variable 1", num_cols, index=0)
                col2 = st.selectbox("Variable 2", num_cols, index=1 if len(num_cols) > 1 else 0)
                z = df[[col1, col2]].dropna()
                x1 = pd.to_numeric(z[col1], errors="coerce").dropna().values
                x2 = pd.to_numeric(z[col2], errors="coerce").dropna().values
                if st.button("Run test", type="primary", use_container_width=True):
                    diff = x1 - x2
                    p_norm = shapiro_p(diff)
                    if np.isfinite(p_norm) and p_norm < 0.05:
                        stat, p = stats.wilcoxon(diff)
                        out = pd.DataFrame([{
                            "Test": "Wilcoxon Signed-Rank (paired, nonparametric)",
                            "N": int(len(diff)),
                            "Statistic": float(stat),
                            "Sig. (2-sided)": format_p_value(p),
                            "Significant (p<0.05)": yesno_from_p(p),
                        }])
                    else:
                        stat, p = stats.ttest_rel(x1, x2)
                        out = pd.DataFrame([{
                            "Test": "Paired-Samples t Test",
                            "N": int(len(diff)),
                            "t": float(stat),
                            "df": int(len(diff)-1),
                            "Sig. (2-sided)": format_p_value(p),
                            "Significant (p<0.05)": yesno_from_p(p),
                        }])
                    show_table(out, "Test Results")
                    download_table_block(out, "ttest_paired", "Paired samples test")

# =========================================================
# Page: Tests - ANOVA (baseline)
# =========================================================
if main == "Tests" and sub == "ANOVA":
    st.markdown("## ANOVA (baseline)")
    st.caption("Baseline one-way ANOVA. You can extend to repeated measures / two-way later.")

    df = st.session_state.dataset
    if df is None:
        st.warning("Upload a dataset in Data first.")
    else:
        cols = list(df.columns)
        group_col = st.selectbox("Factor (categorical)", cols)
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
        val_col = st.selectbox("Outcome (numeric)", num_cols if num_cols else cols)

        if st.button("Run one-way ANOVA", type="primary", use_container_width=True):
            try:
                d = df[[group_col, val_col]].dropna().copy()
                d[group_col] = d[group_col].astype(str)
                d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
                d = d.dropna()

                # OLS with formula
                formula = f'Q("{val_col}") ~ C(Q("{group_col}"))'
                mod = smf.ols(formula, data=d.rename(columns={group_col: group_col, val_col: val_col})).fit()
                an = anova_lm(mod, typ=1).reset_index().rename(columns={"index": "Source"})
                an = an.rename(columns={"df": "df", "sum_sq": "Sum Sq", "mean_sq": "Mean Sq", "F": "F", "PR(>F)": "Sig."})
                an["Sig."] = an["Sig."].apply(format_p_value)
                show_table(an, "ANOVA")
                download_table_block(an, "anova_oneway", "ANOVA")

            except Exception as e:
                st.error(f"Failed: {e}")
