import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy import stats
from scipy.stats import shapiro
from scipy.stats import chi2_contingency, chisquare

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import (
    mcnemar, StratifiedTable
)
from statsmodels.stats.contingency_tables import cochrans_q as sm_cochrans_q
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Data Analysis in Health Sciences",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CSS (stable + no clipping + blue sidebar + nice expanders)
# =========================================================
st.markdown("""
<style>
/* Reduce top empty space */
.block-container { padding-top: 1.0rem !important; }

/* Sidebar background */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0B3A66 0%, #0A2D4E 100%);
}

/* Sidebar text safe */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span{
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
  font-weight: 650 !important;
  padding: 10px 12px !important;
}
section[data-testid="stSidebar"] .stButton button:hover,
section[data-testid="stSidebar"] .stDownloadButton button:hover{
  background: rgba(255,255,255,0.18) !important;
  border-color: rgba(255,255,255,0.55) !important;
  transform: translateY(-1px);
}

/* Expander header styled (fix white block) */
section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary{
  background: rgba(255,255,255,0.10) !important;
  border: 1px solid rgba(255,255,255,0.30) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  color: #ffffff !important;
  font-weight: 800 !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] details[open] summary{
  background: rgba(255,255,255,0.16) !important;
  border-color: rgba(255,255,255,0.50) !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary svg{
  fill: #ffffff !important;
}

/* File uploader: remove white card feeling */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{
  background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
  border: 1px dashed rgba(255,255,255,0.35) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div > div{
  background: rgba(255,255,255,0.10) !important;
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
  font-weight: 650 !important;
}

/* Header */
.app-header {
  background: linear-gradient(90deg, #0B3A66, #0A2D4E);
  padding: 16px 28px;
  margin: 0 0 1.0rem 0;
  border-radius: 18px;
}
.app-header h1 {
  color: #ffffff;
  font-size: 1.55rem;
  margin: 0;
  font-weight: 850;
}
.app-header p {
  color: rgba(255,255,255,0.85);
  margin: 6px 0 0 0;
  font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>Data Analysis in Health Sciences</h1>
        <p>Regression modeling + hypothesis tests + diagnostics + reporting (SPSS-like tables)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Session state
# =========================================================
if "main_menu" not in st.session_state:
    st.session_state.main_menu = "Logistic Regression"
if "sub_menu" not in st.session_state:
    st.session_state.sub_menu = "Data"
if "df" not in st.session_state:
    st.session_state.df = None
if "last_results" not in st.session_state:
    st.session_state.last_results = {}

# =========================================================
# Utilities
# =========================================================
def df_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, _df in sheets.items():
            _df.to_excel(writer, index=False, sheet_name=str(sheet)[:31])
    bio.seek(0)
    return bio.getvalue()

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 200) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
    bio.seek(0)
    return bio.getvalue()

def df_to_png_bytes(df: pd.DataFrame, title: str = "", dpi: int = 200) -> bytes:
    df2 = df.copy().replace([np.nan, None, "nan", "NaN"], "")
    n_rows, n_cols = df2.shape
    fig_w = min(18, max(6, n_cols * 1.5))
    fig_h = min(18, max(2.2, (n_rows + 1) * 0.45))
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
        mime="image/png"
    )

def read_uploaded_file(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None
    st.error("Unsupported file type. Please upload CSV or Excel.")
    return None

def show_dataset_status():
    if st.session_state.df is None:
        st.info("No dataset loaded yet. Upload a CSV/XLSX in the Data section.")
        return
    df = st.session_state.df
    st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    with st.expander("Preview (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

def format_p_value(p) -> str:
    try:
        p = float(p)
    except Exception:
        return ""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"

def yesno_from_p(p) -> str:
    try:
        p = float(p)
    except Exception:
        return ""
    if np.isnan(p):
        return ""
    return "Yes" if p < 0.05 else "No"

def show_table(df: pd.DataFrame, title: str, center_all: bool = True):
    st.markdown(f"### {title}")
    df2 = df.copy().replace([np.nan, None, "nan", "NaN"], "")
    if center_all:
        styled = df2.style.set_properties(**{"text-align": "center"})
    else:
        num_cols = df2.select_dtypes(include=["number"]).columns
        styled = df2.style.set_properties(subset=num_cols, **{"text-align": "center"})
    st.dataframe(styled, use_container_width=True)

def q(name: str) -> str:
    return f'Q("{name}")'

# =========================================================
# Templates
# =========================================================
def make_logistic_template() -> pd.DataFrame:
    return pd.DataFrame({
        "Outcome_binary": [0, 1, 0],
        "X1": [None, None, None],
        "X2": [None, None, None],
        "X3": [None, None, None],
    })

def make_linear_template() -> pd.DataFrame:
    return pd.DataFrame({
        "Outcome_y": [None, None, None],
        "X1": [None, None, None],
        "X2": [None, None, None],
        "X3": [None, None, None],
    })

def make_counts_template(r: int, c: int) -> pd.DataFrame:
    df = pd.DataFrame(np.zeros((r, c), dtype=int))
    df.columns = [f"Col{i+1}" for i in range(c)]
    df.index = [f"Row{i+1}" for i in range(r)]
    df = df.reset_index().rename(columns={"index": "Row"})
    return df

def make_gof_template() -> pd.DataFrame:
    return pd.DataFrame({
        "Category": ["A", "B", "C"],
        "Observed": [None, None, None],
        "Expected proportion (optional)": [None, None, None]
    })

def make_mh_long_template() -> pd.DataFrame:
    return pd.DataFrame({
        "Stratum": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
        "Exposure": ["E1", "E1", "E2", "E2", "E1", "E1", "E2", "E2"],
        "Outcome":  [1, 0, 1, 0, 1, 0, 1, 0]
    })

def make_cochran_template() -> pd.DataFrame:
    return pd.DataFrame({
        "Subject": [1, 2, 3],
        "Cond_A": [0, 1, 1],
        "Cond_B": [0, 1, 0],
        "Cond_C": [1, 1, 0],
    })

def make_diag2x2_template() -> pd.DataFrame:
    # Confusion matrix style input
    return pd.DataFrame({
        "": ["Actual Positive", "Actual Negative"],
        "Test Positive": [None, None],
        "Test Negative": [None, None],
    })

def make_ci_template() -> pd.DataFrame:
    return pd.DataFrame({"Value": [None, None, None]})

# =========================================================
# Logistic regression (OR table + model compare)
# =========================================================
def run_logistic_or_table(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    data = df[[target] + features].dropna().copy()

    y_raw = data[target]
    if y_raw.nunique() != 2:
        raise ValueError("Binary logistic regression requires exactly 2 unique target values.")

    if set(pd.unique(y_raw)) != {0, 1}:
        classes = list(pd.unique(y_raw))
        mapping = {classes[0]: 0, classes[1]: 1}
        y = y_raw.map(mapping).astype(int)
    else:
        y = y_raw.astype(int)

    X = data[features].apply(pd.to_numeric, errors="coerce")
    tmp = pd.concat([y, X], axis=1).dropna()
    y = tmp[target]
    X = tmp[features]

    X_sm = sm.add_constant(X)
    logit = sm.Logit(y, X_sm).fit(disp=False)

    params = logit.params
    conf = logit.conf_int()
    pvals = logit.pvalues

    odds = pd.DataFrame({
        "Term": params.index,
        "B": params.values,
        "Exp(B)": np.exp(params.values),
        "95% CI for Exp(B) Lower": np.exp(conf[0].values),
        "95% CI for Exp(B) Upper": np.exp(conf[1].values),
        "p_raw": pvals.values,
    })

    odds["p-value"] = odds["p_raw"].apply(format_p_value)
    odds["Significant (p<0.05)"] = odds["p_raw"].apply(lambda p: "Yes" if float(p) < 0.05 else "No")
    odds = odds.drop(columns=["p_raw"])

    # SPSS-like: hide Exp(B), CI for constant
    is_const = odds["Term"].isin(["const", "Intercept"])
    odds.loc[is_const, ["Exp(B)", "95% CI for Exp(B) Lower", "95% CI for Exp(B) Upper"]] = np.nan

    for col in ["B", "Exp(B)", "95% CI for Exp(B) Lower", "95% CI for Exp(B) Upper"]:
        odds[col] = pd.to_numeric(odds[col], errors="coerce").round(4)

    return odds[
        ["Term", "B", "Exp(B)", "95% CI for Exp(B) Lower", "95% CI for Exp(B) Upper", "p-value", "Significant (p<0.05)"]
    ]

def compare_classification_models(df: pd.DataFrame, target: str, features: list[str],
                                 test_size: float = 0.25, random_state: int = 42):
    data = df[[target] + features].dropna().copy()

    y_raw = data[target]
    if y_raw.nunique() != 2:
        raise ValueError("Model comparison requires a binary target with 2 unique values.")

    if set(pd.unique(y_raw)) != {0, 1}:
        classes = list(pd.unique(y_raw))
        mapping = {classes[0]: 0, classes[1]: 1}
        y = y_raw.map(mapping).astype(int)
    else:
        y = y_raw.astype(int)

    X = data[features].apply(pd.to_numeric, errors="coerce")
    tmp = pd.concat([y, X], axis=1).dropna()
    y = tmp[target].astype(int)
    X = tmp[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", SkLogisticRegression(max_iter=2000))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=random_state))]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
    }

    rows = []
    roc_lines = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            score = model.decision_function(X_test)
            prob = (score - score.min()) / (score.max() - score.min() + 1e-12)

        pred = (prob >= 0.5).astype(int)

        auc_val = roc_auc_score(y_test, prob)
        rows.append({
            "Model": name,
            "AUC": round(float(auc_val), 4),
            "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "Precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "F1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        })

        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_lines[name] = (fpr, tpr, float(auc_val))

    metrics_df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    top_name = metrics_df.loc[0, "Model"]
    top_model = models[top_name]
    if hasattr(top_model, "predict_proba"):
        top_prob = top_model.predict_proba(X_test)[:, 1]
    else:
        score = top_model.decision_function(X_test)
        top_prob = (score - score.min()) / (score.max() - score.min() + 1e-12)
    top_pred = (top_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, top_pred)

    return {"metrics": metrics_df, "roc_lines": roc_lines, "cm_top": cm, "top_model_name": top_name}

# =========================================================
# Linear regression (OLS via formula; SPSS-like ANOVA/coef)
# =========================================================
def _clean_term(s: str) -> str:
    if isinstance(s, str) and s.startswith('Q("') and s.endswith('")'):
        return s[3:-2]
    if isinstance(s, str) and s.startswith('C(Q("') and s.endswith('"))'):
        return s[5:-3]
    return s

def run_linear_regression(df: pd.DataFrame, target: str, features: list[str]):
    data = df[[target] + features].dropna().copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    for c in features:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()

    if data.shape[0] < max(10, len(features) + 3):
        raise ValueError("Not enough rows after cleaning.")

    formula = f'{q(target)} ~ ' + " + ".join(q(c) for c in features)
    model = smf.ols(formula=formula, data=data).fit()

    conf = model.conf_int()
    coef_df = pd.DataFrame({
        "Term": model.params.index,
        "B": model.params.values,
        "Std. Error": model.bse.values,
        "t": model.tvalues.values,
        "p_raw": model.pvalues.values,
        "95% CI Lower": conf[0].values,
        "95% CI Upper": conf[1].values,
    })
    coef_df["Term"] = coef_df["Term"].apply(_clean_term)
    coef_df["p-value"] = coef_df["p_raw"].apply(format_p_value)
    coef_df["Significant (p<0.05)"] = coef_df["p_raw"].apply(lambda p: "Yes" if float(p) < 0.05 else "No")
    coef_df = coef_df.drop(columns=["p_raw"])
    for col in ["B", "Std. Error", "t", "95% CI Lower", "95% CI Upper"]:
        coef_df[col] = pd.to_numeric(coef_df[col], errors="coerce").round(4)

    anova_df = anova_lm(model, typ=1).reset_index().rename(columns={"index": "Source"})
    anova_df["Source"] = anova_df["Source"].apply(_clean_term)
    anova_df = anova_df.rename(columns={"sum_sq": "Sum Sq", "mean_sq": "Mean Sq", "PR(>F)": "p_raw"})
    if "p_raw" in anova_df.columns:
        anova_df["p-value"] = anova_df["p_raw"].apply(format_p_value)
        anova_df = anova_df.drop(columns=["p_raw"])
    for col in ["Sum Sq", "Mean Sq", "F"]:
        if col in anova_df.columns:
            anova_df[col] = pd.to_numeric(anova_df[col], errors="coerce").round(4)

    # IMPORTANT: display blanks instead of nan/None everywhere
    anova_df = anova_df.replace([np.nan, None, "nan", "NaN"], "")

    metrics = pd.DataFrame([{
        "N": int(model.nobs),
        "R-squared": round(float(model.rsquared), 4),
        "Adj. R-squared": round(float(model.rsquared_adj), 4),
        "AIC": round(float(model.aic), 4),
        "BIC": round(float(model.bic), 4),
        "F": round(float(model.fvalue), 4) if model.fvalue is not None else "",
        "Sig.": format_p_value(float(model.f_pvalue)) if model.f_pvalue is not None else "",
    }])

    resid = model.resid
    fitted = model.fittedvalues

    # VIF
    X = data[features]
    X_sm = sm.add_constant(X)
    vif_rows = []
    for i, col in enumerate(X_sm.columns):
        if col == "const":
            continue
        vif_rows.append({"Variable": col, "VIF": round(float(variance_inflation_factor(X_sm.values, i)), 4)})
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)

    # Residual diagnostics
    s_stat, s_p = shapiro(resid) if len(resid) >= 3 else (np.nan, np.nan)
    shapiro_df = pd.DataFrame([{"Test": "Shapiro-Wilk", "Statistic": round(float(s_stat), 4) if not np.isnan(s_stat) else "",
                               "Sig.": format_p_value(float(s_p)) if not np.isnan(s_p) else ""}])

    lm, lm_p, fval, f_p = het_breuschpagan(resid, sm.add_constant(X))
    bp_df = pd.DataFrame([{
        "Test": "Breusch-Pagan",
        "LM": round(float(lm), 4),
        "Sig. (LM)": format_p_value(float(lm_p)),
        "F": round(float(fval), 4),
        "Sig. (F)": format_p_value(float(f_p)),
    }])

    dw_df = pd.DataFrame([{"Test": "Durbin-Watson", "Statistic": round(float(durbin_watson(resid)), 4)}])

    return {
        "model": model,
        "coef_table": coef_df,
        "anova_table": anova_df,
        "metrics": metrics,
        "resid": resid,
        "fitted": fitted,
        "vif": vif_df,
        "shapiro": shapiro_df,
        "bp": bp_df,
        "dw": dw_df
    }

# =========================================================
# Categorical metrics (OR/RR/VE + Diagnostic accuracy 2x2)
# =========================================================
def _cc_if_zero(a, b, c, d):
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) == 0:
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return a, b, c, d

def _log_ci(est, se):
    lo = math.exp(math.log(est) - 1.96 * se)
    hi = math.exp(math.log(est) + 1.96 * se)
    return lo, hi

def calc_or_rr_ve_2x2(a, b, c, d):
    a, b, c, d = _cc_if_zero(a, b, c, d)

    OR = (a * d) / (b * c)
    se_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
    lo_or, hi_or = _log_ci(OR, se_or)

    rr = (a/(a+b)) / (c/(c+d))
    se_rr = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    lo_rr, hi_rr = _log_ci(rr, se_rr)

    ve = (1 - rr) * 100
    ve_lo = (1 - hi_rr) * 100
    ve_hi = (1 - lo_rr) * 100

    out = pd.DataFrame([
        {"Measure": "Odds Ratio (OR)", "Estimate": round(OR, 4), "95% CI Lower": round(lo_or, 4), "95% CI Upper": round(hi_or, 4)},
        {"Measure": "Risk Ratio (RR)", "Estimate": round(rr, 4), "95% CI Lower": round(lo_rr, 4), "95% CI Upper": round(hi_rr, 4)},
        {"Measure": "Vaccine Effectiveness (VE, %)", "Estimate": round(ve, 2), "95% CI Lower": round(ve_lo, 2), "95% CI Upper": round(ve_hi, 2)},
    ])
    return out

def wilson_ci(x, n, alpha=0.05):
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha/2)
    phat = x / n
    denom = 1 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    return (max(0.0, center-half), min(1.0, center+half))

def diagnostic_accuracy_2x2(tp, fp, fn, tn):
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    # rates
    sens = tp / (tp + fn) if (tp+fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn+fp) > 0 else np.nan
    fpr = 1 - spec if not np.isnan(spec) else np.nan
    fnr = 1 - sens if not np.isnan(sens) else np.nan
    ppv = tp / (tp + fp) if (tp+fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn+fn) > 0 else np.nan

    # LR+ LR-
    lr_pos = sens / (1 - spec) if (not np.isnan(sens) and not np.isnan(spec) and (1-spec) > 0) else np.nan
    lr_neg = (1 - sens) / spec if (not np.isnan(sens) and not np.isnan(spec) and spec > 0) else np.nan

    # Wilson CI for proportions
    sens_ci = wilson_ci(tp, tp+fn) if (tp+fn) > 0 else (np.nan, np.nan)
    spec_ci = wilson_ci(tn, tn+fp) if (tn+fp) > 0 else (np.nan, np.nan)
    ppv_ci  = wilson_ci(tp, tp+fp) if (tp+fp) > 0 else (np.nan, np.nan)
    npv_ci  = wilson_ci(tn, tn+fn) if (tn+fn) > 0 else (np.nan, np.nan)

    def fmt_ci(lo, hi, pct=True):
        if np.isnan(lo) or np.isnan(hi):
            return ""
        if pct:
            return f"{lo*100:.1f}–{hi*100:.1f}"
        return f"{lo:.4f}–{hi:.4f}"

    out = pd.DataFrame([
        {"Measure": "Sensitivity", "Estimate": "" if np.isnan(sens) else round(sens*100, 1), "95% CI": fmt_ci(*sens_ci, pct=True)},
        {"Measure": "Specificity", "Estimate": "" if np.isnan(spec) else round(spec*100, 1), "95% CI": fmt_ci(*spec_ci, pct=True)},
        {"Measure": "False Positive Rate", "Estimate": "" if np.isnan(fpr) else round(fpr*100, 1), "95% CI": ""},
        {"Measure": "False Negative Rate", "Estimate": "" if np.isnan(fnr) else round(fnr*100, 1), "95% CI": ""},
        {"Measure": "PPV", "Estimate": "" if np.isnan(ppv) else round(ppv*100, 1), "95% CI": fmt_ci(*ppv_ci, pct=True)},
        {"Measure": "NPV", "Estimate": "" if np.isnan(npv) else round(npv*100, 1), "95% CI": fmt_ci(*npv_ci, pct=True)},
        {"Measure": "LR+", "Estimate": "" if np.isnan(lr_pos) else round(lr_pos, 4), "95% CI": ""},
        {"Measure": "LR-", "Estimate": "" if np.isnan(lr_neg) else round(lr_neg, 4), "95% CI": ""},
    ])
    return out

def chi_square_tests(table: np.ndarray):
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    out = pd.DataFrame([{
        "Test": "Pearson Chi-Square",
        "Value": round(float(chi2), 4),
        "df": int(dof),
        "Asymp. Sig. (2-sided)": format_p_value(p),
        "Significant (p<0.05)": yesno_from_p(p)
    }])

    yates_df = None
    if table.shape == (2, 2):
        chi2_y, p_y, dof_y, _ = chi2_contingency(table, correction=True)
        yates_df = pd.DataFrame([{
            "Test": "Continuity Correction (Yates)",
            "Value": round(float(chi2_y), 4),
            "df": int(dof_y),
            "Asymp. Sig. (2-sided)": format_p_value(p_y),
            "Significant (p<0.05)": yesno_from_p(p_y)
        }])

    expected_df = pd.DataFrame(expected)
    expected_df.columns = [f"Col{i+1}" for i in range(expected_df.shape[1])]
    expected_df.insert(0, "Row", [f"Row{i+1}" for i in range(expected_df.shape[0])])
    return out, yates_df, expected_df

def fisher_exact_2x2(table: np.ndarray):
    if table.shape != (2, 2):
        raise ValueError("Fisher's Exact requires a 2×2 table.")
    OR, p = stats.fisher_exact(table, alternative="two-sided")
    out = pd.DataFrame([{
        "Test": "Fisher's Exact Test",
        "Odds Ratio": round(float(OR), 4),
        "Exact Sig. (2-sided)": format_p_value(p),
        "Significant (p<0.05)": yesno_from_p(p)
    }])
    return out

def goodness_of_fit(df: pd.DataFrame):
    d = df.copy()
    d["Observed"] = pd.to_numeric(d["Observed"], errors="coerce")
    d = d.dropna(subset=["Observed"])
    obs = d["Observed"].values.astype(float)

    if "Expected proportion (optional)" in d.columns and d["Expected proportion (optional)"].notna().any():
        p = pd.to_numeric(d["Expected proportion (optional)"], errors="coerce").fillna(0).values.astype(float)
        if p.sum() <= 0:
            raise ValueError("Expected proportions must sum to a positive value.")
        p = p / p.sum()
        exp = obs.sum() * p
        chi2, pval = chisquare(f_obs=obs, f_exp=exp)
    else:
        chi2, pval = chisquare(f_obs=obs)
        exp = np.repeat(obs.sum()/len(obs), len(obs))

    exp_df = pd.DataFrame({"Category": d["Category"].astype(str).values, "Expected": np.round(exp, 4)})
    out = pd.DataFrame([{
        "Test": "Chi-Square Goodness-of-Fit",
        "Value": round(float(chi2), 4),
        "df": int(len(obs)-1),
        "Asymp. Sig.": format_p_value(pval),
        "Significant (p<0.05)": yesno_from_p(pval)
    }])
    return out, exp_df

def mantel_haenszel_from_long(df: pd.DataFrame, stratum="Stratum", exposure="Exposure", outcome="Outcome"):
    d = df[[stratum, exposure, outcome]].dropna().copy()
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    d = d.dropna()
    if not set(d[outcome].unique()).issubset({0, 1}):
        raise ValueError("Outcome must be binary coded as 0/1.")

    strata_tables = []
    for s, g in d.groupby(stratum):
        ex_levels = list(g[exposure].unique())
        if len(ex_levels) != 2:
            raise ValueError(f"Each stratum must have exactly 2 exposure levels. Problem in {s}.")
        a = len(g[(g[exposure]==ex_levels[0]) & (g[outcome]==1)])
        b = len(g[(g[exposure]==ex_levels[0]) & (g[outcome]==0)])
        c = len(g[(g[exposure]==ex_levels[1]) & (g[outcome]==1)])
        d0= len(g[(g[exposure]==ex_levels[1]) & (g[outcome]==0)])
        strata_tables.append([[a,b],[c,d0]])

    stbl = StratifiedTable(strata_tables)
    mh_or = float(stbl.oddsratio_pooled)
    ci_lo, ci_hi = stbl.oddsratio_pooled_confint()
    chi2 = float(stbl.test_null_odds().statistic)
    pval = float(stbl.test_null_odds().pvalue)

    out = pd.DataFrame([{
        "Test": "Mantel–Haenszel Common Odds Ratio",
        "Common OR": round(mh_or, 4),
        "95% CI Lower": round(float(ci_lo), 4),
        "95% CI Upper": round(float(ci_hi), 4),
        "Chi-Square": round(chi2, 4),
        "Asymp. Sig.": format_p_value(pval),
        "Significant (p<0.05)": yesno_from_p(pval)
    }])
    return out

def cochran_q_with_posthoc(df: pd.DataFrame, subject_col="Subject"):
    d = df.copy()
    if subject_col not in d.columns:
        raise ValueError("Subject column not found.")
    cond_cols = [c for c in d.columns if c != subject_col]
    if len(cond_cols) < 3:
        raise ValueError("Cochran's Q requires 3+ condition columns.")

    X = d[cond_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if not set(np.unique(X.values)).issubset({0, 1}):
        raise ValueError("Conditions must be binary coded as 0/1.")

    q_res = sm_cochrans_q(X.values)
    Q = float(q_res.statistic)
    p = float(q_res.pvalue)

    main = pd.DataFrame([{
        "Test": "Cochran's Q Test",
        "Q": round(Q, 4),
        "df": int(len(cond_cols)-1),
        "Asymp. Sig.": format_p_value(p),
        "Significant (p<0.05)": yesno_from_p(p)
    }])

    # McNemar post-hoc + Bonferroni
    pairs = []
    m = len(cond_cols)
    n_tests = m*(m-1)//2
    alpha_bonf = 0.05 / n_tests if n_tests > 0 else 0.05

    for i in range(m):
        for j in range(i+1, m):
            c1, c2 = cond_cols[i], cond_cols[j]
            tab = pd.crosstab(X[c1], X[c2]).reindex(index=[0,1], columns=[0,1], fill_value=0).values
            mc = mcnemar(tab, exact=False, correction=True)
            p_ij = float(mc.pvalue)
            pairs.append({
                "Comparison": f"{c1} vs {c2}",
                "McNemar Chi-Square": round(float(mc.statistic), 4),
                "Sig.": format_p_value(p_ij),
                "Bonferroni alpha": round(alpha_bonf, 6),
                "Significant (Bonferroni)": "Yes" if p_ij < alpha_bonf else "No"
            })

    posthoc = pd.DataFrame(pairs)
    return main, posthoc

# =========================================================
# CI Estimation
# =========================================================
def ci_mean(x: np.ndarray, alpha=0.05):
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 values.")
    m = np.mean(x)
    s = np.std(x, ddof=1)
    tcrit = stats.t.ppf(1-alpha/2, df=n-1)
    half = tcrit * s / math.sqrt(n)
    return m, m-half, m+half

def ci_variance(x: np.ndarray, alpha=0.05):
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 values.")
    s2 = np.var(x, ddof=1)
    df = n - 1
    lo = df * s2 / stats.chi2.ppf(1 - alpha/2, df)
    hi = df * s2 / stats.chi2.ppf(alpha/2, df)
    return s2, lo, hi

def ci_median_bootstrap(x: np.ndarray, alpha=0.05, n_boot=2000, seed=42):
    x = x[~np.isnan(x)]
    if len(x) < 3:
        raise ValueError("Need at least 3 values.")
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        meds.append(np.median(samp))
    meds = np.array(meds)
    lo = np.quantile(meds, alpha/2)
    hi = np.quantile(meds, 1-alpha/2)
    return np.median(x), lo, hi

# =========================================================
# Sidebar navigation (Expander + Buttons)
# =========================================================
with st.sidebar:
    st.markdown("## Navigation")

    with st.expander("Logistic Regression", expanded=(st.session_state.main_menu == "Logistic Regression")):
        if st.button("Data (Upload & Template)", use_container_width=True):
            st.session_state.main_menu = "Logistic Regression"
            st.session_state.sub_menu = "Data"
        if st.button("EDA", use_container_width=True):
            st.session_state.main_menu = "Logistic Regression"
            st.session_state.sub_menu = "EDA"
        if st.button("Modeling (OR Table)", use_container_width=True):
            st.session_state.main_menu = "Logistic Regression"
            st.session_state.sub_menu = "Modeling"
        if st.button("Model Comparison (ML + ROC)", use_container_width=True):
            st.session_state.main_menu = "Logistic Regression"
            st.session_state.sub_menu = "Comparison"
        if st.button("Export", use_container_width=True):
            st.session_state.main_menu = "Logistic Regression"
            st.session_state.sub_menu = "Export"

    with st.expander("Linear Regression", expanded=(st.session_state.main_menu == "Linear Regression")):
        if st.button("Data (Upload & Template)", key="lin_data", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Data"
        if st.button("Modeling (OLS)", key="lin_model", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Modeling"
        if st.button("ANOVA & Coefficients", key="lin_tables", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Tables"
        if st.button("Assumptions & Diagnostics", key="lin_diag", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Diagnostics"
        if st.button("Export", key="lin_export", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Export"

    with st.expander("Categorical Tests", expanded=(st.session_state.main_menu == "Categorical")):
        if st.button("Contingency Table (r×c) / Chi-square", key="c1", use_container_width=True):
            st.session_state.main_menu = "Categorical"
            st.session_state.sub_menu = "Chi-square"
        if st.button("Fisher's Exact (2×2)", key="c2", use_container_width=True):
            st.session_state.main_menu = "Categorical"
            st.session_state.sub_menu = "Fisher"
        if st.button("Goodness-of-fit", key="c3", use_container_width=True):
            st.session_state.main_menu = "Categorical"
            st.session_state.sub_menu = "GOF"
        if st.button("Mantel–Haenszel (stratified 2×2)", key="c4", use_container_width=True):
            st.session_state.main_menu = "Categorical"
            st.session_state.sub_menu = "MH"
        if st.button("Cochran's Q (+ McNemar post-hoc)", key="c5", use_container_width=True):
            st.session_state.main_menu = "Categorical"
            st.session_state.sub_menu = "CochranQ"
        if st.button("Diagnostic Accuracy (2×2)", key="c6", use_container_width=True):
            st.session_state.main_menu = "Categorical"
            st.session_state.sub_menu = "Diag2x2"

    with st.expander("Estimation (Confidence Intervals)", expanded=(st.session_state.main_menu == "Estimation")):
        if st.button("Mean / Variance / Median CI", key="e1", use_container_width=True):
            st.session_state.main_menu = "Estimation"
            st.session_state.sub_menu = "CI"

# =========================================================
# Main routing
# =========================================================
main = st.session_state.main_menu
sub = st.session_state.sub_menu
st.caption(f"Section: {main} → {sub}")

# =========================================================
# LOGISTIC: Data
# =========================================================
if main == "Logistic Regression" and sub == "Data":
    st.markdown("## Data")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "Download Excel template",
            data=df_to_excel_bytes({"template": make_logistic_template()}),
            file_name="logistic_regression_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with c2:
        st.info("Upload a CSV/XLSX file. Use the template to keep column names consistent.")

    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    df_new = read_uploaded_file(uploaded)
    if df_new is not None:
        st.session_state.df = df_new
        st.session_state.last_results = {}

    show_dataset_status()

# LOGISTIC: EDA
elif main == "Logistic Regression" and sub == "EDA":
    st.markdown("## Explore (EDA)")
    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df
        desc = df.describe(include="all").transpose().reset_index().rename(columns={"index": "Variable"})
        show_table(desc, "Descriptive Statistics", center_all=True)
        download_table_block(desc, "eda_descriptives", "Descriptive Statistics")

        miss = pd.DataFrame({
            "Variable": df.columns,
            "Missing": df.isna().sum().values,
            "Missing %": (df.isna().mean().values * 100).round(2),
        }).sort_values("Missing %", ascending=False).reset_index(drop=True)
        show_table(miss, "Missing Values", center_all=True)
        download_table_block(miss, "eda_missing", "Missing Values")

# LOGISTIC: Modeling
elif main == "Logistic Regression" and sub == "Modeling":
    st.markdown("## Logistic Regression")
    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)
        target = st.selectbox("Dependent Variable (binary)", options=cols)
        features = st.multiselect("Covariates", options=[c for c in cols if c != target])

        if st.button("Run Logistic Regression", type="primary", use_container_width=True):
            try:
                odds_table = run_logistic_or_table(df, target, features)
                st.session_state.last_results["logistic_or"] = {"odds_table": odds_table}

                show_table(odds_table, "Variables in the Equation", center_all=True)
                download_table_block(odds_table, "logistic_variables_in_equation", "Variables in the Equation")
            except Exception as e:
                st.error(f"Modeling failed: {e}")

# LOGISTIC: Comparison
elif main == "Logistic Regression" and sub == "Comparison":
    st.markdown("## Model Comparison (Classification)")
    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)

        target = st.selectbox("Target (binary)", options=cols, key="cmp_target")
        features = st.multiselect("Predictors", options=[c for c in cols if c != target], key="cmp_features")

        c1, c2 = st.columns([1, 1])
        with c1:
            test_size = st.slider("Test size", 0.1, 0.5, 0.25, 0.05)
        with c2:
            random_state = st.number_input("Random state", value=42, step=1)

        if st.button("Run Comparison", type="primary", use_container_width=True):
            try:
                res = compare_classification_models(df, target, features, float(test_size), int(random_state))
                st.session_state.last_results["logistic_cmp"] = res

                show_table(res["metrics"], "Model Summary (AUC, Accuracy, Precision, Recall, F1)", center_all=True)
                download_table_block(res["metrics"], "model_comparison", "Model Summary")

                st.markdown("### ROC Curve")
                fig, ax = plt.subplots()
                for name, (fpr, tpr, auc_val) in res["roc_lines"].items():
                    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
                ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                download_figure_block(fig, "roc_curve_comparison")
                plt.close(fig)

                st.markdown("### Confusion Matrix (Top model, threshold=0.5)")
                cm = res["cm_top"]
                fig, ax = plt.subplots()
                im = ax.imshow(cm)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix: {res['top_model_name']}")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, str(v), ha="center", va="center")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                download_figure_block(fig, "confusion_matrix_top")
                plt.close(fig)

            except Exception as e:
                st.error(f"Comparison failed: {e}")

# LOGISTIC: Export
elif main == "Logistic Regression" and sub == "Export":
    st.markdown("## Export (Logistic)")
    show_dataset_status()

    sheets = {}
    if st.session_state.df is not None:
        sheets["Dataset"] = st.session_state.df

    if "logistic_or" in st.session_state.last_results:
        sheets["Variables_in_Equation"] = st.session_state.last_results["logistic_or"]["odds_table"]
    if "logistic_cmp" in st.session_state.last_results:
        sheets["Model_Comparison"] = st.session_state.last_results["logistic_cmp"]["metrics"]

    if sheets:
        st.download_button(
            "Download (Excel)",
            data=df_to_excel_bytes(sheets),
            file_name="logistic_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("No outputs available yet.")

# =========================================================
# LINEAR
# =========================================================
elif main == "Linear Regression" and sub == "Data":
    st.markdown("## Data (Linear Regression)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "Download Excel template",
            data=df_to_excel_bytes({"template": make_linear_template()}),
            file_name="linear_regression_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with c2:
        st.info("Upload a CSV/XLSX file. Outcome must be numeric (continuous).")

    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"], label_visibility="collapsed", key="lin_up")
    df_new = read_uploaded_file(uploaded)
    if df_new is not None:
        st.session_state.df = df_new
        st.session_state.last_results = {}
    show_dataset_status()

elif main == "Linear Regression" and sub == "Modeling":
    st.markdown("## Multivariable Linear Regression (OLS)")
    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)
        target = st.selectbox("Dependent Variable (numeric)", options=cols, key="lin_target")
        features = st.multiselect("Independent Variables", options=[c for c in cols if c != target], key="lin_feat")

        if st.button("Run Linear Regression", type="primary", use_container_width=True):
            try:
                res = run_linear_regression(df, target, features)
                st.session_state.last_results["linear"] = res
                show_table(res["metrics"], "Model Summary", center_all=True)
                download_table_block(res["metrics"], "linear_model_summary", "Model Summary")
            except Exception as e:
                st.error(f"Modeling failed: {e}")

elif main == "Linear Regression" and sub == "Tables":
    st.markdown("## Linear Regression Tables")
    show_dataset_status()
    if "linear" not in st.session_state.last_results:
        st.info("Run the model first (Modeling).")
    else:
        res = st.session_state.last_results["linear"]
        show_table(res["anova_table"], "ANOVA", center_all=True)
        download_table_block(res["anova_table"], "linear_anova", "ANOVA")

        show_table(res["coef_table"], "Coefficients", center_all=True)
        download_table_block(res["coef_table"], "linear_coefficients", "Coefficients")

elif main == "Linear Regression" and sub == "Diagnostics":
    st.markdown("## Assumptions & Diagnostics")
    show_dataset_status()
    if "linear" not in st.session_state.last_results:
        st.info("Run the model first (Modeling).")
    else:
        res = st.session_state.last_results["linear"]
        resid = res["resid"]
        fitted = res["fitted"]

        st.markdown("### Residuals vs Fitted")
        fig, ax = plt.subplots()
        ax.scatter(fitted, resid)
        ax.axhline(0)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)
        download_figure_block(fig, "linear_residuals_vs_fitted")
        plt.close(fig)

        st.markdown("### Q-Q Plot (Residuals)")
        fig = sm.qqplot(resid, line="45")
        st.pyplot(fig)
        download_figure_block(plt.gcf(), "linear_qqplot")
        plt.close(plt.gcf())

        show_table(res["shapiro"], "Tests of Normality", center_all=True)
        download_table_block(res["shapiro"], "linear_normality", "Tests of Normality")

        show_table(res["bp"], "Homoscedasticity (Breusch-Pagan)", center_all=True)
        download_table_block(res["bp"], "linear_breusch_pagan", "Breusch-Pagan")

        show_table(res["vif"], "Collinearity Statistics (VIF)", center_all=True)
        download_table_block(res["vif"], "linear_vif", "VIF")

        show_table(res["dw"], "Independence of Errors (Durbin-Watson)", center_all=True)
        download_table_block(res["dw"], "linear_dw", "Durbin-Watson")

elif main == "Linear Regression" and sub == "Export":
    st.markdown("## Export (Linear)")
    show_dataset_status()

    sheets = {}
    if st.session_state.df is not None:
        sheets["Dataset"] = st.session_state.df
    if "linear" in st.session_state.last_results:
        res = st.session_state.last_results["linear"]
        sheets["Model_Summary"] = res["metrics"]
        sheets["ANOVA"] = res["anova_table"]
        sheets["Coefficients"] = res["coef_table"]
        sheets["VIF"] = res["vif"]
        sheets["Normality"] = res["shapiro"]
        sheets["Breusch_Pagan"] = res["bp"]
        sheets["Durbin_Watson"] = res["dw"]

    if sheets:
        st.download_button(
            "Download (Excel)",
            data=df_to_excel_bytes(sheets),
            file_name="linear_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("No outputs available yet.")

# =========================================================
# CATEGORICAL TESTS
# =========================================================
elif main == "Categorical" and sub == "Chi-square":
    st.markdown("## Contingency Table (r×c) and Chi-square")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        r = st.number_input("Rows (r)", min_value=2, max_value=20, value=2, step=1)
    with c2:
        c = st.number_input("Columns (c)", min_value=2, max_value=20, value=2, step=1)
    with c3:
        st.info("You can edit counts directly, or download a template and upload back.")

    # Editable table
    if "ctable_df" not in st.session_state or st.session_state.get("ctable_shape") != (int(r), int(c)):
        st.session_state.ctable_shape = (int(r), int(c))
        st.session_state.ctable_df = make_counts_template(int(r), int(c))

    st.download_button(
        "Download Excel template (counts table)",
        data=df_to_excel_bytes({"contingency_table": st.session_state.ctable_df}),
        file_name=f"contingency_{int(r)}x{int(c)}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    up = st.file_uploader("Or upload counts table (Excel/CSV)", type=["xlsx", "xls", "csv"], key="chi_up")
    if up is not None:
        df_up = read_uploaded_file(up)
        if df_up is not None:
            st.session_state.ctable_df = df_up.copy()

    st.markdown("### Input counts (edit here)")
    edited = st.data_editor(st.session_state.ctable_df, use_container_width=True, num_rows="fixed")
    st.session_state.ctable_df = edited

    if st.button("Run Chi-square", type="primary", use_container_width=True):
        try:
            df_in = st.session_state.ctable_df.copy()
            if "Row" in df_in.columns:
                df_in = df_in.drop(columns=["Row"])
            table = df_in.apply(pd.to_numeric, errors="coerce").values
            if np.isnan(table).any():
                raise ValueError("Please fill all cells with numeric counts.")
            if (table < 0).any():
                raise ValueError("Counts must be non-negative.")

            chi_df, yates_df, expected_df = chi_square_tests(table)

            show_table(chi_df, "Chi-Square Tests", center_all=True)
            download_table_block(chi_df, "chi_square_tests", "Chi-Square Tests")

            if yates_df is not None:
                show_table(yates_df, "Continuity Correction (2×2)", center_all=True)
                download_table_block(yates_df, "chi_square_yates", "Continuity Correction")

            show_table(expected_df, "Expected Counts", center_all=True)
            download_table_block(expected_df, "expected_counts", "Expected Counts")

            # If 2x2: Risk estimate + Diagnostic accuracy
            if table.shape == (2, 2):
                a, b = table[0, 0], table[0, 1]
                c0, d0 = table[1, 0], table[1, 1]

                risk_df = calc_or_rr_ve_2x2(a, b, c0, d0)
                show_table(risk_df, "Risk Estimate (OR / RR / VE)", center_all=True)
                download_table_block(risk_df, "risk_estimate", "Risk Estimate")

                # Assume table layout is:
                # Row1/Row2 are "Exposure" groups and Col1/Col2 are outcomes? That's ambiguous.
                # For Diagnostic Accuracy we need TP/FP/FN/TN. We'll ask user to use Diag2x2 menu for that.
                st.info("For diagnostic performance metrics (Sensitivity/Specificity/LR/PPV/NPV), use 'Diagnostic Accuracy (2×2)' menu to input TP/FP/FN/TN explicitly.")

        except Exception as e:
            st.error(f"Failed: {e}")

elif main == "Categorical" and sub == "Fisher":
    st.markdown("## Fisher's Exact Test (2×2)")

    st.download_button(
        "Download Excel template (2×2 counts)",
        data=df_to_excel_bytes({"contingency_2x2": make_counts_template(2, 2)}),
        file_name="fisher_2x2_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    if "fisher_df" not in st.session_state:
        st.session_state.fisher_df = make_counts_template(2, 2)

    up = st.file_uploader("Upload counts table (2×2)", type=["xlsx", "xls", "csv"], key="fisher_up")
    if up is not None:
        df_up = read_uploaded_file(up)
        if df_up is not None:
            st.session_state.fisher_df = df_up.copy()

    st.markdown("### Input counts (edit here)")
    edited = st.data_editor(st.session_state.fisher_df, use_container_width=True, num_rows="fixed")
    st.session_state.fisher_df = edited

    if st.button("Run Fisher's Exact", type="primary", use_container_width=True):
        try:
            df_in = st.session_state.fisher_df.copy()
            if "Row" in df_in.columns:
                df_in = df_in.drop(columns=["Row"])
            table = df_in.apply(pd.to_numeric, errors="coerce").values
            if table.shape != (2, 2):
                raise ValueError("Input must be 2×2.")
            if np.isnan(table).any():
                raise ValueError("Fill all cells with numeric counts.")
            if (table < 0).any():
                raise ValueError("Counts must be non-negative.")

            fish_df = fisher_exact_2x2(table)
            show_table(fish_df, "Fisher's Exact Test", center_all=True)
            download_table_block(fish_df, "fisher_exact", "Fisher's Exact Test")

            a, b = table[0, 0], table[0, 1]
            c0, d0 = table[1, 0], table[1, 1]
            risk_df = calc_or_rr_ve_2x2(a, b, c0, d0)
            show_table(risk_df, "Risk Estimate (OR / RR / VE)", center_all=True)
            download_table_block(risk_df, "risk_estimate_fisher", "Risk Estimate")

        except Exception as e:
            st.error(f"Failed: {e}")

elif main == "Categorical" and sub == "GOF":
    st.markdown("## Chi-square Goodness-of-fit")

    st.download_button(
        "Download Excel template",
        data=df_to_excel_bytes({"goodness_of_fit": make_gof_template()}),
        file_name="goodness_of_fit_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    up = st.file_uploader("Upload GOF dataset", type=["xlsx", "xls", "csv"], key="gof_up")
    if up is not None:
        df_new = read_uploaded_file(up)
        if df_new is not None:
            st.session_state.df = df_new

    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### Input data")
        st.dataframe(df, use_container_width=True)

        if st.button("Run Goodness-of-fit", type="primary", use_container_width=True):
            try:
                out, exp = goodness_of_fit(df)
                show_table(out, "Chi-Square Goodness-of-Fit", center_all=True)
                download_table_block(out, "gof_test", "Goodness-of-Fit")

                show_table(exp, "Expected Frequencies", center_all=True)
                download_table_block(exp, "gof_expected", "Expected Frequencies")
            except Exception as e:
                st.error(f"Failed: {e}")

elif main == "Categorical" and sub == "MH":
    st.markdown("## Mantel–Haenszel (Stratified 2×2)")

    st.download_button(
        "Download Excel template",
        data=df_to_excel_bytes({"mh_long": make_mh_long_template()}),
        file_name="mantel_haenszel_long_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    up = st.file_uploader("Upload long-format dataset", type=["xlsx", "xls", "csv"], key="mh_up")
    if up is not None:
        df_new = read_uploaded_file(up)
        if df_new is not None:
            st.session_state.df = df_new

    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)
        s = st.selectbox("Stratum", cols, index=0)
        e = st.selectbox("Exposure (2 levels)", cols, index=min(1, len(cols)-1))
        o = st.selectbox("Outcome (0/1)", cols, index=min(2, len(cols)-1))

        if st.button("Run Mantel–Haenszel", type="primary", use_container_width=True):
            try:
                out = mantel_haenszel_from_long(df, s, e, o)
                show_table(out, "Mantel–Haenszel Test", center_all=True)
                download_table_block(out, "mantel_haenszel", "Mantel–Haenszel")
            except Exception as ex:
                st.error(f"Failed: {ex}")

elif main == "Categorical" and sub == "CochranQ":
    st.markdown("## Cochran's Q (+ McNemar post-hoc Bonferroni)")

    st.download_button(
        "Download Excel template",
        data=df_to_excel_bytes({"cochran_q": make_cochran_template()}),
        file_name="cochran_q_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    up = st.file_uploader("Upload dataset", type=["xlsx", "xls", "csv"], key="cq_up")
    if up is not None:
        df_new = read_uploaded_file(up)
        if df_new is not None:
            st.session_state.df = df_new

    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df
        subject = st.selectbox("Subject column", df.columns, index=0)

        if st.button("Run Cochran's Q", type="primary", use_container_width=True):
            try:
                main_df, post_df = cochran_q_with_posthoc(df, subject_col=subject)
                show_table(main_df, "Cochran's Q Test", center_all=True)
                download_table_block(main_df, "cochran_q", "Cochran's Q")

                show_table(post_df, "Pairwise McNemar Tests (Bonferroni)", center_all=True)
                download_table_block(post_df, "mcnemar_posthoc", "McNemar Post-hoc")
            except Exception as ex:
                st.error(f"Failed: {ex}")

elif main == "Categorical" and sub == "Diag2x2":
    st.markdown("## Diagnostic Accuracy (2×2)")

    st.download_button(
        "Download Excel template (TP/FP/FN/TN)",
        data=df_to_excel_bytes({"diagnostic_2x2": make_diag2x2_template()}),
        file_name="diagnostic_2x2_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    if "diag_df" not in st.session_state:
        st.session_state.diag_df = make_diag2x2_template()

    up = st.file_uploader("Upload diagnostic 2×2 table", type=["xlsx", "xls", "csv"], key="diag_up")
    if up is not None:
        df_up = read_uploaded_file(up)
        if df_up is not None:
            st.session_state.diag_df = df_up.copy()

    st.markdown("### Input table (edit here)")
    edited = st.data_editor(st.session_state.diag_df, use_container_width=True, num_rows="fixed")
    st.session_state.diag_df = edited

    if st.button("Compute Diagnostic Metrics", type="primary", use_container_width=True):
        try:
            d = st.session_state.diag_df.copy()
            # Expect rows: Actual Positive/Actual Negative; cols: Test Positive/Test Negative
            if d.shape[0] < 2 or d.shape[1] < 3:
                raise ValueError("Template must have 2 rows and 2 count columns.")

            tp = float(pd.to_numeric(d.loc[0, "Test Positive"], errors="coerce"))
            fn = float(pd.to_numeric(d.loc[0, "Test Negative"], errors="coerce"))
            fp = float(pd.to_numeric(d.loc[1, "Test Positive"], errors="coerce"))
            tn = float(pd.to_numeric(d.loc[1, "Test Negative"], errors="coerce"))

            if any(np.isnan([tp, fn, fp, tn])):
                raise ValueError("Fill all four cells (TP, FN, FP, TN).")
            if min(tp, fn, fp, tn) < 0:
                raise ValueError("Counts must be non-negative.")

            # SPSS-like: show the table too
            show_table(pd.DataFrame([{
                "TP": tp, "FN": fn, "FP": fp, "TN": tn
            }]), "Case Processing Summary", center_all=True)

            acc = diagnostic_accuracy_2x2(tp, fp, fn, tn)
            show_table(acc, "Classification Statistics", center_all=True)
            download_table_block(acc, "diagnostic_accuracy", "Classification Statistics")

        except Exception as e:
            st.error(f"Failed: {e}")

# =========================================================
# ESTIMATION: CI
# =========================================================
elif main == "Estimation" and sub == "CI":
    st.markdown("## Confidence Intervals (Estimation)")

    st.download_button(
        "Download Excel template",
        data=df_to_excel_bytes({"values": make_ci_template()}),
        file_name="ci_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    up = st.file_uploader("Upload dataset", type=["xlsx", "xls", "csv"], key="ci_up")
    if up is not None:
        df_new = read_uploaded_file(up)
        if df_new is not None:
            st.session_state.df = df_new

    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df
        col = st.selectbox("Value column", df.columns)
        alpha = st.slider("Alpha (1 - confidence level)", 0.01, 0.20, 0.05, 0.01)

        x = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(float)
        if st.button("Compute CI", type="primary", use_container_width=True):
            try:
                m, lo, hi = ci_mean(x, alpha=float(alpha))
                v, vlo, vhi = ci_variance(x, alpha=float(alpha))
                med, med_lo, med_hi = ci_median_bootstrap(x, alpha=float(alpha))

                out = pd.DataFrame([
                    {"Parameter": "Mean", "Estimate": round(m, 4), "CI Lower": round(lo, 4), "CI Upper": round(hi, 4)},
                    {"Parameter": "Variance", "Estimate": round(v, 4), "CI Lower": round(vlo, 4), "CI Upper": round(vhi, 4)},
                    {"Parameter": "Std. Deviation", "Estimate": round(math.sqrt(v), 4),
                     "CI Lower": round(math.sqrt(vlo), 4), "CI Upper": round(math.sqrt(vhi), 4)},
                    {"Parameter": "Median (bootstrap)", "Estimate": round(med, 4), "CI Lower": round(med_lo, 4), "CI Upper": round(med_hi, 4)},
                ])

                show_table(out, "Estimates and Confidence Intervals", center_all=True)
                download_table_block(out, "confidence_intervals", "Estimates and CI")

            except Exception as e:
                st.error(f"Failed: {e}")

else:
    st.info("Select a function from the sidebar.")
