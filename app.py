import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.anova import anova_lm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Regression Applications in Health Sciences",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CSS (fix expander white + header clipped)
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

/* ===== FIX: Expander header white background ===== */
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

/* Force file uploader styling in sidebar (no white card) */
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

/* ===== Header =====
   IMPORTANT: do NOT use negative left margin (it gets clipped by layout) */
.app-header {
  background: linear-gradient(90deg, #0B3A66, #0A2D4E);
  padding: 16px 28px;
  margin: 0 0 1.0rem 0;     /* no negative margin */
  border-radius: 18px;
  overflow: visible;
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
        <h1>Regression Applications in Health Sciences</h1>
        <p>Logistic & multivariable linear regression with diagnostics, model comparison, and exports</p>
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
# Utilities: downloads
# =========================================================
def df_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
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
    n_rows, n_cols = df.shape
    fig_w = min(18, max(6, n_cols * 1.5))
    fig_h = min(18, max(2.2, (n_rows + 1) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
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

def format_p_value(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return ""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


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


# =========================================================
# Data I/O
# =========================================================
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
        st.info("No dataset loaded yet. Please upload a CSV/XLSX file in the Data section.")
        return
    df = st.session_state.df
    st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    with st.expander("Preview (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)


# =========================================================
# Logistic regression
# =========================================================
def run_logistic_or_table(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    data = df[[target] + features].dropna().copy()

    y_raw = data[target]
    if y_raw.nunique() != 2:
        raise ValueError("Binary logistic regression requires exactly 2 unique target values.")

    # map to 0/1 if needed
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

    odds_table = pd.DataFrame({
        "Term": params.index,
        "Coefficient (β)": params.values,
        "Odds Ratio": np.exp(params.values),
        "CI 2.5%": np.exp(conf[0].values),
        "CI 97.5%": np.exp(conf[1].values),
        "p_raw": pvals.values,
    })

    odds_table["p-value"] = odds_table["p_raw"].apply(format_p_value)
    odds_table["Significant (p<0.05)"] = odds_table["p_raw"].apply(lambda p: "Yes" if float(p) < 0.05 else "")
    odds_table = odds_table.drop(columns=["p_raw"])

    # Hide OR & CI for const (SPSS style)
    is_const = odds_table["Term"].isin(["const", "Intercept"])
    odds_table.loc[is_const, ["Odds Ratio", "CI 2.5%", "CI 97.5%"]] = np.nan

    for col in ["Coefficient (β)", "Odds Ratio", "CI 2.5%", "CI 97.5%"]:
        odds_table[col] = pd.to_numeric(odds_table[col], errors="coerce").round(4)

    odds_table = odds_table[
        ["Term", "Coefficient (β)", "Odds Ratio", "CI 2.5%", "CI 97.5%", "p-value", "Significant (p<0.05)"]
    ]
    return odds_table


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

        # probabilities for ROC
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            # fallback
            prob = model.decision_function(X_test)
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-12)

        pred = (prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        pre = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc_val = roc_auc_score(y_test, prob)

        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_lines[name] = (fpr, tpr, auc_val)

        rows.append({
            "Model": name,
            "AUC": round(float(auc_val), 4),
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(pre), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
        })

    metrics_df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    # Confusion matrix for top model
    top_name = metrics_df.loc[0, "Model"]
    top_model = models[top_name]
    if hasattr(top_model, "predict_proba"):
        top_prob = top_model.predict_proba(X_test)[:, 1]
    else:
        top_prob = top_model.decision_function(X_test)
        top_prob = (top_prob - top_prob.min()) / (top_prob.max() - top_prob.min() + 1e-12)
    top_pred = (top_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, top_pred)

    return {"metrics": metrics_df, "roc_lines": roc_lines, "cm_top": cm, "top_model_name": top_name}


# =========================================================
# Linear regression (OLS via formula to fix design_info error)
# =========================================================
def q(name: str) -> str:
    # Patsy quote helper for any column name
    return f'Q("{name}")'

def run_linear_regression(df: pd.DataFrame, target: str, features: list[str]):
    data = df[[target] + features].dropna().copy()

    # Ensure numeric
    data[target] = pd.to_numeric(data[target], errors="coerce")
    for c in features:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()

    if data.shape[0] < max(10, len(features) + 3):
        raise ValueError("Not enough rows after cleaning. Please check missing or non-numeric values.")

    # Build formula (for ANOVA compatibility)
    def q(name: str) -> str:
        return f'Q("{name}")'

    formula = f'{q(target)} ~ ' + " + ".join(q(c) for c in features)

    model = smf.ols(formula=formula, data=data).fit()

    # =========================
    # Coefficients table
    # =========================
    conf = model.conf_int()

    coef_df = pd.DataFrame({
        "Term": model.params.index,
        "Coefficient (B)": model.params.values,
        "Std. Error": model.bse.values,
        "t": model.tvalues.values,
        "p_raw": model.pvalues.values,
        "CI 2.5%": conf[0].values,
        "CI 97.5%": conf[1].values,
    })

    # Clean Q("var") → var
    def _clean_term(s: str) -> str:
        if isinstance(s, str) and s.startswith('Q("') and s.endswith('")'):
            return s[3:-2]
        return s

    coef_df["Term"] = coef_df["Term"].apply(_clean_term)

    coef_df["p-value"] = coef_df["p_raw"].apply(format_p_value)
    coef_df["Significant (p<0.05)"] = coef_df["p_raw"].apply(
        lambda p: "Yes" if float(p) < 0.05 else "No"
    )
    coef_df = coef_df.drop(columns=["p_raw"])

    for col in ["Coefficient (B)", "Std. Error", "t", "CI 2.5%", "CI 97.5%"]:
        coef_df[col] = pd.to_numeric(coef_df[col], errors="coerce").round(4)

    coef_df = coef_df[
        ["Term", "Coefficient (B)", "Std. Error", "t", "CI 2.5%", "CI 97.5%", "p-value", "Significant (p<0.05)"]
    ]

    # =========================
    # ANOVA table
    # =========================
    anova_df = anova_lm(model, typ=1).reset_index().rename(columns={"index": "Source"})
    anova_df["Source"] = anova_df["Source"].apply(_clean_term)

    anova_df = anova_df.rename(columns={
        "df": "df",
        "sum_sq": "Sum Sq",
        "mean_sq": "Mean Sq",
        "F": "F",
        "PR(>F)": "p_raw"
    })

    if "p_raw" in anova_df.columns:
        anova_df["p-value"] = anova_df["p_raw"].apply(format_p_value)
        anova_df = anova_df.drop(columns=["p_raw"])

    for col in ["Sum Sq", "Mean Sq", "F"]:
        if col in anova_df.columns:
            anova_df[col] = pd.to_numeric(anova_df[col], errors="coerce").round(4)

    # Replace NaN -> None (SPSS-style)  ✅ apply to the whole ANOVA table
    anova_df = anova_df.astype(object).where(pd.notna(anova_df), None)


    # =========================
    # Model fit metrics
    # =========================
    metrics = pd.DataFrame([{
        "N": int(model.nobs),
        "R-squared": round(float(model.rsquared), 4),
        "Adj. R-squared": round(float(model.rsquared_adj), 4),
        "AIC": round(float(model.aic), 4),
        "BIC": round(float(model.bic), 4),
        "F-statistic": round(float(model.fvalue), 4) if model.fvalue is not None else None,
        "F p-value": format_p_value(float(model.f_pvalue)) if model.f_pvalue is not None else None,
    }])

    # =========================
    # Diagnostics
    # =========================
    resid = model.resid
    fitted = model.fittedvalues

    # VIF
    X = data[features]
    X_sm = sm.add_constant(X)
    vif_rows = []
    for i, col in enumerate(X_sm.columns):
        if col == "const":
            continue
        vif_rows.append({
            "Variable": col,
            "VIF": round(float(variance_inflation_factor(X_sm.values, i)), 4)
        })
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)

    # Normality
    s_stat, s_p = shapiro(resid)
    shapiro_df = pd.DataFrame([{
        "Test": "Shapiro-Wilk",
        "Statistic": round(float(s_stat), 4),
        "p-value": format_p_value(float(s_p))
    }])

    # Homoscedasticity
    lm, lm_p, fval, f_p = het_breuschpagan(resid, sm.add_constant(X))
    bp_df = pd.DataFrame([{
        "Test": "Breusch-Pagan",
        "LM Statistic": round(float(lm), 4),
        "LM p-value": format_p_value(float(lm_p)),
        "F Statistic": round(float(fval), 4),
        "F p-value": format_p_value(float(f_p)),
    }])

    # Independence
    dw_df = pd.DataFrame([{
        "Test": "Durbin-Watson",
        "Statistic": round(float(durbin_watson(resid)), 4)
    }])

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
        if st.button("Model Comparison (ML)", use_container_width=True):
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

# =========================================================
# LOGISTIC: EDA
# =========================================================
elif main == "Logistic Regression" and sub == "EDA":
    st.markdown("## Explore (EDA)")
    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df

        desc = df.describe(include="all").transpose().reset_index().rename(columns={"index": "Column"})
        st.markdown("### Summary")
        st.dataframe(desc, use_container_width=True)
        download_table_block(desc, "eda_summary", "EDA Summary")

        miss = pd.DataFrame({
            "Column": df.columns,
            "Missing count": df.isna().sum().values,
            "Missing %": (df.isna().mean().values * 100).round(2),
        }).sort_values("Missing %", ascending=False).reset_index(drop=True)

        st.markdown("### Missing values")
        st.dataframe(miss, use_container_width=True)
        download_table_block(miss, "eda_missing", "Missing Values")

# =========================================================
# LOGISTIC: Modeling (OR table)
# =========================================================
elif main == "Logistic Regression" and sub == "Modeling":
    st.markdown("## Modeling (Logistic Regression)")
    show_dataset_status()

    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)

        target = st.selectbox("Target (binary)", options=cols)
        features = st.multiselect("Predictors", options=[c for c in cols if c != target])

        run = st.button("Run Logistic OR Table", type="primary", use_container_width=True)
        if run:
            try:
                odds_table = run_logistic_or_table(df, target, features)
                st.session_state.last_results["logistic_or"] = {"odds_table": odds_table}

                st.markdown("### Variables in the Equation")
                st.dataframe(odds_table, use_container_width=True)
                download_table_block(odds_table, "logistic_variables_in_equation", "Variables in the Equation")
            except Exception as e:
                st.error(f"Modeling failed: {e}")

# =========================================================
# LOGISTIC: Comparison (restore ML comparison + ROC legend)
# =========================================================
elif main == "Logistic Regression" and sub == "Comparison":
    st.markdown("## Model Comparison (Classification)")
    show_dataset_status()

    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)

        st.markdown("### Setup")
        target = st.selectbox("Target (binary)", options=cols, key="cmp_target")
        features = st.multiselect("Predictors", options=[c for c in cols if c != target], key="cmp_features")

        c1, c2 = st.columns([1, 1])
        with c1:
            test_size = st.slider("Test size", 0.1, 0.5, 0.25, 0.05, key="cmp_test")
        with c2:
            random_state = st.number_input("Random state", value=42, step=1, key="cmp_rs")

        run = st.button("Run Model Comparison", type="primary", use_container_width=True)

        if run:
            try:
                res = compare_classification_models(
                    df, target, features, test_size=float(test_size), random_state=int(random_state)
                )
                st.session_state.last_results["logistic_cmp"] = res

                st.markdown("### Performance metrics")
                st.dataframe(res["metrics"], use_container_width=True)
                download_table_block(res["metrics"], "classification_model_comparison", "Model Comparison Metrics")

                st.markdown("### ROC curves (with AUC legend)")
                fig, ax = plt.subplots()
                for name, (fpr, tpr, auc_val) in res["roc_lines"].items():
                    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
                ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                download_figure_block(fig, "roc_model_comparison")
                plt.close(fig)

                st.info(f"Top model by AUC: {res['top_model_name']}")

                st.markdown("### Confusion matrix (Top model, threshold=0.5)")
                cm = res["cm_top"]
                fig, ax = plt.subplots()
                im = ax.imshow(cm)
                ax.set_title("Confusion Matrix (Top model)")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, str(v), ha="center", va="center")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                download_figure_block(fig, "confusion_matrix_top_model")
                plt.close(fig)

            except Exception as e:
                st.error(f"Comparison failed: {e}")

# =========================================================
# LOGISTIC: Export
# =========================================================
elif main == "Logistic Regression" and sub == "Export":
    st.markdown("## Export (Logistic)")
    show_dataset_status()

    if st.session_state.df is not None:
        st.download_button(
            "Download dataset as Excel",
            data=df_to_excel_bytes({"dataset": st.session_state.df}),
            file_name="dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    sheets = {}
    if "logistic_or" in st.session_state.last_results:
        sheets["Variables_in_Equation"] = st.session_state.last_results["logistic_or"]["odds_table"]
    if "logistic_cmp" in st.session_state.last_results:
        sheets["Model_Comparison"] = st.session_state.last_results["logistic_cmp"]["metrics"]

    if sheets:
        st.download_button(
            "Download latest logistic outputs (Excel)",
            data=df_to_excel_bytes(sheets),
            file_name="logistic_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("No logistic outputs yet. Run a model first.")

# =========================================================
# LINEAR: Data
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
        st.info("Upload a CSV/XLSX file. The linear outcome must be numeric (continuous).")

    uploaded = st.file_uploader(
        "Upload CSV/XLSX", type=["csv", "xlsx", "xls"], label_visibility="collapsed", key="lin_upload"
    )
    df_new = read_uploaded_file(uploaded)
    if df_new is not None:
        st.session_state.df = df_new
        st.session_state.last_results = {}

    show_dataset_status()

# =========================================================
# LINEAR: Modeling
# =========================================================
elif main == "Linear Regression" and sub == "Modeling":
    st.markdown("## Modeling (Multivariable Linear Regression / OLS)")
    show_dataset_status()

    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)

        target = st.selectbox("Outcome (numeric)", options=cols, key="lin_target")
        features = st.multiselect("Predictors", options=[c for c in cols if c != target], key="lin_features")

        run = st.button("Run Linear Regression (OLS)", type="primary", use_container_width=True)
        if run:
            try:
                res = run_linear_regression(df, target, features)
                st.session_state.last_results["linear"] = res

                st.markdown("### Model fit")
                st.dataframe(res["metrics"], use_container_width=True)
                download_table_block(res["metrics"], "linear_model_fit", "Model Fit")

                st.success("Model fitted successfully.")
            except Exception as e:
                st.error(f"Modeling failed: {e}")

# =========================================================
# LINEAR: Tables
# =========================================================
elif main == "Linear Regression" and sub == "Tables":
    st.markdown("## ANOVA & Coefficients")
    show_dataset_status()

    if "linear" not in st.session_state.last_results:
        st.info("No linear results yet. Run the model in 'Modeling (OLS)' first.")
    else:
        res = st.session_state.last_results["linear"]

        st.markdown("### ANOVA")
        st.dataframe(res["anova_table"], use_container_width=True)
        download_table_block(res["anova_table"], "linear_anova", "ANOVA Table")

        st.markdown("### Coefficients")
        st.dataframe(res["coef_table"], use_container_width=True)
        download_table_block(res["coef_table"], "linear_coefficients", "Coefficients")

# =========================================================
# LINEAR: Diagnostics
# =========================================================
elif main == "Linear Regression" and sub == "Diagnostics":
    st.markdown("## Assumptions & Diagnostics")
    show_dataset_status()

    if "linear" not in st.session_state.last_results:
        st.info("No linear results yet. Run the model in 'Modeling (OLS)' first.")
    else:
        res = st.session_state.last_results["linear"]
        resid = res["resid"]
        fitted = res["fitted"]

        st.markdown("### Linearity: Residuals vs Fitted")
        fig, ax = plt.subplots()
        ax.scatter(fitted, resid)
        ax.axhline(0)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        st.pyplot(fig)
        download_figure_block(fig, "linear_residuals_vs_fitted")
        plt.close(fig)

        st.markdown("### Normality of residuals")
        fig = sm.qqplot(resid, line="45")
        plt.title("Q-Q Plot of Residuals")
        st.pyplot(fig)
        download_figure_block(plt.gcf(), "linear_qqplot_residuals")
        plt.close(plt.gcf())

        st.dataframe(res["shapiro"], use_container_width=True)
        download_table_block(res["shapiro"], "linear_shapiro", "Normality Test")

        st.markdown("### Homoscedasticity")
        st.dataframe(res["bp"], use_container_width=True)
        download_table_block(res["bp"], "linear_breusch_pagan", "Homoscedasticity")

        st.markdown("### Multicollinearity: VIF")
        st.dataframe(res["vif"], use_container_width=True)
        download_table_block(res["vif"], "linear_vif", "VIF Table")

        st.markdown("### Independence of errors")
        st.dataframe(res["dw"], use_container_width=True)
        download_table_block(res["dw"], "linear_durbin_watson", "Independence of Errors")

# =========================================================
# LINEAR: Export
# =========================================================
elif main == "Linear Regression" and sub == "Export":
    st.markdown("## Export (Linear)")
    show_dataset_status()

    if st.session_state.df is not None:
        st.download_button(
            "Download dataset as Excel",
            data=df_to_excel_bytes({"dataset": st.session_state.df}),
            file_name="dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    if "linear" in st.session_state.last_results:
        res = st.session_state.last_results["linear"]
        st.download_button(
            "Download latest linear outputs (Excel)",
            data=df_to_excel_bytes({
                "Model_Fit": res["metrics"],
                "ANOVA": res["anova_table"],
                "Coefficients": res["coef_table"],
                "VIF": res["vif"],
                "Shapiro": res["shapiro"],
                "Breusch_Pagan": res["bp"],
                "Durbin_Watson": res["dw"],
            }),
            file_name="linear_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("No linear outputs yet. Run a model first.")
