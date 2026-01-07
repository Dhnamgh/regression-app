import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.anova import anova_lm


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Regression Applications in Health Sciences",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CSS (Sidebar + Header + Reduce top gap)
# =========================================================
st.markdown("""
<style>
/* ===== Reduce top empty space ===== */
.block-container {
  padding-top: 1.2rem !important;
}

/* ===== Sidebar background ===== */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0B3A66 0%, #0A2D4E 100%);
}

/* Sidebar text (safe) */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span{
  color: #ffffff !important;
}

/* Sidebar buttons style */
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stDownloadButton button{
  width: 100%;
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
  border-color: rgba(255,255,255,0.55) !important;
  transform: translateY(-1px);
}

/* Expander header in sidebar */
section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary {
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* ===== FORCE file uploader to match sidebar style (remove white card) ===== */
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
  font-weight: 600 !important;
}

/* ===== App Header ===== */
.app-header {
  background: linear-gradient(90deg, #0B3A66, #0A2D4E);
  padding: 14px 28px;
  margin: -1.2rem -1.2rem 1.0rem -1.2rem;
  border-radius: 0 0 18px 18px;
}
.app-header h1 {
  color: #ffffff;
  font-size: 1.6rem;
  margin: 0;
  font-weight: 800;
}
.app-header p {
  color: rgba(255,255,255,0.85);
  margin: 4px 0 0 0;
  font-size: 0.95rem;
}

/* Pull content closer to header */
.app-header + div {
  margin-top: -1.4rem !important;
}
</style>
""", unsafe_allow_html=True)

# Header (global)
st.markdown(
    """
    <div class="app-header">
        <h1>Regression Applications in Health Sciences</h1>
        <p>Unified platform for logistic and multivariable linear regression, with diagnostics and exports</p>
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
    st.session_state.last_results = {}  # store outputs for Export


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
    # SPSS-like formatting
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


# =========================================================
# Modeling functions
# =========================================================
def run_logistic_regression(df: pd.DataFrame, target: str, features: list[str],
                            test_size: float = 0.25, random_state: int = 42):
    data = df[[target] + features].dropna().copy()

    # Ensure binary target
    y_raw = data[target]
    if y_raw.nunique() != 2:
        raise ValueError("Binary logistic regression requires a target with exactly 2 unique values.")

    # Map to 0/1 if needed
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

    # Statsmodels OR table
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

    # Hide OR and CI for constant (SPSS style)
    is_const = odds_table["Term"].isin(["const", "Intercept"])
    odds_table.loc[is_const, ["Odds Ratio", "CI 2.5%", "CI 97.5%"]] = np.nan

    # Round numeric columns
    for col in ["Coefficient (β)", "Odds Ratio", "CI 2.5%", "CI 97.5%"]:
        odds_table[col] = pd.to_numeric(odds_table[col], errors="coerce").round(4)

    # Reorder columns: p-value before Significant
    odds_table = odds_table[
        ["Term", "Coefficient (β)", "Odds Ratio", "CI 2.5%", "CI 97.5%", "p-value", "Significant (p<0.05)"]
    ]

    # Simple ROC using train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logit_tt = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=False)
    prob = logit_tt.predict(sm.add_constant(X_test))
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)

    # Confusion matrix using 0.5
    y_pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": logit,
        "odds_table": odds_table,
        "roc": {"fpr": fpr, "tpr": tpr, "auc": roc_auc},
        "cm": cm,
        "features": features,
        "target": target,
    }


def run_linear_regression(df: pd.DataFrame, target: str, features: list[str]):
    data = df[[target] + features].dropna().copy()

    y = pd.to_numeric(data[target], errors="coerce")
    X = data[features].apply(pd.to_numeric, errors="coerce")
    tmp = pd.concat([y, X], axis=1).dropna()
    y = tmp[target]
    X = tmp[features]

    X_sm = sm.add_constant(X)
    ols = sm.OLS(y, X_sm).fit()

    # Coefficients table
    conf = ols.conf_int()
    coef_df = pd.DataFrame({
        "Term": ols.params.index,
        "Coefficient (B)": ols.params.values,
        "Std. Error": ols.bse.values,
        "t": ols.tvalues.values,
        "p_raw": ols.pvalues.values,
        "CI 2.5%": conf[0].values,
        "CI 97.5%": conf[1].values,
    })
    coef_df["p-value"] = coef_df["p_raw"].apply(format_p_value)
    coef_df["Significant (p<0.05)"] = coef_df["p_raw"].apply(lambda p: "Yes" if float(p) < 0.05 else "")
    coef_df = coef_df.drop(columns=["p_raw"])

    for col in ["Coefficient (B)", "Std. Error", "t", "CI 2.5%", "CI 97.5%"]:
        coef_df[col] = pd.to_numeric(coef_df[col], errors="coerce").round(4)

    coef_df = coef_df[
        ["Term", "Coefficient (B)", "Std. Error", "t", "CI 2.5%", "CI 97.5%", "p-value", "Significant (p<0.05)"]
    ]

    # ANOVA table
    anova_df = anova_lm(ols, typ=1).reset_index().rename(columns={"index": "Source"})
    anova_df = anova_df.rename(columns={
        "df": "df",
        "sum_sq": "Sum Sq",
        "mean_sq": "Mean Sq",
        "F": "F",
        "PR(>F)": "p_raw",
    })
    if "p_raw" in anova_df.columns:
        anova_df["p-value"] = anova_df["p_raw"].apply(format_p_value)
        anova_df = anova_df.drop(columns=["p_raw"])
    for col in ["Sum Sq", "Mean Sq", "F"]:
        if col in anova_df.columns:
            anova_df[col] = pd.to_numeric(anova_df[col], errors="coerce").round(4)

    metrics = pd.DataFrame([{
        "N": int(ols.nobs),
        "R-squared": round(float(ols.rsquared), 4),
        "Adj. R-squared": round(float(ols.rsquared_adj), 4),
        "AIC": round(float(ols.aic), 4),
        "BIC": round(float(ols.bic), 4),
        "F-statistic": round(float(ols.fvalue), 4) if ols.fvalue is not None else np.nan,
        "F p-value": format_p_value(float(ols.f_pvalue)) if ols.f_pvalue is not None else "",
    }])

    return {"model": ols, "coef_table": coef_df, "anova_table": anova_df, "metrics": metrics, "X": X, "y": y}


# =========================================================
# Sidebar navigation: Expander + Buttons
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
        if st.button("Modeling (OR, ROC)", use_container_width=True):
            st.session_state.main_menu = "Logistic Regression"
            st.session_state.sub_menu = "Modeling"
        if st.button("Export", use_container_width=True):
            st.session_state.main_menu = "Logistic Regression"
            st.session_state.sub_menu = "Export"

    with st.expander("Linear Regression", expanded=(st.session_state.main_menu == "Linear Regression")):
        if st.button("Data (Upload & Template)", key="lin_data", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Data"
        if st.button("Assumptions & Diagnostics", key="lin_diag", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Diagnostics"
        if st.button("Modeling (OLS)", key="lin_model", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Modeling"
        if st.button("ANOVA & Coefficients", key="lin_tables", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Tables"
        if st.button("Export", key="lin_export", use_container_width=True):
            st.session_state.main_menu = "Linear Regression"
            st.session_state.sub_menu = "Export"


# =========================================================
# Main routing
# =========================================================
main = st.session_state.main_menu
sub = st.session_state.sub_menu

# Optional breadcrumb
st.caption(f"Section: {main} → {sub}")

# =========================================================
# Shared: show current dataset summary if available
# =========================================================
def show_dataset_status():
    if st.session_state.df is None:
        st.info("No dataset loaded yet. Please upload a CSV/XLSX file in the Data section.")
        return
    df = st.session_state.df
    st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    with st.expander("Preview (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

# =========================================================
# Logistic pages
# =========================================================
if main == "Logistic Regression" and sub == "Data":
    st.markdown("## Data")

    st.markdown("### Templates")
    c1, c2 = st.columns([1, 2])
    with c1:
        tmpl = make_logistic_template()
        st.download_button(
            "Download Excel template",
            data=df_to_excel_bytes({"template": tmpl}),
            file_name="logistic_regression_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with c2:
        st.info("Upload a CSV/XLSX file using the uploader below. Use the template to keep column names consistent.")

    st.markdown("### Upload CSV/XLSX")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    df_new = read_uploaded_file(uploaded)
    if df_new is not None:
        st.session_state.df = df_new
        st.session_state.last_results = {}  # reset old results

    show_dataset_status()

elif main == "Logistic Regression" and sub == "EDA":
    st.markdown("## Explore (EDA)")
    show_dataset_status()
    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("### Summary")
        desc = df.describe(include="all").transpose().reset_index().rename(columns={"index": "Column"})
        st.dataframe(desc, use_container_width=True)
        download_table_block(desc, "eda_summary", "EDA Summary")

        st.markdown("### Missing values")
        miss = pd.DataFrame({
            "Column": df.columns,
            "Missing count": df.isna().sum().values,
            "Missing %": (df.isna().mean().values * 100).round(2),
        }).sort_values("Missing %", ascending=False).reset_index(drop=True)
        st.dataframe(miss, use_container_width=True)
        download_table_block(miss, "eda_missing", "Missing Values")

        st.markdown("### Correlation heatmap (numeric)")
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            corr = num.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr.values)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            ax.set_title("Correlation Heatmap (Numeric)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            download_figure_block(fig, "eda_correlation_heatmap")
            plt.close(fig)

            st.dataframe(corr.round(4), use_container_width=True)
            download_table_block(corr.round(4).reset_index().rename(columns={"index": "Variable"}), "eda_correlation_table", "Correlation Table")
        else:
            st.info("Not enough numeric columns to compute correlations.")

elif main == "Logistic Regression" and sub == "Modeling":
    st.markdown("## Modeling (Logistic Regression)")
    show_dataset_status()

    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)

        st.markdown("### Model setup")
        c1, c2 = st.columns([1, 2])
        with c1:
            target = st.selectbox("Target (binary)", options=cols)
        with c2:
            candidates = [c for c in cols if c != target]
            features = st.multiselect("Predictors", options=candidates)

        c3, c4 = st.columns([1, 1])
        with c3:
            test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
        with c4:
            random_state = st.number_input("Random state", value=42, step=1)

        run = st.button("Run Logistic Regression", type="primary", use_container_width=True)

        if run:
            try:
                res = run_logistic_regression(df, target, features, test_size=float(test_size), random_state=int(random_state))
                st.session_state.last_results["logistic"] = res

                st.markdown("### Variables in the Equation")
                st.dataframe(res["odds_table"], use_container_width=True)
                download_table_block(res["odds_table"], "logistic_variables_in_equation", "Variables in the Equation")

                st.markdown("### ROC Curve")
                fig, ax = plt.subplots()
                ax.plot(res["roc"]["fpr"], res["roc"]["tpr"])
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC Curve (AUC = {res['roc']['auc']:.3f})")
                st.pyplot(fig)
                download_figure_block(fig, "logistic_roc")
                plt.close(fig)

                st.markdown("### Confusion Matrix (threshold = 0.5)")
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(res["cm"])
                disp.plot(ax=ax, colorbar=False)
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                download_figure_block(fig, "logistic_confusion_matrix")
                plt.close(fig)

            except Exception as e:
                st.error(f"Modeling failed: {e}")

elif main == "Logistic Regression" and sub == "Export":
    st.markdown("## Export (Logistic Regression)")
    show_dataset_status()

    if st.session_state.df is not None:
        st.markdown("### Dataset")
        st.download_button(
            "Download dataset as Excel",
            data=df_to_excel_bytes({"dataset": st.session_state.df}),
            file_name="dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    if "logistic" in st.session_state.last_results:
        st.markdown("### Latest Logistic Regression Outputs")
        odds = st.session_state.last_results["logistic"]["odds_table"]
        st.download_button(
            "Download logistic tables (Excel)",
            data=df_to_excel_bytes({"Variables_in_Equation": odds}),
            file_name="logistic_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("No logistic results yet. Run a model first.")

# =========================================================
# Linear pages
# =========================================================
elif main == "Linear Regression" and sub == "Data":
    st.markdown("## Data (Linear Regression)")

    st.markdown("### Templates")
    c1, c2 = st.columns([1, 2])
    with c1:
        tmpl = make_linear_template()
        st.download_button(
            "Download Excel template",
            data=df_to_excel_bytes({"template": tmpl}),
            file_name="linear_regression_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with c2:
        st.info("Upload a CSV/XLSX file using the uploader below. The linear outcome should be numeric (continuous).")

    st.markdown("### Upload CSV/XLSX")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"], label_visibility="collapsed", key="linear_upload")
    df_new = read_uploaded_file(uploaded)
    if df_new is not None:
        st.session_state.df = df_new
        st.session_state.last_results = {}  # reset old results

    show_dataset_status()

elif main == "Linear Regression" and sub == "Modeling":
    st.markdown("## Modeling (Multivariable Linear Regression / OLS)")
    show_dataset_status()

    if st.session_state.df is not None:
        df = st.session_state.df
        cols = list(df.columns)

        st.markdown("### Model setup")
        target = st.selectbox("Outcome (numeric)", options=cols, key="lin_target")

        candidates = [c for c in cols if c != target]
        features = st.multiselect("Predictors", options=candidates, key="lin_features")

        run = st.button("Run Linear Regression (OLS)", type="primary", use_container_width=True)
        if run:
            try:
                res = run_linear_regression(df, target, features)
                st.session_state.last_results["linear"] = res

                st.markdown("### Model fit")
                st.dataframe(res["metrics"], use_container_width=True)
                download_table_block(res["metrics"], "linear_model_fit", "Model Fit")

                st.success("Model fitted successfully. Use the 'ANOVA & Coefficients' and 'Assumptions & Diagnostics' sections for detailed outputs.")
            except Exception as e:
                st.error(f"Modeling failed: {e}")

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

elif main == "Linear Regression" and sub == "Diagnostics":
    st.markdown("## Assumptions & Diagnostics")
    show_dataset_status()

    if "linear" not in st.session_state.last_results:
        st.info("No linear results yet. Run the model in 'Modeling (OLS)' first.")
    else:
        res = st.session_state.last_results["linear"]
        ols = res["model"]
        X = res["X"]
        y = res["y"]

        fitted = ols.fittedvalues
        resid = ols.resid

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
        # QQ plot
        fig = sm.qqplot(resid, line="45")
        plt.title("Q-Q Plot of Residuals")
        st.pyplot(fig)
        download_figure_block(plt.gcf(), "linear_qqplot_residuals")
        plt.close(plt.gcf())

        # Shapiro-Wilk
        stat, p = shapiro(resid)
        shapiro_df = pd.DataFrame([{
            "Test": "Shapiro-Wilk",
            "Statistic": round(float(stat), 4),
            "p-value": format_p_value(float(p))
        }])
        st.dataframe(shapiro_df, use_container_width=True)
        download_table_block(shapiro_df, "linear_shapiro", "Normality Test")

        st.markdown("### Homoscedasticity")
        lm, lm_p, fval, f_p = het_breuschpagan(resid, sm.add_constant(X))
        bp_df = pd.DataFrame([{
            "Test": "Breusch-Pagan",
            "LM Statistic": round(float(lm), 4),
            "LM p-value": format_p_value(float(lm_p)),
            "F Statistic": round(float(fval), 4),
            "F p-value": format_p_value(float(f_p)),
        }])
        st.dataframe(bp_df, use_container_width=True)
        download_table_block(bp_df, "linear_breusch_pagan", "Homoscedasticity")

        st.markdown("### Multicollinearity: VIF")
        X_vif = sm.add_constant(X)
        vif_rows = []
        for i, col in enumerate(X_vif.columns):
            if col == "const":
                continue
            vif_rows.append({
                "Variable": col,
                "VIF": round(float(variance_inflation_factor(X_vif.values, i)), 4)
            })
        vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)
        st.dataframe(vif_df, use_container_width=True)
        download_table_block(vif_df, "linear_vif", "VIF Table")

        st.markdown("### Independence of errors")
        dw = durbin_watson(resid)
        dw_df = pd.DataFrame([{"Test": "Durbin-Watson", "Statistic": round(float(dw), 4)}])
        st.dataframe(dw_df, use_container_width=True)
        download_table_block(dw_df, "linear_durbin_watson", "Independence of Errors")

elif main == "Linear Regression" and sub == "Export":
    st.markdown("## Export (Linear Regression)")
    show_dataset_status()

    if st.session_state.df is not None:
        st.markdown("### Dataset")
        st.download_button(
            "Download dataset as Excel",
            data=df_to_excel_bytes({"dataset": st.session_state.df}),
            file_name="dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    if "linear" in st.session_state.last_results:
        st.markdown("### Latest Linear Regression Outputs")
        coef = st.session_state.last_results["linear"]["coef_table"]
        anova_t = st.session_state.last_results["linear"]["anova_table"]
        metrics = st.session_state.last_results["linear"]["metrics"]

        st.download_button(
            "Download linear tables (Excel)",
            data=df_to_excel_bytes({
                "Model_Fit": metrics,
                "ANOVA": anova_t,
                "Coefficients": coef,
            }),
            file_name="linear_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("No linear results yet. Run a model first.")
