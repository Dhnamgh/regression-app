import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy import stats
from docx import Document


# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Pneumonia Analytics", page_icon="🫁", layout="wide")

st.markdown("""
<style>
/* App width & spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] { width: 320px !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
section[data-testid="stSidebar"] h3 { margin-bottom: .6rem; }

/* Typography */
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(49,51,63,0.65); font-size: 0.95rem; }

/* Card */
.card {
  border: 1px solid rgba(49,51,63,0.12);
  background: rgba(255,255,255,0.9);
  border-radius: 16px;
  padding: 16px 16px 14px 16px;
  box-shadow: 0 1px 10px rgba(0,0,0,0.04);
}
.card h3 { margin: 0 0 6px 0; }
.card .meta { margin-top: 8px; color: rgba(49,51,63,0.65); font-size: 0.92rem; }

/* Buttons */
.stDownloadButton button, .stButton button {
  border-radius: 12px !important;
  padding: .55rem .8rem !important;
  font-weight: 600 !important;
}

/* Dataframe container */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(49,51,63,0.10); }

/* Tabs spacing */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 12px; }

/* Alerts */
.stAlert { border-radius: 14px; }

/* Remove excessive empty space above widgets */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdownContainer"]) { margin-top: .2rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DeLong test (same logic)
# =========================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])
    aucs = tx.sum(axis=1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / n

def delong_roc_test(y_true, y_scores_1, y_scores_2):
    y_true = np.array(y_true)
    order = np.argsort(-y_scores_1)
    y_true = y_true[order]
    y_scores_1 = y_scores_1[order]
    y_scores_2 = y_scores_2[order]
    label_1_count = int(np.sum(y_true))
    preds = np.vstack((y_scores_1, y_scores_2))
    aucs, cov = fastDeLong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = np.abs(diff) / np.sqrt(var)
    return float(2 * (1 - stats.norm.cdf(z)))


# =========================================================
# Data helpers
# =========================================================
REQUIRED_COLS = ["Pneumonia", "CRP", "WBC", "SpO2", "Temperature"]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Vietnamese -> English mapping (keep both acceptable)
    df.rename(columns={
        "Viêm phổi": "Pneumonia",
        "Nhiệt độ": "Temperature",
        "Bạch cầu": "WBC",
        "CRP": "CRP",
        "SpO2": "SpO2",
    }, inplace=True)
    return df

def validate_df(df: pd.DataFrame):
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}. Required: {REQUIRED_COLS}")
    if df["Pneumonia"].dropna().nunique() < 2:
        raise ValueError("Column 'Pneumonia' must contain both classes (0 and 1).")

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

def make_template_excel_bytes() -> bytes:
    template = pd.DataFrame({
        "Pneumonia": [0, 1],
        "CRP": [10.0, 50.0],
        "WBC": [7.5, 14.2],
        "SpO2": [98, 92],
        "Temperature": [36.8, 39.1],
    })
    notes = pd.DataFrame({
        "Notes": [
            "Pneumonia must be 0 (No) or 1 (Yes).",
            "CRP: numeric (e.g., mg/L).",
            "WBC: numeric (e.g., 10^9/L).",
            "SpO2: numeric (0–100).",
            "Temperature: numeric (°C)."
        ]
    })
    return df_to_excel_bytes({"template": template, "notes": notes})


# =========================================================
# Core modeling
# =========================================================
@st.cache_data(show_spinner=False)
def run_models(df: pd.DataFrame, test_size: float, random_state: int, n_bootstraps: int):
    df = standardize_columns(df)
    validate_df(df)

    X = df[["CRP", "WBC", "SpO2", "Temperature"]]
    y = df["Pneumonia"]

    # Statsmodels logistic regression
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
        "p-value": pvals.values,
    }).round(4)
    odds_table["Significant (p<0.05)"] = odds_table["p-value"].apply(lambda p: "Yes" if p < 0.05 else "")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "k-NN": KNeighborsClassifier(),
    }

    probs = {}
    auc_scores = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]
        probs[name] = p
        fpr, tpr, _ = roc_curve(y_test, p)
        auc_scores[name] = auc(fpr, tpr)

    # Bootstrap CI
    def bootstrap_auc_ci(y_true, y_scores, n_bootstraps=200, alpha=0.05):
        rng = np.random.RandomState(42)
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        scores = []
        for _ in range(n_bootstraps):
            idx = rng.randint(0, len(y_scores), len(y_scores))
            if len(np.unique(y_true[idx])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true[idx], y_scores[idx])
            scores.append(auc(fpr, tpr))
        scores = np.array(scores)
        scores.sort()
        lo = scores[int((alpha / 2) * len(scores))]
        hi = scores[int((1 - alpha / 2) * len(scores))]
        return float(lo), float(hi)

    rows = []
    for m in auc_scores:
        lo, hi = bootstrap_auc_ci(y_test.values, probs[m], n_bootstraps=n_bootstraps)
        rows.append({
            "Model": m,
            "AUC": round(auc_scores[m], 4),
            "95% CI Lower": round(lo, 4),
            "95% CI Upper": round(hi, 4),
        })
    auc_df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    top1, top2 = auc_df["Model"].iloc[0], auc_df["Model"].iloc[1]
    p_value = delong_roc_test(y_test.values, probs[top1], probs[top2])

    # ROC figure
    roc_fig = plt.figure(figsize=(8, 5.5))
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_scores[name]:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Reports + confusion matrices (top2)
    reports = {}
    cm_figs = {}
    for m in [top1, top2]:
        y_pred = (probs[m] >= 0.5).astype(int)
        rep = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.round(4)
        reports[m] = rep

        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(5.2, 4.2))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix — {m}")
        plt.tight_layout()
        cm_figs[m] = fig

    return {
        "df": df,
        "odds_table": odds_table,
        "auc_df": auc_df,
        "roc_fig": roc_fig,
        "top1": top1,
        "top2": top2,
        "delong_p": float(p_value),
        "reports": reports,
        "cm_figs": cm_figs,
    }


def build_docx_bytes(result: dict) -> bytes:
    doc = Document()
    doc.add_heading("Pneumonia — Analysis Report", level=1)

    doc.add_heading("AUC Summary", level=2)
    doc.add_paragraph(result["auc_df"].to_string(index=False))

    doc.add_heading("Logistic Regression (Odds Ratios)", level=2)
    doc.add_paragraph(result["odds_table"].to_string(index=False))

    doc.add_heading("DeLong Test", level=2)
    doc.add_paragraph(f"Top1: {result['top1']} | Top2: {result['top2']} | p-value: {result['delong_p']:.6f}")

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()


# =========================================================
# UI helpers
# =========================================================
def download_table(df: pd.DataFrame, base_name: str, title: str = ""):
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

def download_figure(fig: plt.Figure, base_name: str):
    st.download_button(
        "Download PNG",
        data=fig_to_png_bytes(fig),
        file_name=f"{base_name}.png",
        mime="image/png",
        use_container_width=False
    )


# =========================================================
# Sidebar (compact + professional)
# =========================================================
with st.sidebar:
    st.markdown("### Navigation")
    section = st.radio(
        "Navigation",
        ["Data", "Explore (EDA)", "Logistic (OR)", "Model Comparison", "Reports", "Export"],
        index=0
    )


    st.divider()

    st.markdown("### Data")
    st.download_button(
        "Download Excel template",
        data=make_template_excel_bytes(),
        file_name="pneumonia_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"], label_visibility="visible")

    with st.expander("Model settings", expanded=False):
        test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)
        n_bootstraps = st.slider("Bootstrap iterations (AUC CI)", 100, 500, 200, 50)

    st.caption("Tip: Keep the template columns unchanged.")


# =========================================================
# Load data
# =========================================================
def load_dataframe(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return standardize_columns(df)

df = load_dataframe(uploaded)


# =========================================================
# Header
# =========================================================
st.markdown("""
<div class="card">
  <h1 style="margin:0;">🫁 Pneumonia Analytics</h1>
  <div class="small-muted">Upload data, explore EDA, run logistic regression, and compare ML models.</div>
  <div class="meta">Recommended: use the Excel template to keep column names consistent.</div>
</div>
""", unsafe_allow_html=True)
st.write("")



# =========================================================
# Content per section
# =========================================================
if section == "Data":
    st.subheader("Data")
    if df is None:
        st.info("Upload a CSV/XLSX file from the sidebar to get started.")
        st.stop()

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Preview")
        st.dataframe(df.head(30), use_container_width=True)
        download_table(df.head(200).reset_index(drop=True), base_name="data_preview", title="Data Preview (first rows)")
    with right:
        st.markdown("#### Quick checks")
        info = pd.DataFrame({
            "Rows": [df.shape[0]],
            "Columns": [df.shape[1]],
            "Missing values": [int(df.isna().sum().sum())],
            "Duplicate rows": [int(df.duplicated().sum())],
        })
        st.dataframe(info, use_container_width=True)
        try:
            validate_df(df)
            st.success("Dataset is valid for modeling.")
        except Exception as e:
            st.error(str(e))

elif section == "Explore (EDA)":
    st.subheader("Explore (EDA)")
    if df is None:
        st.info("Upload a CSV/XLSX file from the sidebar to get started.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Summary", "Distributions", "Correlations"])

    with tab1:
        st.markdown("#### Descriptive statistics")
        desc = df[["CRP", "WBC", "SpO2", "Temperature"]].describe().T.round(4).reset_index(names="Feature")
        st.dataframe(desc, use_container_width=True)
        download_table(desc, "eda_descriptive_stats", "Descriptive Statistics")

        st.markdown("#### Missing values by column")
        miss = df.isna().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        st.dataframe(miss, use_container_width=True)
        download_table(miss, "eda_missing_by_column", "Missing Values by Column")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure(figsize=(6.2, 4.2))
            ax = fig.add_subplot(111)
            df["Pneumonia"].value_counts(dropna=False).sort_index().plot(kind="bar", ax=ax)
            ax.set_title("Class distribution (Pneumonia)")
            ax.set_xlabel("Pneumonia")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=False)
            download_figure(fig, "eda_class_distribution")
        with c2:
            feature = st.selectbox("Feature", ["CRP", "WBC", "SpO2", "Temperature"])
            fig2 = plt.figure(figsize=(6.2, 4.2))
            ax2 = fig2.add_subplot(111)
            sns.histplot(data=df, x=feature, hue="Pneumonia", kde=True, ax=ax2)
            ax2.set_title(f"Distribution — {feature}")
            plt.tight_layout()
            st.pyplot(fig2, clear_figure=False)
            download_figure(fig2, f"eda_hist_{feature}".lower())

    with tab3:
        corr = df[["CRP", "WBC", "SpO2", "Temperature", "Pneumonia"]].corr(numeric_only=True).round(4)
        fig3 = plt.figure(figsize=(7, 5))
        ax3 = fig3.add_subplot(111)
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax3)
        ax3.set_title("Correlation heatmap")
        plt.tight_layout()
        st.pyplot(fig3, clear_figure=False)
        download_figure(fig3, "eda_correlation_heatmap")

        corr_table = corr.reset_index(names="Feature")
        st.dataframe(corr_table, use_container_width=True)
        download_table(corr_table, "eda_correlation_table", "Correlation Table")

elif section in ["Logistic (OR)", "Model Comparison", "Reports", "Export"]:
    if df is None:
        st.info("Upload a CSV/XLSX file from the sidebar to get started.")
        st.stop()
    try:
        validate_df(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.markdown("#### Run analysis")

bar1, bar2, bar3 = st.columns([1.2, 1, 1])

with bar1:
    st.markdown(
        '<div class="small-muted">'
        'Compute results on demand to keep the app responsive.'
        '</div>',
        unsafe_allow_html=True
    )

with bar2:
    run = st.button(
        "▶ Run analysis",
        type="primary",
        use_container_width=True
    )

with bar3:
    st.download_button(
        "Download Excel template",
        data=make_template_excel_bytes(),
        file_name="pneumonia_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

if not run:
    st.info("Click **Run analysis** to start model training and evaluation.")
    st.stop()


    with st.spinner("Running models..."):
        result = run_models(df, test_size=float(test_size), random_state=int(random_state), n_bootstraps=int(n_bootstraps))

    if section == "Logistic (OR)":
        st.subheader("Logistic Regression — Odds Ratios")
        st.dataframe(result["odds_table"], use_container_width=True)
        download_table(result["odds_table"], "logistic_odds_ratios", "Logistic Regression — Odds Ratios")

    elif section == "Model Comparison":
        st.subheader("Model Comparison")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("#### AUC with 95% CI")
            st.dataframe(result["auc_df"], use_container_width=True)
            download_table(result["auc_df"], "auc_summary", "AUC Summary")
        with c2:
            st.markdown("#### DeLong test (Top 2)")
            delong_df = pd.DataFrame([{
                "Top 1": result["top1"],
                "Top 2": result["top2"],
                "p-value": round(result["delong_p"], 6)
            }])
            st.dataframe(delong_df, use_container_width=True)
            download_table(delong_df, "delong_test", "DeLong Test")

        st.markdown("#### ROC curve")
        st.pyplot(result["roc_fig"], clear_figure=False)
        download_figure(result["roc_fig"], "roc_curve")

    elif section == "Reports":
        st.subheader("Reports (Top 2 models)")
        for m in [result["top1"], result["top2"]]:
            st.markdown(f"### {m}")
            left, right = st.columns([1.2, 1])
            with left:
                st.markdown("**Classification report**")
                rep = result["reports"][m]
                st.dataframe(rep, use_container_width=True)
                download_table(rep.reset_index(names="Metric"), f"report_{m}".replace(" ", "_").lower(), f"Classification Report — {m}")
            with right:
                st.markdown("**Confusion matrix**")
                fig_cm = result["cm_figs"][m]
                st.pyplot(fig_cm, clear_figure=False)
                download_figure(fig_cm, f"confusion_matrix_{m}".replace(" ", "_").lower())

    elif section == "Export":
        st.subheader("Export")
        st.markdown("Download a complete report package.")

        excel_bytes = df_to_excel_bytes({
            "data": result["df"],
            "auc_summary": result["auc_df"],
            "odds_ratios": result["odds_table"],
            "delong": pd.DataFrame([{"Top 1": result["top1"], "Top 2": result["top2"], "p-value": result["delong_p"]}]),
        })
        st.download_button(
            "Download Excel report",
            data=excel_bytes,
            file_name="pneumonia_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        docx_bytes = build_docx_bytes(result)
        st.download_button(
            "Download Word report",
            data=docx_bytes,
            file_name="pneumonia_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )


# =========================================================
# Footer note (clean)
# =========================================================
st.caption("Upload CSV/XLSX • Use the template for best results.")

