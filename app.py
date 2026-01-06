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


# ======================
# Page config
# ======================
st.set_page_config(page_title="Pneumonia App", layout="wide")


# ======================
# DeLong test functions (giữ như bạn)
# ======================
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
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = compute_midrank(positive_examples[r])
        ty[r] = compute_midrank(negative_examples[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])
    aucs = tx.sum(axis=1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_roc_test(y_true, y_scores_1, y_scores_2):
    y_true = np.array(y_true)
    order = np.argsort(-y_scores_1)
    y_true = y_true[order]
    y_scores_1 = y_scores_1[order]
    y_scores_2 = y_scores_2[order]
    label_1_count = int(np.sum(y_true))
    predictions = np.vstack((y_scores_1, y_scores_2))
    aucs, delongcov = fastDeLong(predictions, label_1_count)
    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = np.abs(diff) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(z))
    return float(p)


# ======================
# Data helpers
# ======================
REQUIRED_COLS = ["Pneumonia", "CRP", "WBC", "SpO2", "Temperature"]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={
        "Viêm phổi": "Pneumonia",
        "Nhiệt độ": "Temperature",
        "Bạch cầu": "WBC",
        "CRP": "CRP",
        "SpO2": "SpO2"
    }, inplace=True)
    return df

def validate_df(df: pd.DataFrame):
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(list(missing))}. Cần đủ: {REQUIRED_COLS}")
    if df["Pneumonia"].dropna().nunique() < 2:
        raise ValueError("Cột Pneumonia phải có đủ 2 lớp (0/1).")

def make_template_excel_bytes() -> bytes:
    tmpl = pd.DataFrame({
        "Pneumonia": [0, 1],
        "CRP": [10.0, 50.0],
        "WBC": [7.5, 14.2],
        "SpO2": [98, 92],
        "Temperature": [36.8, 39.1],
    })
    return df_to_excel_bytes({"template": tmpl})

def df_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    bio.seek(0)
    return bio.getvalue()

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 200) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
    bio.seek(0)
    return bio.getvalue()

def df_to_png_bytes(df: pd.DataFrame, title: str = "", dpi: int = 200) -> bytes:
    # Render dataframe thành ảnh (matplotlib table)
    n_rows, n_cols = df.shape
    fig_w = min(18, max(6, n_cols * 1.6))
    fig_h = min(18, max(2.5, (n_rows + 1) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=10)

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)

    png = fig_to_png_bytes(fig, dpi=dpi)
    plt.close(fig)
    return png


# ======================
# Core analysis
# ======================
@st.cache_data(show_spinner=False)
def run_models(df: pd.DataFrame, test_size: float, random_state: int):
    df = standardize_columns(df)
    validate_df(df)

    X = df[["CRP", "WBC", "SpO2", "Temperature"]]
    y = df["Pneumonia"]

    # statsmodels logistic
    X_sm = sm.add_constant(X)
    logit_model = sm.Logit(y, X_sm).fit(disp=False)

    params = logit_model.params
    conf = logit_model.conf_int()
    pvals = logit_model.pvalues

    odds_table = pd.DataFrame({
        "Coefficient (β)": params,
        "Odds Ratio": np.exp(params),
        "CI 2.5%": np.exp(conf[0]),
        "CI 97.5%": np.exp(conf[1]),
        "p-value": pvals
    }).round(4)
    odds_table["Significant(p<0.05)"] = odds_table["p-value"].apply(lambda p: "Yes" if p < 0.05 else "")

    # ML models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "k-NN": KNeighborsClassifier()
    }

    probs_dict = {}
    auc_scores = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        probs_dict[name] = probs

        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_scores[name] = auc(fpr, tpr)

    # ROC figure
    roc_fig = plt.figure(figsize=(8, 6))
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probs_dict[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_scores[name]:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Bootstrap CI
    def bootstrap_auc_ci(y_true, y_scores, n_bootstraps=1000, alpha=0.05):
        rng = np.random.RandomState(42)
        boot_scores = []
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        for _ in range(n_bootstraps):
            idx = rng.randint(0, len(y_scores), len(y_scores))
            if len(np.unique(y_true[idx])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true[idx], y_scores[idx])
            boot_scores.append(auc(fpr, tpr))
        boot_scores = np.array(boot_scores)
        boot_scores.sort()
        lower = boot_scores[int((alpha / 2) * len(boot_scores))]
        upper = boot_scores[int((1 - alpha / 2) * len(boot_scores))]
        return float(lower), float(upper)

    rows = []
    ci_results = {}
    for m in auc_scores:
        lo, hi = bootstrap_auc_ci(y_test.values, probs_dict[m])
        ci_results[m] = (round(lo, 4), round(hi, 4))
        rows.append({
            "Model": m,
            "AUC": round(auc_scores[m], 4),
            "95% CI Lower": ci_results[m][0],
            "95% CI Upper": ci_results[m][1],
        })
    auc_df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    # DeLong (top2)
    top = auc_df["Model"].tolist()
    model1, model2 = top[0], top[1]
    p_value = delong_roc_test(y_test.values, probs_dict[model1], probs_dict[model2])

    # Reports + Confusion matrices (top2)
    reports = {}
    cm_figs = {}
    for m in [model1, model2]:
        y_pred = (probs_dict[m] >= 0.5).astype(int)
        reports[m] = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.round(4)

        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(5.2, 4.2))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Pneumonia", "Pneumonia"])
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix: {m}")
        plt.tight_layout()
        cm_figs[m] = fig

    return {
        "df": df,
        "odds_table": odds_table,
        "auc_df": auc_df,
        "roc_fig": roc_fig,
        "top1": model1,
        "top2": model2,
        "p_value": p_value,
        "reports": reports,
        "cm_figs": cm_figs,
    }


def build_docx_bytes(result: dict) -> bytes:
    doc = Document()
    doc.add_heading("Pneumonia Classification Report", level=1)
    doc.add_paragraph("Generated by Streamlit app.")

    doc.add_heading("AUC Summary", level=2)
    doc.add_paragraph(result["auc_df"].to_string(index=False))

    doc.add_heading("Logistic Regression Odds Ratio", level=2)
    doc.add_paragraph(result["odds_table"].to_string())

    doc.add_heading("DeLong Test (Top 2 models)", level=2)
    doc.add_paragraph(f"Top1: {result['top1']}, Top2: {result['top2']}, p-value={result['p_value']:.6f}")

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()


# ======================
# UI Components
# ======================
def download_table_block(df: pd.DataFrame, base_name: str, title: str = ""):
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "⬇️ Tải bảng (Excel)",
            data=df_to_excel_bytes({base_name: df}),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with colB:
        st.download_button(
            "⬇️ Tải bảng (PNG)",
            data=df_to_png_bytes(df, title=title),
            file_name=f"{base_name}.png",
            mime="image/png",
            use_container_width=True
        )

def download_figure_block(fig: plt.Figure, base_name: str):
    st.download_button(
        "⬇️ Tải hình (PNG)",
        data=fig_to_png_bytes(fig),
        file_name=f"{base_name}.png",
        mime="image/png",
        use_container_width=False
    )


# ======================
# App State: load data
# ======================
st.title("🫁 Pneumonia – Data, EDA, Modeling")

with st.sidebar:
    st.header("📁 Dữ liệu")
    st.caption("Bạn có thể upload CSV hoặc Excel. Có sẵn form mẫu Excel để nhập.")

    template_bytes = make_template_excel_bytes()
    st.download_button(
        "📄 Tải form mẫu Excel",
        data=template_bytes,
        file_name="pneumonia_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    up = st.file_uploader("Upload dữ liệu (CSV/XLSX)", type=["csv", "xlsx"])

    st.divider()
    st.header("🧭 Menu")
    menu = st.radio(
        "Chọn chức năng",
        [
            "1) Nạp dữ liệu",
            "2) Khám phá dữ liệu (EDA)",
            "3) Hồi quy Logistic (Odds Ratio)",
            "4) So sánh mô hình (AUC/ROC/DeLong)",
            "5) Confusion matrix & Report (Top 2)",
            "6) Xuất báo cáo (Word/Excel)"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.header("⚙️ Cài đặt mô hình")
    test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)


def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        # excel
        df = pd.read_excel(uploaded_file)
    df = standardize_columns(df)
    return df


df = load_dataframe(up)


# ======================
# Menu: 1) Load data
# ======================
if menu == "1) Nạp dữ liệu":
    st.subheader("1) Nạp dữ liệu")

    if df is None:
        st.info("⬅️ Hãy upload file CSV/XLSX ở sidebar. Hoặc tải form mẫu Excel, nhập dữ liệu rồi upload lại.")
        st.stop()

    st.success(f"Đã nạp dữ liệu: {up.name} | Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")
    st.write("Yêu cầu cột:", REQUIRED_COLS)

    st.dataframe(df.head(30), use_container_width=True)
    download_table_block(df, base_name="data_preview", title="Data Preview (top rows)")

    try:
        validate_df(df)
        st.success("✅ Dữ liệu hợp lệ để chạy mô hình.")
    except Exception as e:
        st.error(f"❌ Dữ liệu chưa hợp lệ: {e}")


# ======================
# Menu: 2) EDA
# ======================
elif menu == "2) Khám phá dữ liệu (EDA)":
    st.subheader("2) Khám phá dữ liệu (EDA)")
    if df is None:
        st.info("⬅️ Upload dữ liệu trước.")
        st.stop()

    # Basic checks
    st.markdown("### Tổng quan")
    overview = pd.DataFrame({
        "Rows": [df.shape[0]],
        "Columns": [df.shape[1]],
        "Missing values (total)": [int(df.isna().sum().sum())],
        "Duplicate rows": [int(df.duplicated().sum())]
    })
    st.dataframe(overview, use_container_width=True)
    download_table_block(overview, base_name="eda_overview", title="EDA Overview")

    st.markdown("### Thống kê mô tả")
    desc = df[["CRP", "WBC", "SpO2", "Temperature"]].describe().T.round(4).reset_index(names="Feature")
    st.dataframe(desc, use_container_width=True)
    download_table_block(desc, base_name="eda_describe", title="Descriptive Statistics")

    st.markdown("### Missing theo cột")
    miss = df.isna().sum().reset_index()
    miss.columns = ["Column", "Missing"]
    st.dataframe(miss, use_container_width=True)
    download_table_block(miss, base_name="eda_missing", title="Missing per column")

    st.markdown("### Phân bố theo Pneumonia")
    fig1 = plt.figure(figsize=(6.5, 4.0))
    ax = fig1.add_subplot(111)
    df["Pneumonia"].value_counts(dropna=False).sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Class distribution (Pneumonia)")
    ax.set_xlabel("Pneumonia")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig1, clear_figure=False)
    download_figure_block(fig1, "eda_class_distribution")

    st.markdown("### Correlation heatmap")
    corr = df[["CRP", "WBC", "SpO2", "Temperature", "Pneumonia"]].corr(numeric_only=True).round(4)
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(111)
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax2)
    ax2.set_title("Correlation heatmap")
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=False)
    download_figure_block(fig2, "eda_corr_heatmap")
    st.dataframe(corr, use_container_width=True)
    download_table_block(corr.reset_index(names="Feature"), base_name="eda_corr_table", title="Correlation Table")


# ======================
# Menu: 3-6 need model results
# ======================
else:
    if df is None:
        st.info("⬅️ Upload dữ liệu trước.")
        st.stop()

    try:
        validate_df(df)
    except Exception as e:
        st.error(f"❌ Dữ liệu chưa hợp lệ để chạy mô hình: {e}")
        st.stop()

    with st.spinner("Đang chạy mô hình..."):
        result = run_models(df, test_size=float(test_size), random_state=int(random_state))

    # 3) Logistic OR
    if menu == "3) Hồi quy Logistic (Odds Ratio)":
        st.subheader("3) Hồi quy Logistic (Odds Ratio)")
        st.dataframe(result["odds_table"], use_container_width=True)
        download_table_block(result["odds_table"].reset_index(names="Term"), base_name="logistic_odds_ratio", title="Logistic Odds Ratio")

    # 4) ROC / AUC / DeLong
    elif menu == "4) So sánh mô hình (AUC/ROC/DeLong)":
        st.subheader("4) So sánh mô hình (AUC/ROC/DeLong)")

        st.markdown("### AUC + 95% CI")
        st.dataframe(result["auc_df"], use_container_width=True)
        download_table_block(result["auc_df"], base_name="auc_summary", title="AUC Summary")

        st.markdown("### ROC Curve")
        st.pyplot(result["roc_fig"], clear_figure=False)
        download_figure_block(result["roc_fig"], "roc_curve")

        st.markdown("### DeLong test (Top 2)")
        delong_df = pd.DataFrame([{
            "Top 1": result["top1"],
            "Top 2": result["top2"],
            "p-value": round(result["p_value"], 6)
        }])
        st.dataframe(delong_df, use_container_width=True)
        download_table_block(delong_df, base_name="delong_test", title="DeLong Test")

    # 5) Confusion matrix & report
    elif menu == "5) Confusion matrix & Report (Top 2)":
        st.subheader("5) Confusion matrix & Report (Top 2)")

        for m in [result["top1"], result["top2"]]:
            st.markdown(f"### {m}")

            st.markdown("**Classification report**")
            rep = result["reports"][m]
            st.dataframe(rep, use_container_width=True)
            download_table_block(rep.reset_index(names="Metric"), base_name=f"classification_report_{m}".replace(" ", "_"), title=f"Classification report - {m}")

            st.markdown("**Confusion matrix**")
            fig_cm = result["cm_figs"][m]
            st.pyplot(fig_cm, clear_figure=False)
            download_figure_block(fig_cm, f"confusion_matrix_{m}".replace(" ", "_"))

    # 6) Export report
    elif menu == "6) Xuất báo cáo (Word/Excel)":
        st.subheader("6) Xuất báo cáo (Word/Excel)")

        st.markdown("### Export Excel (nhiều sheet)")
        excel_bytes = df_to_excel_bytes({
            "data": result["df"],
            "auc_summary": result["auc_df"],
            "odds_ratio": result["odds_table"].reset_index(names="Term"),
            "delong": pd.DataFrame([{
                "Top 1": result["top1"], "Top 2": result["top2"], "p-value": result["p_value"]
            }]),
        })
        st.download_button(
            "⬇️ Tải báo cáo Excel",
            data=excel_bytes,
            file_name="Pneumonia_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.markdown("### Export Word")
        docx_bytes = build_docx_bytes(result)
        st.download_button(
            "⬇️ Tải báo cáo Word",
            data=docx_bytes,
            file_name="Pneumonia_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
