import io
import json
import math
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multicomp import pairwise_tukeyhsd

try:
  from statsmodels.stats.diagnostic import het_breuschpagan, lilliefors
except ImportError:
  lilliefors = None
  het_breuschpagan = None

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="Data Analysis in Health Sciences", page_icon="📊", layout="wide"
)

# =========================================================
# CSS Giao diện: Tinh chỉnh chữ to, đậm, rõ nét như Navigation
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
  font-weight: 800 !important;
  font-size: 17px !important;
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
  font-size: 18px !important;
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
  font-weight: 900;
}
.header-banner p{
  margin: 8px 0 0 0;
  opacity: 0.95;
  font-size: 17px;
  font-weight: 600;
}

/* TIÊU ĐỀ H2, H3, H4 */
h2 {
  font-size: 28px !important;
  font-weight: 900 !important;
  color: #0B3A66 !important;
}
h3 {
  font-size: 23px !important;
  font-weight: 800 !important;
  color: #0B3A66 !important;
  margin-top: 15px !important;
}
h4 {
  font-size: 20px !important;
  font-weight: 800 !important;
  color: #0f172a !important;
  margin-top: 12px !important;
}

/* NHÃN CỦA TẤT CẢ WIDGET */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
  font-size: 19px !important;
  font-weight: 800 !important;
  color: #0f172a !important;
  line-height: 1.3 !important;
}

/* TÙY CHỌN RADIO */
div[data-testid="stRadio"] div[role="radiogroup"] label,
div[data-testid="stRadio"] div[role="radiogroup"] label p,
div[data-testid="stRadio"] div[role="radiogroup"] span {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

/* TÙY CHỌN CHECKBOX */
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] span {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

/* VĂN BẢN VÀ CHÚ THÍCH */
.stMarkdown p {
  font-size: 17px !important;
  font-weight: 600 !important;
  color: #0f172a !important;
}
.stCaption, [data-testid="stCaptionContainer"] p {
  font-size: 16px !important;
  font-weight: 700 !important;
  color: #334155 !important;
}

/* CÁC Ô NHẬP LIỆU */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
  border-width: 1.5px !important;
}

div[data-baseweb="select"] * {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

/* BẢNG KẾT QUẢ */
.analysis-table-wrap{
  width: 100%;
  overflow-x: auto;
  margin: 0.6rem 0 1.3rem 0;
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
  padding: 12px 15px;
  text-align: left;
  line-height: 1.25;
}
.analysis-table td{
  border: 1.2px solid #e2e8f0;
  padding: 12px 15px;
  color: #0f172a;
  font-weight: 500;
  line-height: 1.35;
}
.analysis-table tbody tr:nth-child(even){
  background: #f8fafc;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header-banner">
  <h1>Data Analysis in Health Sciences</h1>
  <p>Regression, categorical analysis, quantitative tests and diagnostics for health sciences.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# =========================================================
# Thuật toán 2 bước tính phân vị giáo trình
# =========================================================
def compute_percentile_textbook(arr: np.ndarray, p: float) -> float:
  s = np.sort(np.asarray(arr, dtype=float))
  n = len(s)
  k = n * (p / 100.0)

  if abs(k - round(k)) < 1e-9:
    idx = int(round(k))
    return float((s[idx - 1] + s[idx]) / 2.0)
  else:
    idx = int(math.ceil(k))
    return float(s[idx - 1])


# =========================================================
# Quy tắc làm tròn số thông minh
# =========================================================
def smart_round_val(val, min_dec: int = 3) -> str:
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
  if p < 0.001:
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
# Xuất dữ liệu & Nút chép Excel
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
    if not text or text == "-":
      return 1
    lines = textwrap.wrap(str(text), width=width)
    return len(lines) if len(lines) > 0 else 1

  header_max_lines = max(
      [get_line_count(col, width_limit) for col in df_plot.columns]
  )
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
  ax.axis("off")

  wrapped_headers = [
      textwrap.fill(str(col), width=width_limit) for col in df_plot.columns
  ]
  wrapped_data = []
  for _, row in df_plot.iterrows():
    wrapped_data.append(
        [textwrap.fill(str(val), width=width_limit) for val in row]
    )

  table = ax.table(
      cellText=wrapped_data,
      colLabels=wrapped_headers,
      cellLoc="center",
      loc="center",
  )
  table.auto_set_font_size(False)
  table.set_fontsize(13)

  for i in range(n_cols):
    table[0, i].set_height(h_height / fig_height)
    for j in range(n_rows):
      cell_h = (row_line_counts[j] * line_unit_height + padding) / fig_height
      table[j + 1, i].set_height(cell_h)

  for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#94a3b8")
    cell.set_linewidth(1.0)
    if row == 0:
      cell.set_text_props(weight="bold")
      cell.set_facecolor("#e2e8f0")
    else:
      cell.set_facecolor("white")

  if title:
    plt.title(title, fontsize=18, pad=40, weight="bold")

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
        on_click="ignore",
    )
  with c2:
    st.download_button(
        "Download PNG",
        data=df_to_png_bytes(df, title=title),
        file_name=f"{base_name}.png",
        mime="image/png",
        use_container_width=True,
        key=f"dl_png_{base_name}_{abs(hash(base_name)) % 100000}",
        on_click="ignore",
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
      on_click="ignore",
  )


def show_table(df: pd.DataFrame, title: str):
  st.markdown(f"### {title}")
  display_df = df.copy().fillna("")
  html = display_df.to_html(index=False, escape=True, classes="analysis-table")
  st.markdown(
      f'<div class="analysis-table-wrap">{html}</div>', unsafe_allow_html=True
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


def data_input_panel(
    template_df: pd.DataFrame,
    template_name: str,
    store_key: str,
    store_name_key: str,
    help_text: str = "",
):
  with st.expander("Data input (template + upload)", expanded=True):
    c1, c2 = st.columns([1, 2])
    with c1:
      st.download_button(
          "Download Excel template",
          data=df_to_excel_bytes({template_name: template_df}),
          file_name=f"{template_name}.xlsx",
          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          use_container_width=True,
      )
      if help_text:
        st.caption(help_text)

    with c2:
      up = st.file_uploader(
          "Upload CSV/XLSX",
          type=["csv", "xlsx", "xls"],
          key=f"uploader_{template_name}",
      )
      if up is not None:
        df = load_uploaded_file(up)
        st.session_state[store_key] = df
        st.session_state[store_name_key] = up.name
        st.success(f"Loaded: {up.name} • shape={df.shape}")

  df = st.session_state.get(store_key)
  if isinstance(df, pd.DataFrame) and not df.empty:
    st.caption(
        f"Current dataset: {st.session_state.get(store_name_key,'')} •"
        f" {df.shape}"
    )
    st.dataframe(df.head(30), use_container_width=True)


def require_df(store_key: str) -> pd.DataFrame:
  df = st.session_state.get(store_key)
  if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("No dataset loaded on this page yet. Use the Data input panel.")
    raise RuntimeError("No dataset")
  return df


# =========================================================
# Ước lượng Khoảng tin cậy (CI)
# =========================================================
def ci_combined_estimates(
    x: np.ndarray,
    alpha: float = 0.05,
    use_bootstrap: bool = False,
    n_boot: int = 5000,
    seed: int = 123,
) -> pd.DataFrame:
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


def ci_from_summary_stats(
    n: int, mean_val: float, s_val: float, s2_val: float, alpha: float = 0.05
) -> pd.DataFrame:
  lo_pct = alpha / 2 * 100
  hi_pct = (1 - alpha / 2) * 100
  lo_label = f"CI {lo_pct:.1f}%" if lo_pct % 1 != 0 else f"CI {lo_pct:.0f}%"
  hi_label = f"CI {hi_pct:.1f}%" if hi_pct % 1 != 0 else f"CI {hi_pct:.0f}%"

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

  df_res = pd.DataFrame([
      ["Mean", mean_val, mean_lo, mean_hi],
      ["Std. Deviation", s_val, sd_lo, sd_hi],
      ["Variance", s2_val, var_lo, var_hi],
  ], columns=["Parameter", "Estimate", lo_label, hi_label])

  return compact_numeric_df(df_res, decimals=3)


# =========================================================
# Sidebar & Điều hướng
# =========================================================
if "section" not in st.session_state:
  st.session_state.section = "Confidence Intervals"
if "sub" not in st.session_state:
  st.session_state.sub = "Mean & Variance"


def set_nav(section: str, sub: str):
  st.session_state.section = section
  st.session_state.sub = sub


with st.sidebar:
  st.markdown("## Navigation")

  if st.button("Home", use_container_width=True):
    set_nav("Home", "Overview")

  with st.expander(
      "Confidence Intervals",
      expanded=(st.session_state.section == "Confidence Intervals"),
  ):
    if st.button("Mean, SD & Variance CI", key="ci_1", use_container_width=True):
      set_nav("Confidence Intervals", "Mean & Variance")
    if st.button("Proportion CI", key="ci_prop", use_container_width=True):
      set_nav("Confidence Intervals", "Proportion")

# =========================================================
# Trang: CONFIDENCE INTERVALS — MEAN, SD & VARIANCE
# =========================================================
section = st.session_state.section
sub = st.session_state.sub

if section == "Confidence Intervals" and sub == "Mean & Variance":
  st.markdown("## Confidence Intervals — Mean, SD & Variance")

  template = pd.DataFrame({"X": [1.2, 2.0, 1.8, 2.2, 1.6]})
  st.download_button(
      "Download Excel template",
      data=df_to_excel_bytes({"ci_template": template}),
      file_name="ci_template.xlsx",
      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      use_container_width=False,
  )

  method = st.radio(
      "Input method",
      [
          "Upload file (template)",
          "Paste values",
          "Enter summary statistics (n, Mean, s/s²)",
      ],
      horizontal=True,
  )

  x = None
  summary_params = None

  if method == "Upload file (template)":
    up = st.file_uploader(
        "Upload CI template (XLSX/CSV)", type=["xlsx", "csv"], key="ci_upload"
    )
    if up is not None:
      df = load_uploaded_file(up)
      st.dataframe(df.head(50), use_container_width=True)
      if "X" not in df.columns:
        st.error("Template must have a column named 'X'.")
      else:
        x = pd.to_numeric(df["X"], errors="coerce").dropna().values
  elif method == "Paste values":
    txt = st.text_area(
        "Paste numeric values (separated by ; , space or newline)",
        value=(
            "12; 14; 16; 18; 20; 22; 24; 26; 28; 30; 32; 34; 36; 38; 40; 42;"
            " 44; 46; 48; 50"
        ),
        height=100,
    )
    if txt.strip():
      normalized = txt.replace(";", " ").replace(",", " ")
      parts = [p for p in normalized.split() if p.strip()]
      vals = pd.to_numeric(pd.Series(parts), errors="coerce").dropna()
      x = vals.values
  else:
    # Nhập số liệu tóm tắt (n, Mean, s hoặc s²)
    st.markdown("#### Nhập tham số thống kê mẫu")
    c1, c2 = st.columns(2)
    with c1:
      n_input = st.number_input(
          "Cỡ mẫu (n)", min_value=2, value=20, step=1, key="ci_sum_n"
      )
      mean_input = st.number_input(
          "Trung bình mẫu (Mean, x̄)",
          value=31.000,
          format="%.4f",
          key="ci_sum_mean",
      )
    with c2:
      disp_choice = st.radio(
          "Chọn tham số độ phân tán để nhập:",
          ["Độ lệch chuẩn (s)", "Phương sai (s²)"],
          horizontal=True,
          key="ci_sum_disp_choice",
      )
      if disp_choice == "Độ lệch chuẩn (s)":
        s_input = st.number_input(
            "Độ lệch chuẩn mẫu (s)",
            min_value=0.0001,
            value=11.832,
            format="%.4f",
            key="ci_sum_s",
        )
        var_input = s_input**2
        st.caption(f"Phương sai tương ứng ($s^2$): **{var_input:.4f}**")
      else:
        var_input = st.number_input(
            "Phương sai mẫu (s²)",
            min_value=0.0001,
            value=140.000,
            format="%.4f",
            key="ci_sum_var",
        )
        s_input = math.sqrt(var_input)
        st.caption(f"Độ lệch chuẩn tương ứng ($s$): **{s_input:.4f}**")

    summary_params = {
        "n": int(n_input),
        "mean": float(mean_input),
        "s": float(s_input),
        "s2": float(var_input),
    }

  # Chọn độ tin cậy
  st.markdown("#### Độ tin cậy (Confidence level)")
  c_conf1, c_conf2 = st.columns([1, 1])
  with c_conf1:
    conf_choice = st.radio(
        "Chọn mức tin cậy:",
        ["95%", "99%", "Khác..."],
        index=0,
        horizontal=True,
        key="ci_conf_choice",
    )
  with c_conf2:
    if conf_choice == "95%":
      conf_level = 0.95
    elif conf_choice == "99%":
      conf_level = 0.99
    else:
      conf_level = st.slider(
          "Nhập mức tin cậy tùy chỉnh",
          min_value=0.80,
          max_value=0.999,
          value=0.90,
          step=0.005,
          format="%.3f",
          key="ci_conf_custom",
      )

  alpha_tail = 1.0 - conf_level

  if method in ["Upload file (template)", "Paste values"]:
    c_boot1, c_boot2 = st.columns(2)
    with c_boot1:
      force_boot = st.checkbox(
          "Force bootstrap (recommended if non-normal)",
          value=False,
          key="ci_force_boot",
      )
    with c_boot2:
      n_boot = st.number_input(
          "Bootstrap resamples",
          min_value=1000,
          max_value=20000,
          value=5000,
          step=500,
          key="ci_n_boot",
      )
  else:
    force_boot = False
    n_boot = 5000

  st.markdown("### Confidence Interval Results")
  if st.button("Compute CI", type="primary", use_container_width=True):
    if method == "Enter summary statistics (n, Mean, s/s²)":
      try:
        n_v = summary_params["n"]
        m_v = summary_params["mean"]
        s_v = summary_params["s"]
        s2_v = summary_params["s2"]
        se_v = s_v / math.sqrt(n_v)

        desc_summary_df = pd.DataFrame([{
            "n": n_v,
            "Mean": m_v,
            "s": s_v,
            "s²": s2_v,
            "Std. Error (SE)": se_v,
        }])
        desc_summary_df = compact_numeric_df(desc_summary_df, decimals=3)
        show_table(desc_summary_df, "Sample Summary Statistics")
        download_table_block(
            desc_summary_df, "ci_summary_statistics", "Sample Summary Statistics"
        )

        ci_table = ci_from_summary_stats(
            n=n_v, mean_val=m_v, s_val=s_v, s2_val=s2_v, alpha=alpha_tail
        )
        show_table(
            ci_table,
            "Confidence Interval Estimates (Parametric: Student-t & Chi-square)",
        )
        download_table_block(
            ci_table, "ci_estimates_combined", "Confidence Interval Estimates"
        )
      except Exception as e:
        st.error(f"Tính toán thất bại: {e}")
    else:
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

          # TÍNH TỨ PHÂN VỊ THEO THUẬT TOÁN 2 BƯỚC CỦA GIÁO TRÌNH
          q1 = compute_percentile_textbook(x, 25)
          q3 = compute_percentile_textbook(x, 75)
          iqr = q3 - q1
          lower_bound = q1 - 1.5 * iqr
          upper_bound = q3 + 1.5 * iqr
          has_outliers = (
              "Có"
              if np.any((x < lower_bound) | (x > upper_bound))
              else "Không"
          )

          mode_res = stats.mode(x, keepdims=True)
          mode_v = mode_res.mode[0] if len(mode_res.mode) > 0 else np.nan

          # BẢNG 1: THỐNG KÊ MÔ TẢ
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
              "IQR": iqr,
          }])
          desc_df = compact_numeric_df(desc_df, decimals=3)
          show_table(desc_df, "Descriptive Statistics")
          download_table_block(
              desc_df, "ci_descriptive_statistics", "Descriptive Statistics"
          )

          # BẢNG 2: KIỂM ĐỊNH CHUẨN VÀ NGOẠI LAI
          if 3 <= n <= 5000:
            sw_stat, sw_p = stats.shapiro(x)
          else:
            sw_stat, sw_p = np.nan, np.nan

          if lilliefors is not None and n >= 4:
            ks_stat, ks_p = lilliefors(x, dist="norm")
          else:
            ks_res = stats.kstest(x, "norm", args=(mean_v, s_v))
            ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)

          is_normal = (
              (sw_p >= 0.05)
              if not np.isnan(sw_p)
              else ((ks_p >= 0.05) if not np.isnan(ks_p) else True)
          )
          norm_status = "Có" if is_normal else "Không"

          normality_diag_df = pd.DataFrame([{
              "[Q1-1.5IQR; Q3+1.5IQR]": (
                  f"[{smart_round_val(lower_bound, 3)};"
                  f" {smart_round_val(upper_bound, 3)}]"
              ),
              "Outliers": has_outliers,
              "Statistic (Shapiro-Wilk)": sw_stat,
              "Sig. (Shapiro-Wilk)": format_p_value(sw_p),
              "Statistic (Kolmogorov-Smirnov)": ks_stat,
              "Sig. (Kolmogorov-Smirnov)": format_p_value(ks_p),
              "Phân phối chuẩn": norm_status,
          }])
          normality_diag_df = compact_numeric_df(
              normality_diag_df, decimals=3
          )
          show_table(normality_diag_df, "Normality & Outlier Diagnostics")
          download_table_block(
              normality_diag_df,
              "ci_normality_diagnostics",
              "Normality & Outlier Diagnostics",
          )

          # BẢNG 3: BẢNG ƯỚC LƯỢNG KHOẢNG TIN CẬY GỘP
          use_boot = force_boot or (not is_normal)
          method_title = "Bootstrap" if use_boot else "Parametric"

          ci_table = ci_combined_estimates(
              x=x, alpha=alpha_tail, use_bootstrap=use_boot, n_boot=int(n_boot)
          )

          show_table(
              ci_table, f"Confidence Interval Estimates ({method_title})"
          )
          download_table_block(
              ci_table,
              "ci_estimates_combined",
              f"Confidence Interval Estimates ({method_title})",
          )

        except Exception as e:
          st.error(f"Tính toán thất bại: {e}")
