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

import google.generativeai as genai
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Cấu hình Gemini API Key
if "GEMINI_API_KEY" in st.secrets:
  genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

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
# CSS Giao diện: To, rõ, đậm
# =========================================================
st.markdown(
    """
<style>
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

div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
  font-size: 19px !important;
  font-weight: 800 !important;
  color: #0f172a !important;
  line-height: 1.3 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label,
div[data-testid="stRadio"] div[role="radiogroup"] label p,
div[data-testid="stRadio"] div[role="radiogroup"] span {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] span {
  font-size: 18px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
}

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
  <p>Regression, categorical analysis, quantitative tests, diagnostics, sample size, probability & AI assistant.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# =========================================================
# Thuật toán 2 bước tính phân vị giáo trình (Q1 = 21 khi n=20)
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
# Phân tích chuỗi nhập liệu
# =========================================================
def parse_numeric_text(txt: str) -> np.ndarray:
  if not txt or not txt.strip():
    return np.array([], dtype=float)
  normalized = txt.replace(";", " ").replace(",", " ")
  parts = [p for p in normalized.split() if p.strip()]
  vals = pd.to_numeric(pd.Series(parts), errors="coerce").dropna()
  return vals.values


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
# Bảng liên định r x c & Đo lường 2x2
# =========================================================
def contingency_editor(
    key: str,
    default_rows: List[str],
    default_cols: List[str],
    default_counts: np.ndarray,
):
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
    new_rows = []
    for i, old in enumerate(df["Group"].astype(str).tolist()):
      new_rows.append(
          st.text_input(f"Row {i+1} label", value=old, key=f"{key}_rowlbl_{i}")
      )
    df["Group"] = new_rows

    cat_cols = [c for c in df.columns if c != "Group"]
    new_cols = []
    for j, old in enumerate(cat_cols):
      new_cols.append(
          st.text_input(
              f"Column {j+1} label", value=str(old), key=f"{key}_collbl_{j}"
          )
      )

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

  cat_cols = [c for c in df.columns if c != "Group"]
  for c in cat_cols:
    df[c] = (
        pd.to_numeric(df[c], errors="coerce")
        .fillna(0)
        .round(0)
        .astype(int)
        .clip(lower=0)
    )

  st.markdown("#### Observed counts (edit cells)")
  edited = st.data_editor(
      df,
      key=f"{key}_editor",
      use_container_width=True,
      num_rows="fixed",
      column_config={"Group": st.column_config.TextColumn("Group")}
      | {
          c: st.column_config.NumberColumn(c, min_value=0, step=1, format="%d")
          for c in cat_cols
      },
  )
  edited = edited.copy()
  edited["Group"] = edited["Group"].astype(str)
  for c in cat_cols:
    edited[c] = (
        pd.to_numeric(edited[c], errors="coerce")
        .fillna(0)
        .round(0)
        .astype(int)
        .clip(lower=0)
    )

  st.session_state[ss_key] = edited
  observed_df = edited[cat_cols].copy()

  counts_df = edited.copy()
  counts_df["Total"] = observed_df.sum(axis=1).astype(int)

  total_row = {"Group": "Total"}
  for c in cat_cols:
    total_row[c] = int(observed_df[c].sum())
  total_row["Total"] = int(observed_df.values.sum())
  counts_df = pd.concat(
      [counts_df, pd.DataFrame([total_row])], ignore_index=True
  )

  return counts_df, observed_df


def rc_contingency_ui(key: str, default_r: int = 2, default_c: int = 2):
  st.markdown("### Table size")
  c1, c2, c3 = st.columns([1, 1, 2])
  with c1:
    r = st.number_input(
        "Rows (r)",
        min_value=2,
        max_value=20,
        value=default_r,
        step=1,
        key=f"{key}_r",
    )
  with c2:
    c = st.number_input(
        "Columns (c)",
        min_value=2,
        max_value=20,
        value=default_c,
        step=1,
        key=f"{key}_c",
    )
  with c3:
    if st.button(
        "Apply size (reset table)",
        key=f"{key}_apply_rc",
        use_container_width=True,
    ):
      rows = [f"Group {i+1}" for i in range(int(r))]
      cols = [f"Category {j+1}" for j in range(int(c))]
      init = np.ones((int(r), int(c)), dtype=int)
      df0 = pd.DataFrame(init, columns=cols)
      df0.insert(0, "Group", rows)
      st.session_state[f"ct_{key}"] = df0
      st.rerun()

  return contingency_editor(
      key=key,
      default_rows=[f"Group {i+1}" for i in range(default_r)],
      default_cols=[f"Category {j+1}" for j in range(default_c)],
      default_counts=np.ones((default_r, default_c), dtype=int),
  )


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


def _safe_div(a, b):
  return np.nan if b == 0 else a / b


def wilson_ci(x, n, alpha=0.05) -> Tuple[float, float]:
  if n <= 0:
    return (np.nan, np.nan)
  z = stats.norm.ppf(1 - alpha / 2)
  phat = x / n
  denom = 1 + z * z / n
  center = (phat + z * z / (2 * n)) / denom
  half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
  return (max(0.0, center - half), min(1.0, center + half))


def log_ci_ratio(est, se, alpha=0.05) -> Tuple[float, float]:
  z = stats.norm.ppf(1 - alpha / 2)
  lo = math.exp(math.log(est) - z * se)
  hi = math.exp(math.log(est) + z * se)
  return lo, hi


def two_by_two_measures(obs2x2: np.ndarray, alpha=0.05) -> pd.DataFrame:
  a, b, c, d = (
      int(obs2x2[0, 0]),
      int(obs2x2[0, 1]),
      int(obs2x2[1, 0]),
      int(obs2x2[1, 1]),
  )
  cc = 0.5 if min(a, b, c, d) == 0 else 0.0
  a2, b2, c2, d2 = a + cc, b + cc, c + cc, d + cc

  OR = (a2 * d2) / (b2 * c2)
  se_log_or = math.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
  or_lo, or_hi = log_ci_ratio(OR, se_log_or, alpha)

  risk_e = _safe_div(a2, (a2 + b2))
  risk_u = _safe_div(c2, (c2 + d2))
  RR = _safe_div(risk_e, risk_u)
  se_log_rr = math.sqrt(
      (1 / a2) - (1 / (a2 + b2)) + (1 / c2) - (1 / (c2 + d2))
  )
  rr_lo, rr_hi = log_ci_ratio(RR, se_log_rr, alpha)

  VE = 1 - RR
  ve_lo, ve_hi = 1 - rr_hi, 1 - rr_lo

  TP, FP, FN, TN = a, b, c, d
  sens = _safe_div(TP, TP + FN)
  spec = _safe_div(TN, TN + FP)
  fpr = _safe_div(FP, FP + TN)
  fnr = _safe_div(FN, FN + TP)
  ppv = _safe_div(TP, TP + FP)
  npv = _safe_div(TN, TN + FN)
  lr_p = _safe_div(sens, 1 - spec) if (1 - spec) not in [0, np.nan] else np.nan
  lr_n = _safe_div(1 - sens, spec) if spec not in [0, np.nan] else np.nan

  sens_lo, sens_hi = (
      wilson_ci(TP, TP + FN, alpha) if (TP + FN) > 0 else (np.nan, np.nan)
  )
  spec_lo, spec_hi = (
      wilson_ci(TN, TN + FP, alpha) if (TN + FP) > 0 else (np.nan, np.nan)
  )
  ppv_lo, ppv_hi = (
      wilson_ci(TP, TP + FP, alpha) if (TP + FP) > 0 else (np.nan, np.nan)
  )
  npv_lo, npv_hi = (
      wilson_ci(TN, TN + FN, alpha) if (TN + FN) > 0 else (np.nan, np.nan)
  )

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
  df = pd.DataFrame(
      rows, columns=["Measure", "Estimate", "CI 2.5%", "CI 97.5%"]
  )
  return compact_numeric_df(df, decimals=3)


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


# KTC cho HIỆU 2 SỐ TRUNG BÌNH
def ci_two_means_diff(
    n1: int,
    m1: float,
    s1: float,
    n2: int,
    m2: float,
    s2: float,
    alpha: float = 0.05,
) -> pd.DataFrame:
  diff = m1 - m2
  s1_sq = s1**2
  s2_sq = s2**2

  # 1. Equal variances assumed (Student's t)
  df_pool = n1 + n2 - 2
  sp_sq = ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / df_pool
  se_pool = math.sqrt(sp_sq * (1.0 / n1 + 1.0 / n2))
  tcrit_pool = float(stats.t.ppf(1 - alpha / 2, df=df_pool))
  lo_pool = diff - tcrit_pool * se_pool
  hi_pool = diff + tcrit_pool * se_pool

  # 2. Equal variances not assumed (Welch's t)
  v1 = s1_sq / n1
  v2 = s2_sq / n2
  se_welch = math.sqrt(v1 + v2)
  df_welch = ((v1 + v2) ** 2) / ((v1**2) / (n1 - 1) + (v2**2) / (n2 - 1))
  tcrit_welch = float(stats.t.ppf(1 - alpha / 2, df=df_welch))
  lo_welch = diff - tcrit_welch * se_welch
  hi_welch = diff + tcrit_welch * se_welch

  lo_pct = alpha / 2 * 100
  hi_pct = (1 - alpha / 2) * 100
  lo_lbl = f"CI {lo_pct:.1f}%" if lo_pct % 1 != 0 else f"CI {lo_pct:.0f}%"
  hi_lbl = f"CI {hi_pct:.1f}%" if hi_pct % 1 != 0 else f"CI {hi_pct:.0f}%"

  res = pd.DataFrame([
      [
          "Phương sai đồng nhất (Student's t)",
          diff,
          se_pool,
          df_pool,
          lo_pool,
          hi_pool,
      ],
      [
          "Phương sai không đồng nhất (Welch's t)",
          diff,
          se_welch,
          df_welch,
          lo_welch,
          hi_welch,
      ],
  ], columns=[
      "Giả định phương sai",
      "Hiệu TB (x̄₁ - x̄₂)",
      "Sai số chuẩn (SE)",
      "df",
      lo_lbl,
      hi_lbl,
  ])
  return compact_numeric_df(res, decimals=3)


# KTC cho HIỆU 2 TỶ LỆ
def ci_two_proportions_diff(
    x1: int, n1: int, x2: int, n2: int, conf_level: float = 0.95
) -> pd.DataFrame:
  alpha = 1.0 - conf_level
  p1 = x1 / n1
  p2 = x2 / n2
  diff = p1 - p2
  z = stats.norm.ppf(1.0 - alpha / 2.0)

  # 1. Wald
  se_wald = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
  w_lo = diff - z * se_wald
  w_hi = diff + z * se_wald

  # 2. Agresti-Caffo
  n1_ac, x1_ac = n1 + 2, x1 + 1
  p1_ac = x1_ac / n1_ac
  n2_ac, x2_ac = n2 + 2, x2 + 1
  p2_ac = x2_ac / n2_ac
  diff_ac = p1_ac - p2_ac
  se_ac = math.sqrt(
      p1_ac * (1.0 - p1_ac) / n1_ac + p2_ac * (1.0 - p2_ac) / n2_ac
  )
  ac_lo = diff_ac - z * se_ac
  ac_hi = diff_ac + z * se_ac

  # 3. Newcombe-Wilson
  w1_lo, w1_hi = wilson_ci(x1, n1, alpha)
  w2_lo, w2_hi = wilson_ci(x2, n2, alpha)
  nw_lo = diff - z * math.sqrt(
      w1_lo * (1.0 - w1_lo) / n1 + w2_hi * (1.0 - w2_hi) / n2
  )
  nw_hi = diff + z * math.sqrt(
      w1_hi * (1.0 - w1_hi) / n1 + w2_lo * (1.0 - w2_lo) / n2
  )

  rows = [
      [
          "Wald (Truyền thống)",
          diff * 100,
          w_lo * 100,
          w_hi * 100,
          (w_hi - w_lo) * 100,
          "Cỡ mẫu lớn (np >= 5)",
      ],
      [
          "Newcombe-Wilson (Khuyên dùng)",
          diff * 100,
          nw_lo * 100,
          nw_hi * 100,
          (nw_hi - nw_lo) * 100,
          "Tối ưu cho cả mẫu nhỏ & tỷ lệ gần 0 hoặc 1",
      ],
      [
          "Agresti-Caffo (Hiệu chỉnh)",
          diff * 100,
          ac_lo * 100,
          ac_hi * 100,
          (ac_hi - ac_lo) * 100,
          "Hiệu chỉnh cộng 2 thành công/thất bại",
      ],
  ]

  out = pd.DataFrame(
      rows,
      columns=[
          "Phương pháp (Method)",
          "Hiệu tỷ lệ (%)",
          "Cận dưới (%)",
          "Cận trên (%)",
          "Độ rộng KTC (%)",
          "Khuyến nghị sử dụng",
      ],
  )
  return compact_numeric_df(out, decimals=3)


# =========================================================
# Sidebar & Điều hướng
# =========================================================
if "section" not in st.session_state:
  st.session_state.section = "Home"
if "sub" not in st.session_state:
  st.session_state.sub = "Overview"


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

  with st.expander(
      "AI Trợ lý Thông minh",
      expanded=(st.session_state.section == "AI Trợ lý"),
  ):
    if st.button(
        "Giải toán & Tạo trắc nghiệm", key="ai_nav_btn", use_container_width=True
    ):
      set_nav("AI Trợ lý", "Giải toán & Trắc nghiệm")

# =========================================================
# Routing Pages
# =========================================================
section = st.session_state.section
sub = st.session_state.sub

# -----------------------------
# HOME
# -----------------------------
if section == "Home":
  st.markdown("## Overview")
  st.info("Chọn chức năng từ thanh menu bên trái để bắt đầu phân tích.")

# -----------------------------
# CONFIDENCE INTERVALS — PROPORTION
# -----------------------------
elif section == "Confidence Intervals" and sub == "Proportion":
  st.markdown("## Confidence Intervals — Proportion")
  ci_prop_target = st.radio(
      "Chọn mục tiêu ước lượng tỷ lệ:",
      [
          "Ước lượng một tỷ lệ (Single Proportion)",
          (
              "Ước lượng hiệu hai tỷ lệ (Difference between Two"
              " Proportions)"
          ),
      ],
      horizontal=True,
  )

  if ci_prop_target.startswith("Ước lượng một tỷ lệ"):
    c1, c2, c3 = st.columns(3)
    with c1:
      x_evt = st.number_input(
          "Số biến cố (x / events)", min_value=0, value=50, step=1
      )
    with c2:
      n_tot = st.number_input(
          "Cỡ mẫu (n / total)", min_value=1, value=100, step=1
      )
    with c3:
      conf_prop_choice = st.radio(
          "Độ tin cậy", ["95%", "99%", "Khác..."], horizontal=True
      )
      conf_l = (
          0.95
          if conf_prop_choice == "95%"
          else (
              0.99
              if conf_prop_choice == "99%"
              else st.slider(
                  "Độ tin cậy tùy chỉnh", 0.80, 0.999, 0.90, 0.005, format="%.3f"
              )
          )
      )

    if int(x_evt) > int(n_tot):
      st.error("Số biến cố không thể lớn hơn cỡ mẫu.")
    else:
      if st.button(
          "Compute Proportion CI", type="primary", use_container_width=True
      ):
        # Wald & Wilson methods
        alpha = 1.0 - conf_l
        w_lo, w_hi = wilson_ci(int(x_evt), int(n_tot), alpha)
        p_hat = int(x_evt) / int(n_tot)
        se_w = math.sqrt(p_hat * (1 - p_hat) / int(n_tot))
        z_c = stats.norm.ppf(1 - alpha / 2)
        tbl = pd.DataFrame([
            [
                "Wald (Normal)",
                p_hat * 100,
                max(0.0, p_hat - z_c * se_w) * 100,
                min(1.0, p_hat + z_c * se_w) * 100,
            ],
            ["Wilson (Score)", p_hat * 100, w_lo * 100, w_hi * 100],
        ], columns=["Method", "Proportion (%)", "Lower CI (%)", "Upper CI (%)"])
        show_table(
            compact_numeric_df(tbl, 3), "Proportion Confidence Intervals"
        )
        download_table_block(tbl, "prop_ci", "Proportion CI")

  else:
    c1, c2 = st.columns(2)
    with c1:
      n1_in = st.number_input("Cỡ mẫu nhóm 1 (n1)", min_value=1, value=100)
      x1_in = st.number_input(
          "Số biến cố nhóm 1 (x1)", min_value=0, max_value=int(n1_in), value=45
      )
    with c2:
      n2_in = st.number_input("Cỡ mẫu nhóm 2 (n2)", min_value=1, value=120)
      x2_in = st.number_input(
          "Số biến cố nhóm 2 (x2)", min_value=0, max_value=int(n2_in), value=30
      )

    if st.button(
        "Compute Difference in Proportions CI",
        type="primary",
        use_container_width=True,
    ):
      diff_tbl = ci_two_proportions_diff(
          int(x1_in), int(n1_in), int(x2_in), int(n2_in), conf_level=0.95
      )
      show_table(
          diff_tbl,
          "Khoảng tin cậy cho hiệu hai tỷ lệ (p₁ - p₂) — Mức tin cậy 95%",
      )
      download_table_block(
          diff_tbl, "ci_difference_two_proportions", "KTC hiệu 2 tỷ lệ"
      )

# -----------------------------
# CONFIDENCE INTERVALS — MEAN, SD & VARIANCE
# -----------------------------
elif section == "Confidence Intervals" and sub == "Mean & Variance":
  st.markdown("## Confidence Intervals — Mean, SD & Variance")

  ci_mean_target = st.radio(
      "Chọn mục tiêu ước lượng:",
      [
          "Ước lượng một số trung bình (Mean, SD, Variance)",
          (
              "Ước lượng hiệu hai số trung bình (Difference between Two"
              " Means)"
          ),
      ],
      horizontal=True,
  )

  if ci_mean_target.startswith("Ước lượng một số trung bình"):
    txt = st.text_area(
        "Dán dữ liệu số:",
        value=(
            "12; 14; 16; 18; 20; 22; 24; 26; 28; 30; 32; 34; 36; 38; 40; 42;"
            " 44; 46; 48; 50"
        ),
        height=80,
    )
    if st.button("Compute CI", type="primary", use_container_width=True):
      x_arr = parse_numeric_text(txt)
      if len(x_arr) >= 2:
        ci_tbl = ci_combined_estimates(x_arr, alpha=0.05)
        show_table(ci_tbl, "Confidence Interval Estimates")
        download_table_block(ci_tbl, "ci_single_mean", "CI Estimates")
  else:
    c1, c2 = st.columns(2)
    with c1:
      txt1 = st.text_area(
          "Dãy số nhóm 1:",
          value="12.5; 14.2; 11.8; 15.0; 13.6; 14.8; 12.9",
          height=80,
      )
    with c2:
      txt2 = st.text_area(
          "Dãy số nhóm 2:",
          value="10.2; 11.5; 9.8; 12.0; 10.9; 11.2; 9.5",
          height=80,
      )
    if st.button(
        "Compute Difference in Means CI",
        type="primary",
        use_container_width=True,
    ):
      s1 = parse_numeric_text(txt1)
      s2 = parse_numeric_text(txt2)
      if len(s1) >= 2 and len(s2) >= 2:
        diff_tbl = ci_two_means_diff(
            len(s1),
            float(np.mean(s1)),
            float(np.std(s1, ddof=1)),
            len(s2),
            float(np.mean(s2)),
            float(np.std(s2, ddof=1)),
            alpha=0.05,
        )
        show_table(
            diff_tbl,
            "Khoảng tin cậy cho hiệu hai số trung bình (μ₁ - μ₂) — Mức tin cậy"
            " 95%",
        )
        download_table_block(
            diff_tbl, "ci_diff_means", "KTC hiệu hai số trung bình"
        )

# -----------------------------
# AI TRỢ LÝ THỐNG KÊ & TRẮC NGHIỆM
# -----------------------------
elif section == "AI Trợ lý" and sub == "Giải toán & Trắc nghiệm":
  st.markdown("## 🤖 AI Trợ lý: Giải bài toán Thống kê Y sinh & Tạo trắc nghiệm")
  st.write(
      "Dán văn bản hoặc tải ảnh đề bài (ảnh chụp, tài liệu in). AI sẽ tự động"
      " nhận diện dạng toán, giải chi tiết và tạo bộ câu hỏi trắc nghiệm A, B,"
      " C, D theo đúng số lượng yêu cầu."
  )

  c1, c2 = st.columns(2)
  with c1:
    txt_input = st.text_area(
        "Nhập hoặc dán văn bản đề bài:",
        value=(
            "Một nghiên cứu so sánh huyết áp tâm thu giữa hai nhóm bệnh nhân.\nNhóm"
            " điều trị có n1 = 40, huyết áp tâm thu trung bình x̄1 = 128,5 mmHg"
            " và độ lệch chuẩn S1 = 10,0 mmHg.\nNhóm chứng có n2 = 40, huyết áp"
            " tâm thu trung bình x̄2 = 134,2 mmHg và độ lệch chuẩn S2 = 9,0"
            " mmHg."
        ),
        height=140,
    )
  with c2:
    img_file = st.file_uploader(
        "Hoặc tải / dán ảnh đề bài (PNG, JPG):", type=["png", "jpg", "jpeg"]
    )
    if img_file is not None:
      st.image(
          Image.open(img_file),
          caption="Ảnh đề bài đã tải lên",
          use_container_width=True,
      )

  c_cfg1, c_cfg2 = st.columns(2)
  with c_cfg1:
    action_mode = st.radio(
        "Chọn yêu cầu xử lý:",
        [
            "Cả giải chi tiết và tạo trắc nghiệm",
            "Chỉ giải bài toán chi tiết (KTC + Kiểm định)",
            "Chỉ tạo câu hỏi trắc nghiệm A, B, C, D",
        ],
        horizontal=False,
    )
  with c_cfg2:
    num_questions = st.number_input(
        "Số lượng câu trắc nghiệm cần tạo:",
        min_value=1,
        max_value=30,
        value=4,
        step=1,
    )

  if st.button(
      "🚀 Bắt đầu xử lý với AI", type="primary", use_container_width=True
  ):
    if not txt_input.strip() and img_file is None:
      st.warning("Vui lòng dán văn bản hoặc tải lên hình ảnh đề bài.")
    elif "GEMINI_API_KEY" not in st.secrets:
      st.error(
          "Chưa cấu hình GEMINI_API_KEY trong mục Settings -> Secrets của"
          " Streamlit."
      )
    else:
      with st.spinner("AI đang đọc đề, nhận diện dạng toán và tính toán..."):
        try:
          model = genai.GenerativeModel("gemini-2.5-flash")

          prompt = f"""
Bạn là chuyên gia Thống kê Y học và giảng viên bộ môn Xác suất Thống kê Y Dược.
Nhiệm vụ: Phân tích bài toán được cung cấp trong văn bản hoặc ảnh đính kèm và thực hiện theo đúng chế độ: "{action_mode}".

HƯỚNG DẪN XỬ LÝ:
1. NẾU CÓ GIẢI BÀI TOÁN:
   - Tự động nhận diện dữ liệu: cỡ mẫu (n), trung bình, độ lệch chuẩn/phương sai, tỷ lệ...
   - Xác định chính xác dạng toán (ước lượng KTC, kiểm định 1 trung bình, 2 trung bình độc lập, bắt cặp, 2 tỷ lệ, Chi-square...).
   - Tính toán đầy đủ từng bước: Sai số chuẩn (SE), Thống kê kiểm định (t hoặc Z), bậc tự do (df), giá trị p-value hai phía, Khoảng tin cậy KTC 95%.
   - Đưa ra kết luận có ý nghĩa lâm sàng/y học rõ ràng.

2. NẾU CÓ TẠO TRẮC NGHIỆM:
   - Tạo đúng {num_questions} câu hỏi trắc nghiệm xoay quanh bài toán này (về tính SE, tính t/Z, tính KTC, diễn giải kết luận ý nghĩa y học, nhận diện sai lầm...).
   - Mỗi câu có 4 phương án A, B, C, D (chỉ 1 đáp án đúng, các phương án nhiễu phải hợp lý và bám sát các lỗi sinh viên hay gặp).
   - Dưới mỗi câu phải ghi rõ: **Đáp án đúng: ...** và **Giải thích ngắn gọn: ...**.

Trình bày bằng tiếng Việt với định dạng Markdown rõ ràng, chuẩn mực sư phạm.
"""
          content_parts = [prompt]
          if txt_input.strip():
            content_parts.append(f"VĂN BẢN ĐỀ BÀI:\n{txt_input}")
          if img_file is not None:
            content_parts.append(Image.open(img_file))

          res = model.generate_content(content_parts)

          st.markdown("---")
          st.markdown("### 📋 Kết quả xử lý từ AI")
          st.markdown(res.text)

          st.download_button(
              "📥 Tải kết quả bài giải & trắc nghiệm (.txt)",
              data=res.text,
              file_name="loi_giai_va_trac_nghiem_ai.txt",
              mime="text/plain",
          )
        except Exception as e:
          st.error(f"Lỗi xử lý AI: {e}")
