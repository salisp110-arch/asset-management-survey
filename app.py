# app.py
# -*- coding: utf-8 -*-
import os, json, base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# scikit-learn اختیاری
try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ───────────────────────── تنظیمات پایه ─────────────────────────
st.set_page_config(page_title="پرسشنامه و داشبورد مدیریت دارایی", layout="wide")
BASE = Path(".")
DATA_DIR = BASE / "data"
ASSETS_DIR = BASE / "assets"
DATA_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# تم و فونت (Vazir + fallback به Vazirmatn)
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css">
<style>
/* سراسری: همه عناصر مجبور به استفاده از Vazir */
:root{ --app-font: Vazir, Tahoma, Arial, sans-serif; }
html, body, * { font-family: var(--app-font) !important; direction: rtl; }

/* کمی بهبود چینش کلی */
.block-container { padding-top: .6rem; padding-bottom: 3rem; }
h1,h2,h3,h4 { color:#16325c; }

/* کارت‌ها و KPIها (مثل قبل، با فونت جدید) */
.question-card {
  background: rgba(255,255,255,0.78);
  backdrop-filter: blur(6px);
  padding: 18px 20px; margin: 10px 0 18px 0;
  border-radius: 14px; border: 1px solid #e8eef7;
  box-shadow: 0 6px 16px rgba(36,74,143,0.08), inset 0 1px 0 rgba(255,255,255,0.7);
}
.q-head{ font-weight:700; color:#16325c; font-size:16px; margin-bottom:8px; }
.q-desc{ color:#4a5e85; font-size:14px; line-height:1.9; margin-bottom:10px; }
.q-num{ display:inline-block; background:#e8f0fe; color:#16325c; font-weight:700; border-radius:8px; padding:2px 8px; margin-left:6px; font-size:12px; }

.kpi{
  border-radius:14px; padding:16px 18px; border:1px solid #e6ecf5;
  background:linear-gradient(180deg,#ffffff 0%,#f6f9ff 100%);
  box-shadow:0 8px 20px rgba(0,0,0,0.05); min-height:96px;
}
.kpi .title{ color:#456; font-size:13px; margin-bottom:6px; }
.kpi .value{ color:#0f3b8f; font-size:22px; font-weight:800; }
.kpi .sub{ color:#6b7c93; font-size:12px; }
</style>
""", unsafe_allow_html=True)


PLOTLY_TEMPLATE = "plotly_white"

# ─────────────────────── بارگذاری موضوعات ───────────────────────
TOPICS_PATH = BASE / "topics.json"
if not TOPICS_PATH.exists():
    st.error("فایل topics.json پیدا نشد. آن را کنار app.py قرار دهید.")
    st.stop()
TOPICS = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
if len(TOPICS) != 40:
    st.warning("⚠️ تعداد موضوعات باید دقیقاً ۴۰ باشد تا وزن‌دهی فازی درست محاسبه شود.")

# ─────────────────────── نقش‌ها و رنگ‌ها ───────────────────────
ROLES = [
    "مدیران ارشد",
    "مدیران اجرایی",
    "سرپرستان / خبرگان",
    "متخصصان فنی",
    "متخصصان غیر فنی",
]
ROLE_COLORS = {
    "مدیران ارشد":"#d62728",
    "مدیران اجرایی":"#1f77b4",
    "سرپرستان / خبرگان":"#2ca02c",
    "متخصصان فنی":"#ff7f0e",
    "متخصصان غیر فنی":"#9467bd"
}

# ───────────────────── گزینه‌های پاسخ ─────────────────────
LEVEL_OPTIONS = [
    ("اطلاعی در این مورد ندارم.", 0),
    ("سازمان نیاز به این موضوع را شناسایی کرده ولی جزئیات آن را نمی‌دانم.", 1),
    ("سازمان در حال تدوین دستورالعمل‌های مرتبط است و فعالیت‌هایی به‌صورت موردی انجام می‌شود.", 2),
    ("بله، این موضوع در سازمان به‌صورت کامل و استاندارد پیاده‌سازی و اجرایی شده است.", 3),
    ("بله، چند سال است که نتایج اجرای آن بر اساس شاخص‌های استاندارد ارزیابی می‌شود و از بهترین تجربه‌ها برای بهبود مستمر استفاده می‌گردد.", 4),
]
REL_OPTIONS = [
    ("هیچ ارتباطی ندارد.", 1),
    ("ارتباط کم دارد.", 3),
    ("تا حدی مرتبط است.", 5),
    ("ارتباط زیادی دارد.", 7),
    ("کاملاً مرتبط است.", 10),
]

# ─────────── ضرایب فازی نرمال‌شده برای ۴۰ موضوع (خلاصه‌شده از جدول شما) ───────────
ROLE_MAP_EN2FA = {
    "Senior Managers":"مدیران ارشد",
    "Executives":"مدیران اجرایی",
    "Supervisors/Sr Experts":"سرپرستان / خبرگان",
    "Technical Experts":"متخصصان فنی",
    "Non-Technical Experts":"متخصصان غیر فنی",
}
# توجه: این جدول دقیقاً همان «Normalized»‌های شماست:
NORM_WEIGHTS = {
    1:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    2:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    3:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    4:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    5:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    6:{"Senior Managers":0.1923,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    7:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    8:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    9:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.1923},
    10:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    11:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    12:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    13:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    14:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    15:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    16:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    17:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    18:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    19:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    20:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    21:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    22:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    23:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    24:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    25:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    26:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    27:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    28:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    29:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.0385,"Technical Experts":0.1154,"Non-Technical Experts":0.2692},
    30:{"Senior Managers":0.1154,"Executives":0.3846,"Supervisors/Sr Experts":0.0385,"Technical Experts":0.2692,"Non-Technical Experts":0.1923},
    31:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    32:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.1923},
    33:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.2692},
    34:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.1923},
    35:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.2692},
    36:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    37:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.1923,"Non-Technical Experts":0.1154},
    38:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.1923,"Non-Technical Experts":0.1154},
    39:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    40:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.1923},
}

# ─────────────────────── کمک‌تابع‌ها ───────────────────────
def ensure_company(company: str):
    (DATA_DIR / company).mkdir(parents=True, exist_ok=True)

def load_company_df(company: str) -> pd.DataFrame:
    ensure_company(company)
    p = DATA_DIR / company / "responses.csv"
    if p.exists():
        return pd.read_csv(p)
    cols = ["timestamp","company","respondent","role"]
    for t in TOPICS:
        cols += [f"t{t['id']}_maturity", f"t{t['id']}_rel", f"t{t['id']}_adj"]
    return pd.DataFrame(columns=cols)

def save_response(company: str, record: dict):
    df_old = load_company_df(company)
    df_new = pd.concat([df_old, pd.DataFrame([record])], ignore_index=True)
    df_new.to_csv(DATA_DIR / company / "responses.csv", index=False)

def normalize_adj_to_100(x):
    return (x/40.0)*100.0 if pd.notna(x) else np.nan

def _angles_deg_40():
    base = np.arange(0, 360, 360/40.0)  # 0..351
    return (base + 90) % 360  # رأس اول بالا

def plot_radar(series_dict, title, tick_names, target=45, annotate=False, show_legend=True):
    N = len(tick_names)
    angles_deg = _angles_deg_40()
    fig = go.Figure()
    for label, values in series_dict.items():
        vals = list(values)
        if len(vals)!=N: vals = (vals + [None]*N)[:N]
        vals_closed = vals + [vals[0]]
        theta_closed = angles_deg.tolist() + [angles_deg[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=theta_closed, thetaunit="degrees",
            mode="lines+markers"+("+text" if annotate else ""),
            name=label,
            text=[f"{v:.0f}" if v is not None else "" for v in vals_closed] if annotate else None,
            textposition="top center",
            line=dict(width=2),
            marker=dict(size=6, line=dict(width=1), color=ROLE_COLORS.get(label, None))
        ))
    # خط/باند هدف
    fig.add_trace(go.Scatterpolar(
        r=[target]*(N+1), theta=angles_deg.tolist()+[angles_deg[0]], thetaunit="degrees",
        mode="lines", name=f"هدف {target}", line=dict(dash="dash", width=3, color="#444"), hoverinfo="skip"
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Vazirmatn"),
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], dtick=10, gridcolor="#e6ecf5"),
            angularaxis=dict(thetaunit="degrees", direction="clockwise", rotation=0,
                             tickmode="array", tickvals=angles_deg.tolist(), ticktext=tick_names,
                             gridcolor="#edf2fb"),
            bgcolor="white"
        ),
        paper_bgcolor="#ffffff",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=60,b=90,l=10,r=10)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_bars_multirole(per_role_norm_fa: dict, topic_names, title, target=45):
    x = [f"{i+1:02d} — {n}" for i,n in enumerate(topic_names)]
    fig = go.Figure()
    for label, vals in per_role_norm_fa.items():
        fig.add_trace(go.Bar(x=x, y=vals, name=label, marker_color=ROLE_COLORS.get(label)))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Vazirmatn"),
        title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)",
        xaxis=dict(tickfont=dict(size=10)), barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=60,b=140,l=10,r=10), paper_bgcolor="#ffffff"
    )
    # باند هدف ±5
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red",
                  annotation_text=f"هدف {target}", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

def plot_lines_multirole(per_role_norm_fa: dict, title, target=45):
    x = [f"{i+1:02d}" for i in range(40)]
    fig = go.Figure()
    for label, vals in per_role_norm_fa.items():
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=label,
                                 line=dict(width=2, color=ROLE_COLORS.get(label))))
    fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Vazirmatn"),
                      title=title, xaxis_title="موضوع (01..40)", yaxis_title="نمره (0..100)",
                      paper_bgcolor="#ffffff", hovermode="x unified")
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red",
                  annotation_text=f"هدف {target}", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

def org_weighted_topic(per_role_norm_fa: dict, topic_id: int):
    """میانگین وزنی سازمان برای موضوع «topic_id» با ضرایب فازی نرمال‌شده."""
    w = NORM_WEIGHTS.get(topic_id, {})
    num=0.0; den=0.0
    for en_key, weight in w.items():
        fa_role = ROLE_MAP_EN2FA[en_key]
        vlist = per_role_norm_fa.get(fa_role, [])
        idx = topic_id-1
        if idx < len(vlist) and pd.notna(vlist[idx]):
            num += weight * vlist[idx]; den += weight
    return np.nan if den==0 else num/den

def get_company_logo_path(company: str) -> Path|None:
    folder = DATA_DIR / company
    for ext in ("png","jpg","jpeg"):
        p = folder / f"logo.{ext}"
        if p.exists(): return p
    return None

# ─────────────────────── سایدبار عمومی ───────────────────────
st.sidebar.header("تنظیمات و برندینگ")
# لوگوی هلدینگ (آپلود و ذخیره در assets)
holding_logo_file = st.sidebar.file_uploader("لوگوی هلدینگ انرژی گستر سینا", type=["png","jpg","jpeg"])
if holding_logo_file:
    (ASSETS_DIR / "holding_logo.png").write_bytes(holding_logo_file.getbuffer())
holding_logo_path = ASSETS_DIR / "holding_logo.png"

TARGET = st.sidebar.slider("🎯 خط هدف (0..100)", 0, 100, 45, 1)
annotate_radar = st.sidebar.checkbox("نمایش اعداد روی نقاط رادار", value=False)

tabs = st.tabs(["📝 پرسشنامه","📊 داشبورد"])

# ───────────────────────── تب پرسشنامه ─────────────────────────
with tabs[0]:
    # هدر: لوگوی هلدینگ + عنوان
    col_logo, col_title = st.columns([1,5], vertical_alignment="center")
    with col_logo:
        if holding_logo_path.exists():
            st.image(str(holding_logo_path), width=130)
    with col_title:
        st.markdown("## پرسشنامه تعیین سطح بلوغ هلدینگ انرژی گستر سینا و شرکت‌های تابعه در مدیریت دارایی فیزیکی")

    st.info("برای هر موضوع ابتدا توضیح فارسی آن را بخوانید، سپس به دو پرسش پاسخ دهید.")

    company = st.text_input("نام شرکت")
    respondent = st.text_input("نام و نام خانوادگی (اختیاری)")
    role = st.selectbox("نقش / رده سازمانی", ROLES)

    # صفحه‌بندی
    N_PER_PAGE = st.sidebar.number_input("تعداد موضوع در هر صفحه", 5, 40, 8, step=1)
    total_pages = int(np.ceil(len(TOPICS)/N_PER_PAGE))
    page_idx = st.sidebar.number_input("صفحه", 1, max(1,total_pages), 1, step=1) - 1
    start, end = page_idx * N_PER_PAGE, min(page_idx * N_PER_PAGE + N_PER_PAGE, len(TOPICS))
    st.caption(f"نمایش موضوعات {start+1} تا {end} از {len(TOPICS)}")

    answers = {}
    for t in TOPICS[start:end]:
        with st.expander(f"{t['id']:02d} — {t['name']}", expanded=False):
            st.markdown(f'''
            <div class="question-card">
              <div class="q-head"><span class="q-num">{t["id"]:02d}</span>{t["name"]}</div>
              <div class="q-desc">{t["desc"].replace("\n","<br>")}</div>
            </div>
            ''', unsafe_allow_html=True)
            m_choice = st.radio(f"۱) به نظر شما، موضوع «{t['name']}» در سازمان شما در چه سطحی قرار دارد؟",
                                options=[opt for (opt,_) in LEVEL_OPTIONS], key=f"mat_{t['id']}")
            r_choice = st.radio(f"۲) موضوع «{t['name']}» چقدر به حیطه کاری شما ارتباط مستقیم دارد؟",
                                options=[opt for (opt,_) in REL_OPTIONS], key=f"rel_{t['id']}")
            answers[t['id']] = (m_choice, r_choice)

    # ثبت پاسخ‌ها
    if st.button("ثبت پاسخ"):
        if not company:
            st.error("نام شرکت را وارد کنید.")
        elif not role:
            st.error("نقش/رده سازمانی را انتخاب کنید.")
        elif len(answers) < len(TOPICS):  # چون صفحه‌بندی داریم، باید کل صفحات تکمیل شوند
            st.warning("شما فقط بخشی از موضوعات را پاسخ دادید. لطفاً همه صفحات را تکمیل و سپس ثبت کنید.")
        else:
            ensure_company(company)
            rec = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "company": company, "respondent": respondent, "role": role}
            for t in TOPICS:
                m_choice = answers.get(t['id'], (LEVEL_OPTIONS[0][0], REL_OPTIONS[0][0]))[0]
                r_choice = answers.get(t['id'], (LEVEL_OPTIONS[0][0], REL_OPTIONS[0][0]))[1]
                mat = dict(LEVEL_OPTIONS)[m_choice]
                rel = dict(REL_OPTIONS)[r_choice]
                adj = mat * rel
                rec[f"t{t['id']}_maturity"] = mat
                rec[f"t{t['id']}_rel"] = rel
                rec[f"t{t['id']}_adj"] = adj
            save_response(company, rec)
            st.success("✅ پاسخ شما با موفقیت ذخیره شد.")

# ───────────────────────── تب داشبورد ─────────────────────────
with tabs[1]:
    st.subheader("📊 داشبورد نتایج")

    # ورود با رمز
    password = st.text_input("🔑 رمز عبور داشبورد را وارد کنید", type="password")
    if password != "Emacraven110":
        st.error("دسترسی محدود است. رمز عبور درست را وارد کنید.")
        st.stop()

    # انتخاب شرکت
    companies = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    if not companies:
        st.warning("هنوز هیچ پاسخی ثبت نشده است.")
        st.stop()
    company = st.selectbox("انتخاب شرکت", companies)

    # نمایش لوگوها
    row_logo = st.columns([1, 1, 6])
    with row_logo[0]:
        if holding_logo_path.exists():
            st.image(str(holding_logo_path), width=100, caption="هلدینگ انرژی گستر سینا")
    with row_logo[1]:
        st.caption("لوگوی شرکت منتخب:")
        comp_logo_file = st.file_uploader("آپلود/به‌روزرسانی لوگوی شرکت", key="uplogo", type=["png","jpg","jpeg"])
        if comp_logo_file:
            ensure_company(company)
            # ذخیره به نام ثابت
            (DATA_DIR / company / "logo.png").write_bytes(comp_logo_file.getbuffer())
        comp_logo_path = get_company_logo_path(company)
        if comp_logo_path:
            st.image(str(comp_logo_path), width=100, caption=company)

    df = load_company_df(company)
    if df.empty:
        st.warning("برای این شرکت پاسخی وجود ندارد.")
        st.stop()

    # نرمال‌سازی امتیازها (0..40 -> 0..100)
    adj_cols = [f"t{t['id']}_adj" for t in TOPICS]
    df_norm = df.copy()
    for c in adj_cols: df_norm[c] = df_norm[c].apply(normalize_adj_to_100)

    # میانگین به تفکیک نقش
    role_means = {}
    for role in ROLES:
        sub = df_norm[df_norm["role"]==role]
        if sub.empty:
            role_means[role] = [np.nan]*len(TOPICS)
        else:
            role_means[role] = [sub[f"t{t['id']}_adj"].mean() for t in TOPICS]

    # فیلتر نقش‌ها و بازه موضوع‌ها
    roles_selected = st.multiselect("🎚 نقش‌های قابل نمایش", ROLES, default=ROLES)
    topic_range = st.slider("بازهٔ موضوع‌ها", 1, 40, (1,40))
    idx0, idx1 = topic_range[0]-1, topic_range[1]
    topics_slice = TOPICS[idx0:idx1]
    topic_names_slice = [t['name'] for t in topics_slice]

    role_means_filtered = {
        r: role_means[r][idx0:idx1] for r in roles_selected
    }

    # میانگین وزنی فازی سازمان برای هر موضوع
    per_role_norm_fa = {r: role_means[r] for r in ROLES}
    org_series_full = [org_weighted_topic(per_role_norm_fa, t["id"]) for t in TOPICS]
    org_series = org_series_full[idx0:idx1]

    # KPI ها
    org_avg = np.nanmean(org_series_full)
    pass_rate = np.mean([1 if (v>=TARGET) else 0 for v in org_series_full if pd.notna(v)])*100
    # بهترین/ضعیف‌ترین (میانگین ساده نقش‌ها)
    simple_means = []
    for i, t in enumerate(TOPICS):
        vals = [role_means[r][i] for r in ROLES if pd.notna(role_means[r][i])]
        simple_means.append(np.nanmean(vals) if vals else np.nan)
    best_idx = int(np.nanargmax(simple_means)) if np.isfinite(np.nanmax(simple_means)) else None
    worst_idx = int(np.nanargmin(simple_means)) if np.isfinite(np.nanmin(simple_means)) else None
    best_label = f"{best_idx+1:02d} — {TOPICS[best_idx]['name']}" if best_idx is not None else "-"
    worst_label = f"{worst_idx+1:02d} — {TOPICS[worst_idx]['name']}" if worst_idx is not None else "-"

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"""<div class="kpi"><div class="title">میانگین سازمان (فازی)</div>
    <div class="value">{(org_avg if pd.notna(org_avg) else 0):.1f}</div><div class="sub">از 100</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi"><div class="title">نرخ عبور از هدف</div>
    <div class="value">{pass_rate:.0f}%</div><div class="sub">نقاط ≥ هدف</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi"><div class="title">بهترین موضوع</div>
    <div class="value">{best_label}</div><div class="sub">میانگین ساده نقش‌ها</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi"><div class="title">ضعیف‌ترین موضوع</div>
    <div class="value">{worst_label}</div><div class="sub">میانگین ساده نقش‌ها</div></div>""", unsafe_allow_html=True)

    # Top/Bottom 5 (اطلاع مکمل)
    st.markdown("### 🔝 Top 5 و Bottom 5 (میانگین ساده نقش‌ها)")
    topic_mean_series = pd.Series(simple_means, index=[f"{i+1:02d} — {t['name']}" for i,t in enumerate(TOPICS)])
    colA, colB = st.columns(2)
    colA.write("**Top 5**"); colA.table(topic_mean_series.sort_values(ascending=False).head(5).round(1))
    colB.write("**Bottom 5**"); colB.table(topic_mean_series.sort_values(ascending=True).head(5).round(1))

    # رادار تک‌نقش‌ها (روی بازه انتخابی)
    st.markdown("### 🌐 رادار ۴۰‌بخشی برای هر رده (نمره نرمال‌شده)")
    cols = st.columns(2)
    tick_full_slice = [f"{i+idx0+1:02d} — {t['name']}" for i,t in enumerate(topics_slice)]
    c_idx=0
    for fa in roles_selected:
        vals = role_means_filtered.get(fa, [])
        if not vals or all(pd.isna(vals)): continue
        with cols[c_idx%2]:
            plot_radar({fa: vals}, f"رادار — {fa}", tick_full_slice, target=TARGET, annotate=annotate_radar, show_legend=False)
        c_idx+=1
    if c_idx==0:
        st.info("داده‌ای برای ترسیم رادار تک‌نقش وجود ندارد.")

    # رادار ترکیبی نقش‌ها
    st.markdown("### 🌐 رادار مقایسه‌ای نقش‌ها")
    overlay = {fa: role_means_filtered[fa] for fa in roles_selected if any(pd.notna(x) for x in role_means_filtered[fa])}
    if overlay:
        plot_radar(overlay, "رادار مقایسه‌ای نقش‌ها", tick_full_slice, target=TARGET, annotate=False, show_legend=True)
    else:
        st.info("داده‌ای برای رادار تجمیعی وجود ندارد.")

    # رادار میانگین سازمانِ فازی
    st.markdown("### 🏛️ رادار میانگین سازمان (وزن‌دهی فازی)")
    plot_radar({"میانگین سازمان": org_series}, "رادار — میانگین سازمان (فازی)", tick_full_slice,
               target=TARGET, annotate=annotate_radar, show_legend=False)

    # میله‌ای نقش‌ها
    st.markdown("### 📊 نمودار میله‌ای (نقش‌ها)")
    plot_bars_multirole({r:role_means[r][idx0:idx1] for r in roles_selected},
                        topic_names_slice, "مقایسه رده‌ها (0..100)", target=TARGET)

    # خطی نقش‌ها
    st.markdown("### 📈 نمودار خطی مقایسه‌ای")
    plot_lines_multirole({r:role_means[r][idx0:idx1] for r in roles_selected},
                         "Line Chart — مقایسه رده‌ها", target=TARGET)

    # Heatmap
    st.markdown("### 🔥 Heatmap موضوعات × نقش‌ها (0..100)")
    heat_df = pd.DataFrame({ "موضوع":[f"{i+idx0+1:02d} — {t['name']}" for i,t in enumerate(topics_slice)] })
    for fa in roles_selected:
        heat_df[fa] = role_means[fa][idx0:idx1]
    heat_melt = heat_df.melt(id_vars="موضوع", var_name="نقش", value_name="امتیاز")
    fig_heat = px.density_heatmap(heat_melt, x="نقش", y="موضوع", z="امتیاز",
                                  color_continuous_scale="RdYlGn", height=600, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Boxplot
    st.markdown("### 📦 Boxplot توزیع نمرات")
    fig_box = px.box(heat_melt.dropna(), x="نقش", y="امتیاز", points="all", color="نقش",
                     color_discrete_map=ROLE_COLORS, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_box, use_container_width=True)

    # همبستگی و خوشه‌بندی
    st.markdown("### 🔗 ماتریس همبستگی موضوعات و خوشه‌بندی")
    corr_base = heat_df.set_index("موضوع")[roles_selected]
    if not corr_base.empty:
        corr = corr_base.T.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             aspect="auto", height=700, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_corr, use_container_width=True)
    if SKLEARN_OK and corr_base.notna().any().any():
        k = st.slider("تعداد خوشه‌ها (K)", 2, 6, 3)
        X = corr_base.fillna(corr_base.mean()).values
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        clusters = km.labels_
        cl_df = pd.DataFrame({
            "موضوع":corr_base.index,
            "خوشه":clusters
        }).sort_values("خوشه")
        st.dataframe(cl_df, use_container_width=True)
    else:
        st.caption('<span class="small-note">برای خوشه‌بندی به scikit-learn و دادهٔ کافی نیاز است.</span>', unsafe_allow_html=True)

    # دانلودها
    st.markdown("### ⬇️ دانلود داده‌ها")
    st.download_button("دانلود CSV پاسخ‌های شرکت",
                       data=load_company_df(company).to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{company}_responses.csv", mime="text/csv")
    # برای دانلود تصویر نمودارها می‌توان kaleido نصب کرد و از fig.to_image استفاده نمود.
    st.caption('برای دانلود تصاویر نمودارها، بستهٔ `kaleido` را به requirements اضافه کنید و از `fig.to_image("png")` استفاده کنید.')
