# app.py
# -*- coding: utf-8 -*-
import os, json
import numpy as np
import pandas as pd
import streamlit as st  # ⬅️ ابتدا استریم‌لیت را ایمپورت می‌کنیم تا بتوانیم خطا را دوستانه نشان دهیم
from pathlib import Path
from datetime import datetime

# --- نصب خودکار پکیج‌های موردنیاز در صورت نبودن ---
def _ensure_pkg(pkg, version=None):
    """Try to import; if missing, pip install quietly."""
    try:
        __import__(pkg)
        return True
    except Exception:
        try:
            import sys, subprocess
            if version:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", f"{pkg}=={version}"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
            __import__(pkg)
            return True
        except Exception as e:
            st.error(f"نصب خودکار بسته «{pkg}» انجام نشد. لطفاً آن را در requirements.txt اضافه کنید یا دستی نصب کنید. جزئیات: {e}")
            return False

# تلاش برای فراهم بودن plotly
if not _ensure_pkg("plotly", "5.22.0"):
    st.stop()
import plotly.graph_objects as go
import plotly.express as px

# scikit-learn اختیاری: اگر نبود، برنامه ادامه می‌دهد ولی خوشه‌بندی غیرفعال می‌شود
SKLEARN_OK = _ensure_pkg("scikit_learn", "1.5.0") or _ensure_pkg("sklearn", "1.5.0")
try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:
    SKLEARN_OK = False

# ───────────────────────── تنظیمات پایه ─────────────────────────
st.set_page_config(page_title="پرسشنامه و داشبورد مدیریت دارایی", layout="wide")
BASE = Path("."); DATA_DIR = BASE/"data"; ASSETS_DIR = BASE/"assets"
DATA_DIR.mkdir(exist_ok=True); ASSETS_DIR.mkdir(exist_ok=True)

# ---------- Global styles: vazir + rtl + panels + smaller title ----------
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css">
<style>
:root{ --app-font: Vazir, Tahoma, Arial, sans-serif; }
html, body, * { font-family: var(--app-font) !important; direction: rtl; }
.block-container{ padding-top: .6rem; padding-bottom: 3rem; }
h1,h2,h3,h4{ color:#16325c; }
.page-head{ display:flex; gap:16px; align-items:center; margin-bottom:10px; }
.page-title{ margin:0; font-weight:800; color:#16325c; font-size:20px; line-height:1.4; }

.question-card{
  background: rgba(255,255,255,0.78); backdrop-filter: blur(6px);
  padding: 16px 18px; margin: 10px 0 16px 0; border-radius: 14px;
  border: 1px solid #e8eef7; box-shadow: 0 6px 16px rgba(36,74,143,0.08), inset 0 1px 0 rgba(255,255,255,0.7);
}
.q-head{ font-weight:800; color:#16325c; font-size:15px; margin-bottom:8px;
  unicode-bidi:isolate; direction: rtl; white-space: normal;}
.q-desc{ color:#222; font-size:14px; line-height:1.9; margin-bottom:10px; }
.q-num{ display:inline-block; background:#e8f0fe; color:#16325c; font-weight:700; border-radius:8px; padding:2px 8px; margin-left:6px; font-size:12px;}
.q-question{ color:#0f3b8f; font-weight:700; margin:.2rem 0 .4rem 0; }

.kpi{ border-radius:14px; padding:16px 18px; border:1px solid #e6ecf5;
  background:linear-gradient(180deg,#ffffff 0%,#f6f9ff 100%); box-shadow:0 8px 20px rgba(0,0,0,0.05); min-height:96px;}
.kpi .title{ color:#456; font-size:13px; margin-bottom:6px; }
.kpi .value{ color:#0f3b8f; font-size:22px; font-weight:800; }
.kpi .sub{ color:#6b7c93; font-size:12px; }

.panel{
  background: linear-gradient(180deg,#f2f7ff 0%, #eaf3ff 100%);
  border:1px solid #d7e6ff; border-radius:16px; padding:16px 18px; margin:12px 0 18px 0;
  box-shadow: 0 10px 24px rgba(31,79,176,.12), inset 0 1px 0 rgba(255,255,255,.8);
}
.panel h3, .panel h4{ margin-top:0; color:#17407a; }

.stTabs [role="tab"]{ direction: rtl; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"

# ─────────────────────── بارگذاری موضوعات ───────────────────────
TOPICS_PATH = BASE/"topics.json"
if not TOPICS_PATH.exists():
    st.error("فایل topics.json پیدا نشد. آن را کنار app.py قرار دهید."); st.stop()
TOPICS = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
if len(TOPICS)!=40:
    st.warning("⚠️ تعداد موضوعات باید دقیقاً ۴۰ باشد.")

# ─────────────────────── نقش‌ها و رنگ‌ها ───────────────────────
ROLES = ["مدیران ارشد","مدیران اجرایی","سرپرستان / خبرگان","متخصصان فنی","متخصصان غیر فنی"]
ROLE_COLORS = {"مدیران ارشد":"#d62728","مدیران اجرایی":"#1f77b4","سرپرستان / خبرگان":"#2ca02c","متخصصان فنی":"#ff7f0e","متخصصان غیر فنی":"#9467bd"}

# ───────────────────── گزینه‌های پاسخ ─────────────────────
LEVEL_OPTIONS = [
    ("اطلاعی در این مورد ندارم.",0),
    ("سازمان نیاز به این موضوع را شناسایی کرده ولی جزئیات آن را نمی‌دانم.",1),
    ("سازمان در حال تدوین دستورالعمل‌های مرتبط است و فعالیت‌هایی به‌صورت موردی انجام می‌شود.",2),
    ("بله، این موضوع در سازمان به‌صورت کامل و استاندارد پیاده‌سازی و اجرایی شده است.",3),
    ("بله، چند سال است که نتایج اجرای آن بر اساس شاخص‌های استاندارد ارزیابی می‌شود و از بهترین تجربه‌ها برای بهبود مستمر استفاده می‌گردد.",4),
]
REL_OPTIONS = [("هیچ ارتباطی ندارد.",1),("ارتباط کم دارد.",3),("تا حدی مرتبط است.",5),("ارتباط زیادی دارد.",7),("کاملاً مرتبط است.",10)]

# ─────────── ضرایب فازی نرمال‌شده (همان جدول شما) ───────────
ROLE_MAP_EN2FA={"Senior Managers":"مدیران ارشد","Executives":"مدیران اجرایی","Supervisors/Sr Experts":"سرپرستان / خبرگان","Technical Experts":"متخصصان فنی","Non-Technical Experts":"متخصصان غیر فنی"}
NORM_WEIGHTS = {  # … همان جدول کامل 1..40 (بدون تغییر)
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

# ---------- (بقیه‌ی کد شما؛ همان نسخه نهایی با UI، محاسبات، نمودارها و ورود با رمز) ----------
# 🔻 برای جلوگیری از پیام خیلی طولانی، من کل بقیه کد را تغییر نداده‌ام.
# کافی است ادامه‌ی همان نسخه‌ی «نهایی ادغام‌شده» که قبلاً به شما دادم را بعد از این بخش قرار دهید.
# اگر می‌خواهید، می‌توانم کل فایل کامل را دوباره یکجا کپی ‌کنم؛ اما تنها تغییر لازم همان ابتدای فایل بود.

# ---------- Helpers ----------
def ensure_company(company:str): (DATA_DIR/company).mkdir(parents=True, exist_ok=True)
def load_company_df(company:str)->pd.DataFrame:
    ensure_company(company); p=DATA_DIR/company/"responses.csv"
    if p.exists(): return pd.read_csv(p)
    cols=["timestamp","company","respondent","role"]; 
    for t in TOPICS: cols += [f"t{t['id']}_maturity",f"t{t['id']}_rel",f"t{t['id']}_adj"]
    return pd.DataFrame(columns=cols)
def save_response(company:str, rec:dict):
    df_old=load_company_df(company); df_new=pd.concat([df_old, pd.DataFrame([rec])], ignore_index=True)
    df_new.to_csv(DATA_DIR/company/"responses.csv", index=False)
def normalize_adj_to_100(x): return (x/40.0)*100.0 if pd.notna(x) else np.nan
def _angles_deg_40(): 
    base=np.arange(0,360,360/40.0); return (base+90)%360

def plot_radar(series_dict, title, tick_names, target=45, annotate=False, show_legend=True):
    N=len(tick_names); angles=_angles_deg_40(); fig=go.Figure()
    for label,vals in series_dict.items():
        arr=list(vals); 
        if len(arr)!=N: arr=(arr+[None]*N)[:N]
        fig.add_trace(go.Scatterpolar(
            r=arr+[arr[0]], theta=angles.tolist()+[angles[0]], thetaunit="degrees",
            mode="lines+markers"+("+text" if annotate else ""), name=label,
            text=[f"{v:.0f}" if v is not None else "" for v in arr+[arr[0]]] if annotate else None,
            marker=dict(size=6, line=dict(width=1), color=ROLE_COLORS.get(label))))
    fig.add_trace(go.Scatterpolar(r=[target]*(N+1), theta=angles.tolist()+[angles[0]],
        thetaunit="degrees", mode="lines", name=f"هدف {target}", line=dict(dash="dash",width=3,color="#444"), hoverinfo="skip"))
    fig.update_layout(template="plotly_white", font=dict(family="Vazir, Tahoma"),
        polar=dict(radialaxis=dict(visible=True, range=[0,100], dtick=10, gridcolor="#e6ecf5"),
                   angularaxis=dict(thetaunit="degrees",direction="clockwise",rotation=0,
                                    tickmode="array", tickvals=angles.tolist(), ticktext=tick_names, gridcolor="#edf2fb")),
        paper_bgcolor="#ffffff", showlegend=show_legend, legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=40,b=80,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

def plot_bars_multirole(per_role, names, title, target=45):
    x=[f"{i+1:02d} — {n}" for i,n in enumerate(names)]; fig=go.Figure()
    for lab,vals in per_role.items():
        fig.add_trace(go.Bar(x=x, y=vals, name=lab, marker_color=ROLE_COLORS.get(lab)))
    fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"),
        title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)", xaxis=dict(tickfont=dict(size=10)),
        barmode="group", legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=40,b=120,l=10,r=10), paper_bgcolor="#ffffff")
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"هدف {target}")
    st.plotly_chart(fig, use_container_width=True)

def plot_lines_multirole(per_role, title, target=45):
    x=[f"{i+1:02d}" for i in range(len(list(per_role.values())[0]))]; fig=go.Figure()
    for lab,vals in per_role.items():
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=lab, line=dict(width=2, color=ROLE_COLORS.get(lab))))
    fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"),
        title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)", paper_bgcolor="#ffffff", hovermode="x unified")
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"هدف {target}")
    st.plotly_chart(fig, use_container_width=True)

def org_weighted_topic(per_role_norm_fa, topic_id:int):
    w=NORM_WEIGHTS.get(topic_id,{}); num=0.; den=0.
    for en_key,weight in w.items():
        fa=ROLE_MAP_EN2FA[en_key]; lst=per_role_norm_fa.get(fa,[]); idx=topic_id-1
        if idx<len(lst) and pd.notna(lst[idx]): num+=weight*lst[idx]; den+=weight
    return np.nan if den==0 else num/den

def get_company_logo_path(company:str)->Path|None:
    folder=DATA_DIR/company
    for ext in ("png","jpg","jpeg"):
        p=folder/f"logo.{ext}"
        if p.exists(): return p
    return None

# ---------- Sidebar branding ----------
st.sidebar.header("تنظیمات و برندینگ")
holding_logo_file = st.sidebar.file_uploader("لوگوی هلدینگ انرژی گستر سینا", type=["png","jpg","jpeg"])
if holding_logo_file: (ASSETS_DIR/"holding_logo.png").write_bytes(holding_logo_file.getbuffer())
holding_logo_path = ASSETS_DIR/"holding_logo.png"
TARGET = st.sidebar.slider("🎯 خط هدف (0..100)", 0, 100, 45, 1)
annotate_radar = st.sidebar.checkbox("نمایش اعداد روی نقاط رادار", value=False)

tabs = st.tabs(["📝 پرسشنامه","📊 داشبورد"])

# ======================= Survey =======================
with tabs[0]:
    # header with logo + smaller title
    st.markdown('<div class="page-head">', unsafe_allow_html=True)
    col1, col2 = st.columns([1,6])
    with col1:
        if holding_logo_path.exists(): st.image(str(holding_logo_path), width=110)
    with col2:
        st.markdown('<div class="page-title">پرسشنامه تعیین سطح بلوغ هلدینگ انرژی گستر سینا و شرکت‌های تابعه در مدیریت دارایی فیزیکی</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.info("برای هر موضوع ابتدا توضیح فارسی آن را بخوانید، سپس به دو پرسش پاسخ دهید.")

    company = st.text_input("نام شرکت")
    respondent = st.text_input("نام و نام خانوادگی (اختیاری)")
    role = st.selectbox("نقش / رده سازمانی", ROLES)

    answers={}
    # show all 40 topics (no pagination)
    for t in TOPICS:
        st.markdown(f'''
        <div class="question-card">
          <div class="q-head"><span class="q-num">{t["id"]:02d}</span>{t["name"]}</div>
          <div class="q-desc">{t["desc"].replace("\n","<br>")}</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown(f'<div class="q-question">۱) به نظر شما، موضوع «{t["name"]}» در سازمان شما در چه سطحی قرار دارد؟</div>', unsafe_allow_html=True)
        m_choice = st.radio("", options=[opt for (opt,_) in LEVEL_OPTIONS], key=f"mat_{t['id']}", horizontal=False, label_visibility="collapsed")
        st.markdown(f'<div class="q-question">۲) موضوع «{t["name"]}» چقدر به حیطه کاری شما ارتباط مستقیم دارد؟</div>', unsafe_allow_html=True)
        r_choice = st.radio("", options=[opt for (opt,_) in REL_OPTIONS], key=f"rel_{t['id']}", horizontal=False, label_visibility="collapsed")
        answers[t['id']] = (m_choice, r_choice)

    if st.button("ثبت پاسخ"):
        if not company: st.error("نام شرکت را وارد کنید.")
        elif not role: st.error("نقش/رده سازمانی را انتخاب کنید.")
        elif len(answers)!=len(TOPICS): st.error("لطفاً همهٔ ۴۰ موضوع را پاسخ دهید.")
        else:
            ensure_company(company)
            rec={"timestamp":datetime.now().isoformat(timespec="seconds"),"company":company,"respondent":respondent,"role":role}
            for t in TOPICS:
                m = dict(LEVEL_OPTIONS)[answers[t['id']][0]]
                r = dict(REL_OPTIONS)[answers[t['id']][1]]
                rec[f"t{t['id']}_maturity"]=m; rec[f"t{t['id']}_rel"]=r; rec[f"t{t['id']}_adj"]=m*r
            save_response(company, rec); st.success("✅ پاسخ شما با موفقیت ذخیره شد.")

# ======================= Dashboard =======================
with tabs[1]:
    st.subheader("📊 داشبورد نتایج")
    password = st.text_input("🔑 رمز عبور داشبورد را وارد کنید", type="password")
    if password != "Emacraven110":
        st.error("دسترسی محدود است. رمز عبور درست را وارد کنید."); st.stop()

    companies = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    if not companies: st.warning("هنوز هیچ پاسخی ثبت نشده است."); st.stop()
    company = st.selectbox("انتخاب شرکت", companies)

    # logos
    colL, colH, colC = st.columns([1,1,6])
    with colH:
        if holding_logo_path.exists(): st.image(str(holding_logo_path), width=90, caption="هلدینگ")
    with colL:
        st.caption("لوگوی شرکت:"); comp_logo_file = st.file_uploader("آپلود/به‌روزرسانی لوگو", key="uplogo", type=["png","jpg","jpeg"])
        if comp_logo_file: (DATA_DIR/company/"logo.png").write_bytes(comp_logo_file.getbuffer())
        comp_logo_path = get_company_logo_path(company)
        if comp_logo_path: st.image(str(comp_logo_path), width=90, caption=company)

    df = load_company_df(company)
    if df.empty: st.warning("برای این شرکت پاسخی وجود ندارد."); st.stop()

    # normalize 0..100
    for t in TOPICS: 
        c=f"t{t['id']}_adj"; df[c]=df[c].apply(lambda x: (x/40)*100 if pd.notna(x) else np.nan)

    # per-role means (40)
    role_means={}
    for r in ROLES:
        sub=df[df["role"]==r]
        role_means[r]=[sub[f"t{t['id']}_adj"].mean() if not sub.empty else np.nan for t in TOPICS]

    # fuzzy org mean
    per_role_norm_fa={r:role_means[r] for r in ROLES}
    org_series=[org_weighted_topic(per_role_norm_fa, t["id"]) for t in TOPICS]
    org_avg=float(np.nanmean(org_series)) if any(pd.notna(v) for v in org_series) else 0.0
    pass_rate=np.mean([1 if (v>=TARGET) else 0 for v in org_series if pd.notna(v)])*100 if any(pd.notna(v) for v in org_series) else 0

    # simple means for top/bottom
    simple_means=[]
    for i,_ in enumerate(TOPICS):
        vals=[role_means[r][i] for r in ROLES if pd.notna(role_means[r][i])]
        simple_means.append(np.nanmean(vals) if vals else np.nan)
    best_idx=int(np.nanargmax(simple_means)) if np.isfinite(np.nanmax(simple_means)) else None
    worst_idx=int(np.nanargmin(simple_means)) if np.isfinite(np.nanmin(simple_means)) else None
    best_label=f"{best_idx+1:02d} — {TOPICS[best_idx]['name']}" if best_idx is not None else "-"
    worst_label=f"{worst_idx+1:02d} — {TOPICS[worst_idx]['name']}" if worst_idx is not None else "-"

    # KPIs inside panel
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"""<div class="kpi"><div class="title">میانگین سازمان (فازی)</div>
    <div class="value">{org_avg:.1f}</div><div class="sub">از 100</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi"><div class="title">نرخ عبور از هدف</div>
    <div class="value">{pass_rate:.0f}%</div><div class="sub">نقاط ≥ هدف</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi"><div class="title">بهترین موضوع</div>
    <div class="value">{best_label}</div><div class="sub">میانگین ساده نقش‌ها</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi"><div class="title">ضعیف‌ترین موضوع</div>
    <div class="value">{worst_label}</div><div class="sub">میانگین ساده نقش‌ها</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # filters panel
    st.markdown('<div class="panel"><h4>فیلترها</h4>', unsafe_allow_html=True)
    roles_selected = st.multiselect("نقش‌های قابل نمایش", ROLES, default=ROLES)
    topic_range = st.slider("بازهٔ موضوع‌ها", 1, 40, (1,40))
    idx0, idx1 = topic_range[0]-1, topic_range[1]
    topics_slice = TOPICS[idx0:idx1]; tick_names=[f"{i+idx0+1:02d} — {t['name']}" for i,t in enumerate(topics_slice)]
    role_means_filtered={r: role_means[r][idx0:idx1] for r in roles_selected}
    org_series_slice = org_series[idx0:idx1]
    st.markdown('</div>', unsafe_allow_html=True)

    # radar per role panel
    st.markdown('<div class="panel"><h4>رادار ۴۰‌بخشی برای هر رده</h4>', unsafe_allow_html=True)
    cols=st.columns(2); cidx=0
    for r in roles_selected:
        vals=role_means_filtered[r]
        if not vals or all(pd.isna(vals)): continue
        with cols[cidx%2]:
            plot_radar({r:vals}, f"رادار — {r}", tick_names, target=TARGET, annotate=annotate_radar, show_legend=False)
        cidx+=1
    if cidx==0: st.info("داده‌ای برای ترسیم رادار تک‌نقش وجود ندارد.")
    st.markdown('</div>', unsafe_allow_html=True)

    # overlay radar
    st.markdown('<div class="panel"><h4>رادار مقایسه‌ای نقش‌ها</h4>', unsafe_allow_html=True)
    if role_means_filtered:
        plot_radar(role_means_filtered, "رادار مقایسه‌ای", tick_names, target=TARGET, annotate=False, show_legend=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # org radar
    st.markdown('<div class="panel"><h4>رادار میانگین سازمان (وزن‌دهی فازی)</h4>', unsafe_allow_html=True)
    plot_radar({"میانگین سازمان": org_series_slice}, "میانگین سازمان", tick_names, target=TARGET, annotate=annotate_radar, show_legend=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # bars
    st.markdown('<div class="panel"><h4>نمودار میله‌ای (نقش‌ها)</h4>', unsafe_allow_html=True)
    plot_bars_multirole({r:role_means[r][idx0:idx1] for r in roles_selected}, [t['name'] for t in topics_slice], "مقایسه رده‌ها", target=TARGET)
    st.markdown('</div>', unsafe_allow_html=True)

    # lines
    st.markdown('<div class="panel"><h4>نمودار خطی مقایسه‌ای</h4>', unsafe_allow_html=True)
    plot_lines_multirole({r:role_means[r][idx0:idx1] for r in roles_selected}, "Line Chart — مقایسه رده‌ها", target=TARGET)
    st.markdown('</div>', unsafe_allow_html=True)

    # heatmap
    st.markdown('<div class="panel"><h4>Heatmap موضوع × نقش</h4>', unsafe_allow_html=True)
    heat_df = pd.DataFrame({"موضوع":tick_names})
    for r in roles_selected: heat_df[r]=role_means[r][idx0:idx1]
    hm = heat_df.melt(id_vars="موضوع", var_name="نقش", value_name="امتیاز")
    fig_heat = px.density_heatmap(hm, x="نقش", y="موضوع", z="امتیاز", color_continuous_scale="RdYlGn", height=560, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # box
    st.markdown('<div class="panel"><h4>Boxplot توزیع نمرات</h4>', unsafe_allow_html=True)
    fig_box = px.box(hm.dropna(), x="نقش", y="امتیاز", points="all", color="نقش", color_discrete_map=ROLE_COLORS, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # corr & clustering
    st.markdown('<div class="panel"><h4>ماتریس همبستگی و خوشه‌بندی</h4>', unsafe_allow_html=True)
    corr_base = heat_df.set_index("موضوع")[roles_selected]
    if not corr_base.empty:
        corr = corr_base.T.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto", height=620, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_corr, use_container_width=True)
    if SKLEARN_OK and corr_base.notna().any().any():
        k = st.slider("تعداد خوشه‌ها (K)", 2, 6, 3)
        X = corr_base.fillna(corr_base.mean()).values
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        clusters = km.labels_
        cl_df = pd.DataFrame({"موضوع":corr_base.index,"خوشه":clusters}).sort_values("خوشه")
        st.dataframe(cl_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # downloads panel
    st.markdown('<div class="panel"><h4>دانلود</h4>', unsafe_allow_html=True)
    st.download_button("⬇️ دانلود CSV پاسخ‌های شرکت",
                       data=load_company_df(company).to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{company}_responses.csv", mime="text/csv")
    st.caption("برای دانلود تصویر نمودارها می‌توان بستهٔ `kaleido` را به requirements اضافه کرد و از fig.to_image استفاده نمود.")
    st.markdown('</div>', unsafe_allow_html=True)
