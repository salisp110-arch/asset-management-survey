# app.py
# -*- coding: utf-8 -*-
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------------- پیکربندی صفحه ----------------
st.set_page_config(page_title="پرسشنامه و داشبورد مدیریت دارایی", layout="wide")
BASE = Path(".")
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------- نقش‌ها ----------------
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

# ---------------- گزینه‌های پاسخ ----------------
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

# ---------------- بارگذاری موضوعات ----------------
TOPICS_PATH = BASE / "topics.json"
if not TOPICS_PATH.exists():
    st.error("فایل topics.json پیدا نشد. آن را کنار app.py قرار دهید.")
    st.stop()
TOPICS = json.loads((TOPICS_PATH).read_text(encoding="utf-8"))
if len(TOPICS) != 40:
    st.warning("⚠️ تعداد موضوعات باید دقیقاً ۴۰ باشد.")

# ---------------- استایل کارت ----------------
st.markdown("""
<style>
.question-card {
  background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
  padding: 18px 20px;
  margin: 16px 0 26px 0;
  border-radius: 14px;
  border: 1px solid #e8eef7;
  box-shadow: 0 6px 14px rgba(36,74,143,0.08), inset 0 1px 0 rgba(255,255,255,0.7);
}
.q-head { font-weight: 700; color: #1f3b6e; font-size: 16px; margin-bottom: 8px; }
.q-desc { color: #4a5e85; font-size: 14px; line-height: 1.7; margin-bottom: 10px; }
.q-num { display:inline-block; background:#e8f0fe; color:#1f3b6e; font-weight:700; border-radius: 8px; padding: 2px 8px; margin-left: 6px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------- کمک‌تابع‌ها ----------------
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

# ---------------- Sidebar ----------------
st.sidebar.header("تنظیمات داشبورد")
TARGET = st.sidebar.slider("🎯 خط هدف (0..100)", 0, 100, 45, 1)
annotate_radar = st.sidebar.checkbox("نمایش اعداد روی نقاط رادار", value=False)

tabs = st.tabs(["📝 پرسشنامه","📊 داشبورد"])

# ---------------- Survey Tab ----------------
with tabs[0]:
    st.subheader("📝 پرسشنامه مدیریت دارایی (۴۰ موضوع × ۲ پرسش)")
    company = st.text_input("نام شرکت")
    respondent = st.text_input("نام و نام خانوادگی (اختیاری)")
    role = st.selectbox("نقش / رده سازمانی", ROLES)

    st.info("برای هر موضوع ابتدا توضیح فارسی آن را بخوانید، سپس به دو پرسش پاسخ دهید.")
    answers = {}
    for t in TOPICS:
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

    if st.button("ثبت پاسخ"):
        if not company:
            st.error("نام شرکت را وارد کنید.")
        elif not role:
            st.error("نقش/رده سازمانی را انتخاب کنید.")
        elif len(answers) != len(TOPICS):
            st.error("لطفاً همهٔ موضوعات را پاسخ دهید.")
        else:
            ensure_company(company)
            rec = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "company": company, "respondent": respondent, "role": role}
            for t in TOPICS:
                mat = dict(LEVEL_OPTIONS)[answers[t['id']][0]]
                rel = dict(REL_OPTIONS)[answers[t['id']][1]]
                adj = mat * rel
                rec[f"t{t['id']}_maturity"] = mat
                rec[f"t{t['id']}_rel"] = rel
                rec[f"t{t['id']}_adj"] = adj
            save_response(company, rec)
            st.success("✅ پاسخ شما با موفقیت ذخیره شد.")

# ---------------- Dashboard Tab ----------------
with tabs[1]:
    st.subheader("📊 داشبورد نتایج")

    # --- login ---
    password = st.text_input("🔑 رمز عبور داشبورد را وارد کنید", type="password")
    if password != "Emacraven110":
        st.error("دسترسی محدود است. رمز عبور درست را وارد کنید.")
        st.stop()

    # انتخاب شرکت
    companies = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    if not companies:
        st.warning("هنوز هیچ پاسخی ثبت نشده است.")
        st.stop()
    company = st.selectbox("انتخاب شرکت", companies)
    df = load_company_df(company)
    if df.empty:
        st.warning("برای این شرکت پاسخی وجود ندارد.")
        st.stop()

    # محاسبات
    adj_cols = [f"t{t['id']}_adj" for t in TOPICS]
    df_norm = df.copy()
    for c in adj_cols:
        df_norm[c] = df_norm[c].apply(normalize_adj_to_100)

    role_means = {}
    for role in ROLES:
        subset = df_norm[df_norm["role"]==role]
        if subset.empty: continue
        scores = [subset[f"t{t['id']}_adj"].mean() for t in TOPICS]
        role_means[role] = scores
    org_means = np.nanmean(list(role_means.values()), axis=0) if role_means else None

    # ---------------- داشبورد ----------------
    st.markdown("### ✅ شاخص‌های کلیدی (KPI)")
    c1,c2,c3 = st.columns(3)
    if org_means is not None:
        c1.metric("میانگین سازمان", f"{np.nanmean(org_means):.1f}/100")
        c2.metric("بیشترین", f"{np.nanmax(org_means):.1f}")
        c3.metric("کمترین", f"{np.nanmin(org_means):.1f}")

    # نمودار میله‌ای تجمیعی
    st.markdown("### 📊 مقایسه رده‌های سازمانی (میله‌ای)")
    fig = go.Figure()
    for role,scores in role_means.items():
        fig.add_trace(go.Bar(x=[t['name'] for t in TOPICS], y=scores, name=role,
                             marker=dict(color=ROLE_COLORS[role])))
    fig.add_hline(y=TARGET, line_dash="dash", line_color="red", annotation_text="هدف")
    fig.update_layout(barmode="group", xaxis_tickangle=-45, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # نمودار راداری سازمان
    st.markdown("### 🌐 رادار سازمان")
    theta = [t["name"] for t in TOPICS]
    fig = go.Figure()
    for role,scores in role_means.items():
        fig.add_trace(go.Scatterpolar(r=scores+[scores[0]], theta=theta+[theta[0]],
                                      fill="toself", name=role, line_color=ROLE_COLORS[role]))
    if org_means is not None:
        fig.add_trace(go.Scatterpolar(r=org_means.tolist()+[org_means[0]], theta=theta+[theta[0]],
                                      fill="toself", name="میانگین سازمان", line_color="black"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), height=700)
    st.plotly_chart(fig, use_container_width=True)
