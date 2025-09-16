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

# --------------- جدول وزن‌های فازی (Normalized) برای ۴۰ موضوع ---------------
# توجه: کلیدها انگلیسی‌اند، ولی در کد به فارسی نگاشت می‌شوند.
ROLE_MAP_EN2FA = {
    "Senior Managers":"مدیران ارشد",
    "Executives":"مدیران اجرایی",
    "Supervisors/Sr Experts":"سرپرستان / خبرگان",
    "Technical Experts":"متخصصان فنی",
    "Non-Technical Experts":"متخصصان غیر فنی",
}
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

# ---------------- بارگذاری موضوعات ----------------
TOPICS_PATH = BASE / "topics.json"
if not TOPICS_PATH.exists():
    st.error("فایل topics.json پیدا نشد. آن را کنار app.py قرار دهید.")
    st.stop()
TOPICS = json.loads((TOPICS_PATH).read_text(encoding="utf-8"))

# (اختیاری) هشدار در صورت مغایرت ترتیب ۴۰ موضوع با جدول وزن‌ها
if len(TOPICS) != 40:
    st.warning("⚠️ تعداد موضوعات باید دقیقاً ۴۰ باشد تا وزن‌دهی فازی به‌درستی اعمال شود.")

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
    # امتیاز تعدیل‌شده 0..40 → مقیاس 0..100
    return (x/40.0)*100.0 if pd.notna(x) else np.nan

def org_weighted_topic(per_role_norm_fa: dict, topic_id: int):
    """میانگین وزنی سازمان برای موضوع «topic_id» با ضرایب فازی نرمال‌شده."""
    w = NORM_WEIGHTS.get(topic_id, {})
    num=0.0; den=0.0
    for en_key, weight in w.items():
        fa_role = ROLE_MAP_EN2FA[en_key]
        vlist = per_role_norm_fa.get(fa_role, [])
        idx = topic_id-1
        if idx < len(vlist) and pd.notna(vlist[idx]):
            num += weight * vlist[idx]
            den += weight
    return np.nan if den==0 else num/den

def _angles_deg_40():
    base = np.arange(0, 360, 360/40.0)  # 0..351
    return (base + 90) % 360

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
    # خط هدف
    fig.add_trace(go.Scatterpolar(
        r=[target]*(N+1), theta=angles_deg.tolist()+[angles_deg[0]], thetaunit="degrees",
        mode="lines", name=f"هدف {target}", line=dict(dash="dash", width=3, color="#444"), hoverinfo="skip"
    ))
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], tick0=0, dtick=10),
            angularaxis=dict(thetaunit="degrees", direction="clockwise", rotation=0,
                             tickmode="array", tickvals=angles_deg.tolist(),
                             ticktext=tick_names),
            bgcolor="white"
        ),
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
        title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)",
        xaxis=dict(tickfont=dict(size=9)), barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=60,b=140,l=10,r=10)
    )
    fig.add_hline(y=target, line_dash="dash",
                  annotation_text=f"هدف {target}", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

def plot_lines_multirole(per_role_norm_fa: dict, title, target=45):
    x = [f"{i+1:02d}" for i in range(40)]
    fig = go.Figure()
    for label, vals in per_role_norm_fa.items():
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=label,
                                 line=dict(width=2, color=ROLE_COLORS.get(label))))
    fig.update_layout(title=title, xaxis_title="موضوع (01..40)", yaxis_title="نمره (0..100)")
    fig.add_hline(y=target, line_dash="dash",
                  annotation_text=f"هدف {target}", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

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
                adj = mat * rel  # 0..40
                rec[f"t{t['id']}_maturity"] = mat
                rec[f"t{t['id']}_rel"] = rel
                rec[f"t{t['id']}_adj"] = adj
            save_response(company, rec)
            st.success("✅ پاسخ شما با موفقیت ذخیره شد.")

# ---------------- Dashboard Tab ----------------
with tabs[1]:
    st.subheader("📊 داشبورد نتایج (نرمال‌شده و وزن‌دهی فازی)")
    company_a = st.text_input("نام شرکت برای تحلیل")
    if not company_a:
        st.info("نام شرکت موردنظر برای تحلیل را وارد کنید.")
        st.stop()
    df = load_company_df(company_a)
    if df.empty:
        st.warning("برای این شرکت هنوز پاسخی ثبت نشده است.")
        st.stop()

    topic_names = [t["name"] for t in TOPICS]

    # --- میانگین به تفکیک نقش (برای هر موضوع): adj → normalize(0..100)
    per_role_norm_fa = {}
    for role_fa in ROLES:
        sub = df[df["role"]==role_fa]
        if sub.empty:
            per_role_norm_fa[role_fa] = [np.nan]*len(TOPICS)
        else:
            vals = []
            for t in TOPICS:
                adj = sub[f"t{t['id']}_adj"].mean()
                vals.append(normalize_adj_to_100(adj))
            per_role_norm_fa[role_fa] = vals

    # --- میانگین وزنی سازمان (فازی) برای هر موضوع (0..100)
    org_series = [org_weighted_topic(per_role_norm_fa, t["id"]) for t in TOPICS]

    # --- KPI ها
    org_avg = np.nanmean(org_series) if any(pd.notna(x) for x in org_series) else np.nan
    topic_means_simple = []  # میانگین ساده بین نقش‌ها (برای Top/Bottom)
    for idx, t in enumerate(TOPICS):
        vals = [per_role_norm_fa[r][idx] for r in ROLES if pd.notna(per_role_norm_fa[r][idx])]
        topic_means_simple.append(np.nanmean(vals) if vals else np.nan)
    best_idx = int(np.nanargmax(topic_means_simple)) if np.isfinite(np.nanmax(topic_means_simple)) else None
    worst_idx = int(np.nanargmin(topic_means_simple)) if np.isfinite(np.nanmin(topic_means_simple)) else None
    pass_rate = np.mean([1 if (v>=TARGET) else 0 for v in org_series if pd.notna(v)])*100 if any(pd.notna(v) for v in org_series) else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("میانگین کل سازمان (فازی)", f"{org_avg:.1f}" if pd.notna(org_avg) else "-")
    c2.metric("نرخ عبور از هدف", f"{pass_rate:.0f}%")
    c3.metric("بهترین موضوع (میانگین ساده نقش‌ها)", f"{best_idx+1:02d} — {TOPICS[best_idx]['name']}" if best_idx is not None else "-")
    c4.metric("ضعیف‌ترین موضوع (میانگین ساده نقش‌ها)", f"{worst_idx+1:02d} — {TOPICS[worst_idx]['name']}" if worst_idx is not None else "-")

    # --- Top/Bottom 5 (بر اساس میانگین ساده نقش‌ها برای اطلاع مکمل)
    st.markdown("### 🔝 Top 5 و Bottom 5 (میانگین ساده نقش‌ها)")
    topic_mean_series = pd.Series(topic_means_simple, index=[f"{i+1:02d} — {t['name']}" for i,t in enumerate(TOPICS)])
    left, right = st.columns(2)
    with left:
        st.write("**Top 5**")
        st.table(topic_mean_series.sort_values(ascending=False).head(5).round(1))
    with right:
        st.write("**Bottom 5**")
        st.table(topic_mean_series.sort_values(ascending=True).head(5).round(1))

    # --- رادار تک‌نقش
    st.markdown("### 🌐 رادار ۴۰‌بخشی برای هر رده (نمره نرمال‌شده)")
    cols = st.columns(2)
    idx=0
    tick_full = [f"{i+1:02d} — {t['name']}" for i,t in enumerate(TOPICS)]
    for fa in ROLES:
        vals = per_role_norm_fa.get(fa, [])
        if not vals or all(pd.isna(vals)): continue
        with cols[idx%2]:
            plot_radar({fa: vals}, f"رادار — {fa}", tick_full, target=TARGET, annotate=annotate_radar, show_legend=False)
        idx+=1
    if idx==0:
        st.info("داده‌ای برای ترسیم رادار تک‌نقش وجود ندارد.")

    # --- رادار تجمیعی نقش‌ها
    st.markdown("### 🌐 رادار مقایسه‌ای نقش‌ها")
    overlay = {fa: per_role_norm_fa[fa] for fa in ROLES if any(pd.notna(x) for x in per_role_norm_fa[fa])}
    if overlay:
        plot_radar(overlay, "رادار مقایسه‌ای نقش‌ها", tick_full, target=TARGET, annotate=False, show_legend=True)
    else:
        st.info("داده‌ای برای رادار تجمیعی وجود ندارد.")

    # --- رادار میانگین سازمان (وزن‌دهی فازی)
    st.markdown("### 🏛️ رادار میانگین سازمان (وزن‌دهی فازی)")
    plot_radar({"میانگین سازمان": org_series}, "رادار — میانگین سازمان (فازی)", tick_full,
               target=TARGET, annotate=annotate_radar, show_legend=False)

    # --- میله‌ای مقایسه‌ای نقش‌ها
    st.markdown("### 📊 نمودار میله‌ای (نقش‌ها)")
    bars_input = {fa: per_role_norm_fa[fa] for fa in ROLES if any(pd.notna(x) for x in per_role_norm_fa[fa])}
    if bars_input:
        plot_bars_multirole(bars_input, [t['name'] for t in TOPICS], "مقایسه رده‌ها (0..100)", target=TARGET)
    else:
        st.info("هیچ نقشی برای مقایسه موجود نیست.")

    # --- نمودار خطی مقایسه‌ای
    st.markdown("### 📈 نمودار خطی مقایسه‌ای")
    if bars_input:
        plot_lines_multirole(bars_input, "Line Chart — مقایسه رده‌ها", target=TARGET)

    # --- Heatmap (بر اساس میانگین نرمال‌شده نقش‌ها)
    st.markdown("### 🔥 Heatmap موضوعات × نقش‌ها (0..100)")
    heat_df = pd.DataFrame({ "موضوع":[f"{i+1:02d} — {t['name']}" for i,t in enumerate(TOPICS)] })
    for fa in ROLES:
        heat_df[fa] = per_role_norm_fa.get(fa, [np.nan]*len(TOPICS))
    heat_melt = heat_df.melt(id_vars="موضوع", var_name="نقش", value_name="امتیاز")
    fig = px.density_heatmap(heat_melt, x="نقش", y="موضوع", z="امتیاز",
                             color_continuous_scale="RdYlGn", height=700)
    st.plotly_chart(fig, use_container_width=True)

    # --- Boxplot
    st.markdown("### 📦 Boxplot توزیع نمرات")
    fig = px.box(heat_melt.dropna(), x="نقش", y="امتیاز", points="all", color="نقش",
                 color_discrete_map=ROLE_COLORS)
    st.plotly_chart(fig, use_container_width=True)

    # --- همبستگی و خوشه‌بندی موضوعات (از روی ماتریس نقش‌ها)
    st.markdown("### 🔗 ماتریس همبستگی موضوعات و خوشه‌بندی")
    corr_base = heat_df.set_index("موضوع")[ROLES]
    corr = corr_base.T.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto", height=700)
    st.plotly_chart(fig, use_container_width=True)

    if SKLEARN_OK and corr_base.notna().any().any():
        X = corr_base.fillna(corr_base.mean()).values
        k = st.slider("تعداد خوشه‌ها (K)", 2, 6, 3)
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        clusters = km.labels_
        cl_df = pd.DataFrame({
            "موضوع":[f"{i+1:02d} — {t['name']}" for i,t in enumerate(TOPICS)],
            "خوشه":clusters
        }).sort_values("خوشه")
        st.dataframe(cl_df, use_container_width=True)
    else:
        st.info("برای خوشه‌بندی نیاز به scikit-learn و داده کافی است.")
