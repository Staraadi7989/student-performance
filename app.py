
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px; }
#MainMenu, footer { visibility: hidden; }

.hero {
    padding: 2rem 2.3rem; border-radius: 24px; color: #fff; margin-bottom: 1.3rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
    box-shadow: 0 24px 60px -24px rgba(124,58,237,.7);
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -.02em; }
.hero p  { margin: .4rem 0 0 0; opacity: .92; font-size: 1.02rem; }
.pill {
    display: inline-block; padding: .28rem .8rem; margin: .8rem .5rem 0 0;
    border-radius: 999px; font-size: .78rem; font-weight: 600;
    background: rgba(255,255,255,.18); backdrop-filter: blur(6px);
}

.card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
    border-radius: 20px; padding: 1.15rem 1.3rem; height: 100%;
    transition: transform .2s ease, border-color .2s ease;
}
.card:hover { transform: translateY(-3px); border-color: rgba(167,139,250,.55); }
.card .label { font-size: .76rem; text-transform: uppercase; letter-spacing: .09em; opacity: .65; font-weight: 600; }
.card .value { font-size: 2.3rem; font-weight: 800; line-height: 1.15; margin-top: .25rem; }
.card .sub   { font-size: .85rem; opacity: .7; margin-top: .15rem; }

.tip {
    display: flex; gap: .9rem; align-items: flex-start; padding: .9rem 1.1rem; margin-bottom: .7rem;
    border-radius: 16px; background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08); border-left: 4px solid var(--c, #8b5cf6);
}
.tip .ic { font-size: 1.5rem; }
.tip b { display: block; margin-bottom: .1rem; }
.tip span { opacity: .78; font-size: .92rem; }

.section-title { font-size: 1.2rem; font-weight: 700; margin: 1.2rem 0 .6rem 0; }

div[data-baseweb="tab-list"] { gap: .4rem; }
button[data-baseweb="tab"] { border-radius: 12px !important; padding: .55rem 1.1rem !important; font-weight: 600; }
div.stButton > button, div.stDownloadButton > button {
    border-radius: 12px; font-weight: 600; border: 0; color: #fff;
    background: linear-gradient(135deg, #6366f1, #a855f7);
}
div.stButton > button:hover, div.stDownloadButton > button:hover { filter: brightness(1.12); color: #fff; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.07); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Feature definitions
# ----------------------------------------------------------------------------
FEATURES = [
    dict(key="study_hours", label="Weekly study hours", lo=0, hi=40, default=12, step=1, group="📚 Academics",
         radar="Study"),
    dict(key="attendance", label="Attendance (%)", lo=40, hi=100, default=85, step=1, group="📚 Academics",
         radar="Attendance"),
    dict(key="previous_score", label="Previous exam score", lo=0, hi=100, default=65, step=1, group="📚 Academics",
         radar="Past score"),
    dict(key="assignments", label="Assignments completed (%)", lo=0, hi=100, default=80, step=5, group="📚 Academics",
         radar="Assignments"),
    dict(key="participation", label="Class participation (1-10)", lo=1, hi=10, default=6, step=1, group="📚 Academics",
         radar="Participation"),
    dict(key="tutoring", label="Extra tutoring (hrs/week)", lo=0, hi=10, default=1, step=1, group="📚 Academics",
         radar="Tutoring"),
    dict(key="sleep_hours", label="Sleep per night (hrs)", lo=3.0, hi=10.0, default=7.0, step=0.5, group="🧠 Wellbeing",
         radar="Sleep"),
    dict(key="stress", label="Stress level (1-10)", lo=1, hi=10, default=5, step=1, group="🧠 Wellbeing",
         radar="Low stress"),
    dict(key="screen_time", label="Social media (hrs/day)", lo=0.0, hi=10.0, default=3.0, step=0.5,
         group="🧠 Wellbeing", radar="Focus"),
]
KEYS = [f["key"] for f in FEATURES] + ["parental_support"]
LABELS = {f["key"]: f["label"] for f in FEATURES} | {"parental_support": "Parental support"}
FMAP = {f["key"]: f for f in FEATURES}
SUPPORT = {"Low": 1, "Medium": 2, "High": 3}
SUPPORT_INV = {v: k for k, v in SUPPORT.items()}

PRESETS = {
    "Custom": None,
    "🌱 Needs support": dict(study_hours=3, attendance=60, previous_score=42, assignments=45, participation=3,
                            tutoring=0, sleep_hours=5.5, stress=8, screen_time=6.5, parental_support="Low"),
    "📘 Average student": dict(study_hours=12, attendance=85, previous_score=65, assignments=80, participation=6,
                              tutoring=1, sleep_hours=7.0, stress=5, screen_time=3.0, parental_support="Medium"),
    "🚀 High achiever": dict(study_hours=24, attendance=97, previous_score=88, assignments=98, participation=9,
                            tutoring=2, sleep_hours=7.5, stress=3, screen_time=1.5, parental_support="High"),
}

GRADES = [(90, "A+", "#22c55e"), (80, "A", "#4ade80"), (70, "B+", "#a3e635"), (60, "B", "#facc15"),
          (50, "C", "#fb923c"), (40, "D", "#f97316"), (0, "F", "#ef4444")]


def grade_of(score: float) -> tuple[str, str]:
    for cut, g, c in GRADES:
        if score >= cut:
            return g, c
    return "F", "#ef4444"


def norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
def generate_data(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "study_hours": np.clip(rng.normal(12, 6, n), 0, 40),
        "attendance": np.clip(rng.normal(82, 12, n), 40, 100),
        "previous_score": np.clip(rng.normal(65, 15, n), 10, 100),
        "assignments": np.clip(rng.normal(78, 18, n), 0, 100),
        "participation": np.clip(rng.normal(5.5, 2, n), 1, 10).round(),
        "tutoring": np.clip(rng.exponential(1.2, n), 0, 10).round(),
        "sleep_hours": np.clip(rng.normal(6.9, 1.1, n), 3, 10).round(1),
        "stress": np.clip(rng.normal(5, 2, n), 1, 10).round(),
        "screen_time": np.clip(rng.normal(3.5, 1.8, n), 0, 10).round(1),
        "parental_support": rng.choice([1, 2, 3], n, p=[.25, .5, .25]),
    })
    score = (
        8
        + 0.35 * d.previous_score
        + 18 * (1 - np.exp(-d.study_hours / 10))
        + 0.12 * d.attendance
        + 0.07 * d.assignments
        - 1.2 * (d.sleep_hours - 7.5) ** 2
        + 0.8 * d.participation
        - 0.9 * d.stress
        - 0.8 * d.screen_time
        + 0.9 * d.tutoring
        + 2.0 * d.parental_support
        + rng.normal(0, 4, n)
    )
    d["final_score"] = np.clip(score, 0, 100)
    return d


def radar_norm(key: str, v: float) -> float:
    if key == "parental_support":
        return (v - 1) / 2 * 100
    f = FMAP[key]
    pct = (v - f["lo"]) / (f["hi"] - f["lo"]) * 100
    if key in ("stress", "screen_time"):
        return 100 - pct
    if key == "sleep_hours":
        return max(0.0, 100 - abs(v - 7.5) * 30)
    return pct


@st.cache_resource(show_spinner="Training the prediction model…")
def train_model() -> dict:
    df = generate_data()
    X, y = df[KEYS], df["final_score"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=7
    ).fit(Xtr, ytr)
    pred = np.clip(model.predict(Xte), 0, 100)
    top = df[df.final_score >= 80][KEYS].mean()
    return dict(
        model=model, df=df,
        r2=r2_score(yte, pred),
        mae=mean_absolute_error(yte, pred),
        rmse=float(np.sqrt(np.mean((yte - pred) ** 2))),
        importance=pd.Series(model.feature_importances_, index=KEYS).sort_values(),
        actual=yte.to_numpy()[:500], predicted=pred[:500],
        top_profile={k: radar_norm(k, top[k]) for k in KEYS},
    )


M = train_model()
model = M["model"]


def predict_many(rows: list[dict] | pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame(rows)[KEYS]
    return np.clip(model.predict(X), 0, 100)


def predict_one(vals: dict) -> float:
    return float(predict_many([vals])[0])


# ----------------------------------------------------------------------------
# Sidebar (inputs)
# ----------------------------------------------------------------------------
for f in FEATURES:
    st.session_state.setdefault(f"in_{f['key']}", f["default"])
st.session_state.setdefault("in_parental_support", "Medium")


def apply_preset() -> None:
    p = PRESETS[st.session_state["preset"]]
    if p:
        for k, v in p.items():
            st.session_state[f"in_{k}"] = v


with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    student_name = st.text_input("Student name (optional)", placeholder="e.g. Aarav Sharma")
    st.selectbox("Quick profile", list(PRESETS), key="preset", on_change=apply_preset)

    current_group = None
    for f in FEATURES:
        if f["group"] != current_group:
            current_group = f["group"]
            st.markdown(f"**{current_group}**")
        st.slider(f["label"], f["lo"], f["hi"], step=f["step"], key=f"in_{f['key']}")
    st.markdown("**🏠 Environment**")
    st.select_slider("Parental support", ["Low", "Medium", "High"], key="in_parental_support")

    st.divider()
    pass_mark = st.slider("Pass mark", 30, 70, 50, help="Used to compute pass probability and risk level.")

vals = {f["key"]: st.session_state[f"in_{f['key']}"] for f in FEATURES}
vals["parental_support"] = SUPPORT[st.session_state["in_parental_support"]]

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <h1>🎓 Student Performance Prediction AI</h1>
  <p>Predict academic performance, discover what moves the needle, and plan smarter study strategies.</p>
  <span class="pill">Gradient Boosting model</span>
  <span class="pill">R² {M['r2']:.2f}</span>
  <span class="pill">Avg. error ±{M['mae']:.1f} pts</span>
  <span class="pill">{len(M['df']):,} training records</span>
</div>""",
    unsafe_allow_html=True,
)


def style(fig: go.Figure, h: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=45, b=10), height=h, font=dict(family="Inter, sans-serif"),
    )
    return fig


def card(label: str, value: str, sub: str = "", color: str = "inherit") -> str:
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>')


def tip(icon: str, title: str, text: str, color: str) -> str:
    return (f'<div class="tip" style="--c:{color}"><div class="ic">{icon}</div>'
            f'<div><b>{title}</b><span>{text}</span></div></div>')


tab_pred, tab_whatif, tab_insights, tab_batch, tab_about = st.tabs(
    ["🎯 Prediction", "🧪 What-If Lab", "📊 Insights", "📁 Batch Predict", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# Tab 1 - Prediction
# ----------------------------------------------------------------------------
with tab_pred:
    pred = predict_one(vals)
    rmse = M["rmse"]
    lo, hi = max(0, pred - 1.96 * rmse), min(100, pred + 1.96 * rmse)
    p_pass = 1 - norm_cdf((pass_mark - pred) / rmse)
    pct = float((M["df"].final_score < pred).mean() * 100)
    grade, gcol = grade_of(pred)
    risk, rcol = (("Low risk", "#22c55e") if p_pass >= .9 else
                  ("Moderate risk", "#facc15") if p_pass >= .6 else ("High risk", "#ef4444"))

    who = f"**{student_name}** · " if student_name.strip() else ""
    st.markdown(f"{who}Predicted outcome based on the current profile in the sidebar.")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Predicted score", f"{pred:.1f}", f"Likely range {lo:.0f} – {hi:.0f}"), unsafe_allow_html=True)
    c2.markdown(card("Grade", grade, "Based on predicted score", gcol), unsafe_allow_html=True)
    c3.markdown(card("Pass probability", f"{p_pass * 100:.0f}%", f"Pass mark {pass_mark} · {risk}", rcol),
                unsafe_allow_html=True)
    c4.markdown(card("Class percentile", f"{pct:.0f}th", "Compared with training cohort"), unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-title">Score gauge</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pred,
            delta=dict(reference=float(M["df"].final_score.mean()), suffix=" vs avg"),
            number=dict(suffix="/100", font=dict(size=46)),
            gauge=dict(
                axis=dict(range=[0, 100]), bar=dict(color=gcol, thickness=.28), bgcolor="rgba(0,0,0,0)",
                steps=[dict(range=[0, 40], color="rgba(239,68,68,.25)"),
                       dict(range=[40, 60], color="rgba(249,115,22,.25)"),
                       dict(range=[60, 75], color="rgba(250,204,21,.22)"),
                       dict(range=[75, 100], color="rgba(34,197,94,.25)")],
                threshold=dict(line=dict(color="white", width=3), value=pass_mark),
            ),
        ))
        st.plotly_chart(style(gauge, 340), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Profile vs. top performers</div>', unsafe_allow_html=True)
        cats = [LABELS[k] for k in KEYS]
        me = [radar_norm(k, vals[k]) for k in KEYS]
        top = [M["top_profile"][k] for k in KEYS]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=top + top[:1], theta=cats + cats[:1], name="Top performers (80+)",
                                        line=dict(color="#22c55e", dash="dot"), fill="toself",
                                        fillcolor="rgba(34,197,94,.10)"))
        radar.add_trace(go.Scatterpolar(r=me + me[:1], theta=cats + cats[:1], name="This student",
                                        line=dict(color="#a78bfa", width=3), fill="toself",
                                        fillcolor="rgba(167,139,250,.28)"))
        radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False),
                                       bgcolor="rgba(0,0,0,0)"),
                            legend=dict(orientation="h", y=-.08))
        st.plotly_chart(style(radar, 340), use_container_width=True)

    # Recommendations
    st.markdown('<div class="section-title">Personalised recommendations</div>', unsafe_allow_html=True)
    tips = []
    if vals["study_hours"] < 8:
        tips.append(("📖", "Build a study routine", "Under 8 hours a week is limiting. Aim for 10-15 hours in focused 45-minute blocks.", "#8b5cf6"))
    if vals["attendance"] < 75:
        tips.append(("🏫", "Improve attendance", "Attendance below 75% is strongly linked to lower scores. Every missed class adds up.", "#ef4444"))
    if vals["assignments"] < 70:
        tips.append(("📝", "Finish assignments on time", "Completing assignments reinforces learning and directly lifts predicted performance.", "#f97316"))
    if vals["sleep_hours"] < 6.5 or vals["sleep_hours"] > 9:
        tips.append(("😴", "Fix the sleep schedule", "Performance peaks around 7-8 hours of sleep. Both too little and too much hurt.", "#38bdf8"))
    if vals["stress"] >= 7:
        tips.append(("🧘", "Manage stress", "High stress drags scores down. Try short breaks, exercise, and talking to a counsellor or mentor.", "#ec4899"))
    if vals["screen_time"] >= 5:
        tips.append(("📵", "Cut distraction time", "Over 5 hours of social media a day competes directly with study time. Try app limits.", "#facc15"))
    if vals["participation"] <= 3:
        tips.append(("🙋", "Participate more", "Asking one question per class improves understanding and retention.", "#14b8a6"))
    if vals["tutoring"] == 0 and vals["previous_score"] < 60:
        tips.append(("👩‍🏫", "Consider tutoring", "Even 2 hours of weekly tutoring shows a measurable gain for students in this range.", "#a855f7"))
    if vals["parental_support"] == 1:
        tips.append(("🤝", "Find a mentor", "Teachers, seniors, or study groups can provide the support that is missing at home.", "#22c55e"))
    if not tips:
        tips.append(("🌟", "You're on track!", "No major risk factors detected. Keep the habits going and use the What-If Lab to push higher.", "#22c55e"))
    st.markdown("".join(tip(*t) for t in tips[:5]), unsafe_allow_html=True)

    report = (
        f"EduPredict AI report\n{'=' * 24}\nStudent: {student_name or 'N/A'}\n"
        f"Predicted score: {pred:.1f}/100 (range {lo:.0f}-{hi:.0f})\nGrade: {grade}\n"
        f"Pass probability (pass mark {pass_mark}): {p_pass * 100:.0f}% - {risk}\n"
        f"Percentile: {pct:.0f}th\n\nInputs\n------\n"
        + "\n".join(f"{LABELS[k]}: {SUPPORT_INV[vals[k]] if k == 'parental_support' else vals[k]}" for k in KEYS)
        + "\n\nRecommendations\n---------------\n" + "\n".join(f"- {t[1]}: {t[2]}" for t in tips[:5])
    )
    st.download_button("⬇️ Download report", report, file_name="student_report.txt")

# ----------------------------------------------------------------------------
# Tab 2 - What-if lab
# ----------------------------------------------------------------------------
with tab_whatif:
    st.markdown("Change habits below and see how the predicted score responds, starting from the sidebar profile.")
    base = predict_one(vals)

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown('<div class="section-title">Scenario builder</div>', unsafe_allow_html=True)
        d_study = st.slider("Study hours change", -10, 20, 5)
        d_att = st.slider("Attendance change (%)", -20, 30, 5)
        d_asg = st.slider("Assignments change (%)", -30, 40, 10)
        d_sleep = st.slider("Sleep change (hrs)", -3.0, 3.0, 0.0, 0.5)
        d_stress = st.slider("Stress change", -5, 5, -1)
        d_screen = st.slider("Social media change (hrs)", -5.0, 5.0, -1.0, 0.5)
        d_tutor = st.slider("Tutoring change (hrs)", -3, 8, 1)

        scen = dict(vals)
        for k, dv in [("study_hours", d_study), ("attendance", d_att), ("assignments", d_asg),
                      ("sleep_hours", d_sleep), ("stress", d_stress), ("screen_time", d_screen),
                      ("tutoring", d_tutor)]:
            scen[k] = float(np.clip(vals[k] + dv, FMAP[k]["lo"], FMAP[k]["hi"]))
        new = predict_one(scen)
        delta = new - base
        st.markdown(card("Scenario score", f"{new:.1f}", f"{delta:+.1f} points vs current ({base:.1f})",
                         "#22c55e" if delta > 0 else "#ef4444" if delta < 0 else "inherit"),
                    unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Which single change helps most?</div>', unsafe_allow_html=True)
        levers = [
            ("Study +5 h/week", "study_hours", min(FMAP["study_hours"]["hi"], vals["study_hours"] + 5)),
            ("Attendance +10%", "attendance", min(100, vals["attendance"] + 10)),
            ("Assignments +15%", "assignments", min(100, vals["assignments"] + 15)),
            ("Sleep → 7.5 h", "sleep_hours", 7.5),
            ("Participation +2", "participation", min(10, vals["participation"] + 2)),
            ("Stress −2", "stress", max(1, vals["stress"] - 2)),
            ("Social media −1.5 h", "screen_time", max(0, vals["screen_time"] - 1.5)),
            ("Tutoring +2 h", "tutoring", min(10, vals["tutoring"] + 2)),
        ]
        rows = [{**vals, k: v} for _, k, v in levers]
        gains = predict_many(rows) - base
        lev = pd.DataFrame({"Change": [l[0] for l in levers], "Gain": gains}).sort_values("Gain")
        fig = px.bar(lev, x="Gain", y="Change", orientation="h", color="Gain",
                     color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
                     labels={"Gain": "Score change (pts)", "Change": ""})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style(fig, 340), use_container_width=True)

        st.markdown('<div class="section-title">Study hours vs. predicted score</div>', unsafe_allow_html=True)
        hrs = np.arange(0, 41, 1)
        sweep = predict_many([{**vals, "study_hours": float(h)} for h in hrs])
        fig2 = go.Figure(go.Scatter(x=hrs, y=sweep, mode="lines", line=dict(color="#a78bfa", width=4),
                                    fill="tozeroy", fillcolor="rgba(167,139,250,.12)", name="Predicted"))
        fig2.add_trace(go.Scatter(x=[vals["study_hours"]], y=[base], mode="markers",
                                  marker=dict(size=14, color="#22c55e", line=dict(color="white", width=2)),
                                  name="Current"))
        fig2.update_layout(xaxis_title="Weekly study hours", yaxis_title="Predicted score",
                           yaxis=dict(range=[0, 100]))
        st.plotly_chart(style(fig2, 300), use_container_width=True)
        st.caption("Notice the diminishing returns: the first extra hours help most.")

# ----------------------------------------------------------------------------
# Tab 3 - Insights
# ----------------------------------------------------------------------------
with tab_insights:
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="section-title">What drives performance?</div>', unsafe_allow_html=True)
        imp = M["importance"]
        fig = px.bar(x=imp.values * 100, y=[LABELS[k] for k in imp.index], orientation="h",
                     color=imp.values, color_continuous_scale="Purples",
                     labels={"x": "Importance (%)", "y": ""})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style(fig, 380), use_container_width=True)
    with b:
        st.markdown('<div class="section-title">Where does this student sit?</div>', unsafe_allow_html=True)
        fig = px.histogram(M["df"], x="final_score", nbins=40, color_discrete_sequence=["#6366f1"],
                           labels={"final_score": "Final score"})
        fig.add_vline(x=predict_one(vals), line_color="#22c55e", line_width=3, annotation_text="This student")
        fig.add_vline(x=pass_mark, line_color="#ef4444", line_dash="dash", annotation_text="Pass mark",
                      annotation_position="bottom right")
        st.plotly_chart(style(fig, 380), use_container_width=True)

    c, d = st.columns(2)
    with c:
        st.markdown('<div class="section-title">Model accuracy (held-out test data)</div>', unsafe_allow_html=True)
        fig = px.scatter(x=M["actual"], y=M["predicted"], opacity=.55, color_discrete_sequence=["#a78bfa"],
                         labels={"x": "Actual score", "y": "Predicted score"})
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="#22c55e", dash="dash"))
        st.plotly_chart(style(fig, 360), use_container_width=True)
    with d:
        st.markdown('<div class="section-title">Feature correlation</div>', unsafe_allow_html=True)
        corr = M["df"].corr()
        corr.index = corr.columns = [LABELS.get(k, "Final score") for k in corr.columns]
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(style(fig, 360), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 4 - Batch predict
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown("Upload a CSV with one row per student to predict an entire class at once.")
    template = M["df"][KEYS].head(10).round(1)
    t1, t2 = st.columns([1, 1])
    t1.download_button("⬇️ Download CSV template", template.to_csv(index=False), file_name="students_template.csv")
    use_sample = t2.checkbox("Use 25 sample students instead", value=False)

    up = st.file_uploader("Upload CSV", type="csv")
    data = None
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")

    elif use_sample:
        data = M["df"][KEYS].sample(25, random_state=3).round(1).reset_index(drop=True)

    if data is not None:
        missing = [k for k in KEYS if k not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing) + ". Download the template to see the format.")
        else:
            clean = data.copy()
            for k in KEYS:
                lo_k = 1 if k == "parental_support" else FMAP[k]["lo"]
                hi_k = 3 if k == "parental_support" else FMAP[k]["hi"]
                clean[k] = pd.to_numeric(clean[k], errors="coerce").clip(lo_k, hi_k)
            clean = clean.dropna(subset=KEYS)
            out = clean.copy()
            out["predicted_score"] = predict_many(out).round(1)
            out["grade"] = [grade_of(s)[0] for s in out.predicted_score]
            out["pass_probability_%"] = [round((1 - norm_cdf((pass_mark - s) / M["rmse"])) * 100)
                                         for s in out.predicted_score]
            out["risk"] = np.where(out["pass_probability_%"] >= 90, "Low",
                                   np.where(out["pass_probability_%"] >= 60, "Moderate", "High"))

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(card("Students", f"{len(out)}"), unsafe_allow_html=True)
            k2.markdown(card("Average score", f"{out.predicted_score.mean():.1f}"), unsafe_allow_html=True)
            k3.markdown(card("At high risk", f"{(out.risk == 'High').sum()}", color="#ef4444"),
                        unsafe_allow_html=True)
            k4.markdown(card("Predicted to pass", f"{(out.predicted_score >= pass_mark).mean() * 100:.0f}%",
                             color="#22c55e"), unsafe_allow_html=True)

            st.dataframe(
                out.sort_values("predicted_score"), use_container_width=True, hide_index=True,
                column_config={"predicted_score": st.column_config.ProgressColumn(
                    "Predicted score", min_value=0, max_value=100, format="%.1f")},
            )
            gdist = out.grade.value_counts().reindex([g for _, g, _ in GRADES]).fillna(0).reset_index()
            gdist.columns = ["Grade", "Students"]
            fig = px.bar(gdist, x="Grade", y="Students", color="Grade",
                         color_discrete_map={g: c for _, g, c in GRADES})
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 300), use_container_width=True)
            st.download_button("⬇️ Download predictions", out.to_csv(index=False),
                               file_name="predictions.csv")

# ----------------------------------------------------------------------------
# Tab 5 - About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
### How it works
EduPredict AI trains a **Gradient Boosting Regressor** (scikit-learn) on a synthetic dataset of
{len(M['df']):,} students. The relationships encode well-known education research patterns, such as
diminishing returns from study time and an optimal sleep window of about 7-8 hours.

| Metric | Value |
|---|---|
| R² (test) | {M['r2']:.3f} |
| Mean absolute error | {M['mae']:.2f} points |
| RMSE | {M['rmse']:.2f} points |

**Pass probability** assumes prediction errors are roughly normal with the model's RMSE.
**Likely range** is the predicted score ± 1.96 × RMSE (about 95% of cases).

### Use your own data
Replace `generate_data()` in `app.py` with `pd.read_csv("your_data.csv")`. Keep the same column names
(the nine numeric features, `parental_support` as 1-3, and a `final_score` target) and the whole UI
will work with your real records.

> ⚠️ This is a decision-support tool, not a verdict on any student. Predictions are estimates based on
> the data it was trained on and should be combined with teacher judgement.
"""
    )
