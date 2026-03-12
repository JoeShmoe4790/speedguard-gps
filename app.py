import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SpeedGuard GPS", page_icon="🛣️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
html, body, [class*="css"] { background-color: #080c14; color: #c8d8e8; }
[data-testid="stSidebar"] { background: #060a10 !important; border-right: 1px solid #0a3a5a; }
[data-testid="stSidebar"] * { color: #7ab3cc !important; }
.hud-title { font-family: 'Orbitron', monospace; font-size: 26px; font-weight: 900; color: #00cfff; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 2px; }
.hud-subtitle { font-family: 'Share Tech Mono', monospace; font-size: 11px; color: #1a5a7a; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 14px; }
.speed-hud { font-family: 'Orbitron', monospace; font-size: 72px; font-weight: 900; line-height: 1; text-align: center; }
.speed-unit { font-family: 'Share Tech Mono', monospace; font-size: 12px; letter-spacing: 0.3em; text-align: center; color: #1a5a7a; margin-top: 4px; }
.alert-low { background: #001a0d; border: 1px solid #00ff88; border-left: 4px solid #00ff88; border-radius: 8px; padding: 10px 16px; font-family: 'Orbitron', monospace; font-size: 12px; color: #00ff88; letter-spacing: 0.1em; margin-bottom: 10px; }
.alert-medium { background: #1a0f00; border: 1px solid #ff9500; border-left: 4px solid #ff9500; border-radius: 8px; padding: 10px 16px; font-family: 'Orbitron', monospace; font-size: 12px; color: #ff9500; letter-spacing: 0.1em; margin-bottom: 10px; }
.alert-high { background: #1a0000; border: 1px solid #ff3333; border-left: 4px solid #ff3333; border-radius: 8px; padding: 10px 16px; font-family: 'Orbitron', monospace; font-size: 12px; color: #ff3333; letter-spacing: 0.1em; margin-bottom: 10px; }
.hud-card { background: #060e18; border: 1px solid #0a3a5a; border-top: 2px solid #00cfff; border-radius: 8px; padding: 14px 10px; text-align: center; margin-bottom: 8px; }
.hud-card-label { font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 0.2em; color: #1a5a7a; text-transform: uppercase; margin-bottom: 6px; white-space: nowrap; }
.hud-card-value { font-family: 'Orbitron', monospace; font-size: 24px; font-weight: 700; color: #00cfff; white-space: nowrap; }
.hud-card-value.danger { color: #ff3333; }
.hud-card-value.warn { color: #ff9500; }
.hud-card-value.safe { color: #00ff88; }
.hud-section { font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 0.3em; color: #1a5a7a; text-transform: uppercase; border-bottom: 1px solid #0a2a3a; padding-bottom: 6px; margin-bottom: 12px; }
.warning-strip { border: 1px solid #ff9500; border-radius: 6px; padding: 7px 12px; font-family: 'Share Tech Mono', monospace; font-size: 10px; color: #ff9500; letter-spacing: 0.12em; margin-top: 6px; }
.coord-bar { font-family: 'Share Tech Mono', monospace; font-size: 10px; color: #1a4a6a; letter-spacing: 0.1em; border-top: 1px solid #0a2a3a; padding-top: 8px; margin-top: 8px; }
.location-pin { font-family: 'Share Tech Mono', monospace; font-size: 11px; color: #00cfff; background: #060e18; border: 1px solid #0a3a5a; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("model.pkl not found — run python train_model.py first")
        st.stop()

model = load_model()

def predict(over, hr, hw, lim, constr=False):
    try:
        p = model.predict_proba([[over, hr, int(hw), lim, 1 if (7<=hr<=9 or 16<=hr<=18) else 0, 1 if (hr>=22 or hr<=5) else 0]])[0][1] * 100
    except:
        p = model.predict_proba([[over, hr, int(hw), lim]])[0][1] * 100
    if constr:
        p = min(p * 1.8, 99)
    return round(p, 1)

with st.sidebar:
    st.markdown("### 🛣️ SPEEDGUARD GPS")
    st.markdown('<p style="font-size:10px;letter-spacing:0.2em;color:#0a4a6a;font-family:monospace;">ROUTE PARAMETERS</p>', unsafe_allow_html=True)
    st.divider()
    speed_limit  = st.selectbox("Speed limit (mph)", [55, 60, 65, 70, 75], index=2)
    your_speed   = st.slider("Your speed (mph)", min_value=speed_limit, max_value=speed_limit+40, value=speed_limit+10, step=1)
    hour         = st.slider("Hour of day", 0, 23, 14, format="%d:00")
    st.divider()
    is_highway   = st.toggle("🛣️ Freeway / Highway", value=True)
    construction = st.toggle("🚧 Construction Zone", value=False)
    st.divider()
    speed_over = your_speed - speed_limit
    if hour in range(7,10) or hour in range(16,19):
        t_status, t_color = "🔴 RUSH HOUR — HIGH PATROL", "#ff3333"
    elif hour >= 22 or hour <= 5:
        t_status, t_color = "🟡 LATE NIGHT — ELEVATED", "#ff9500"
    else:
        t_status, t_color = "🟢 OFF-PEAK — NORMAL", "#00ff88"
    st.markdown(f'<p style="font-family:monospace;font-size:11px;color:{t_color};letter-spacing:0.1em;">{t_status}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:monospace;font-size:10px;color:#0a4a6a;">TIME: {hour:02d}:00 | +{speed_over} MPH OVER</p>', unsafe_allow_html=True)

speed_over   = your_speed - speed_limit
risk         = predict(speed_over, hour, is_highway, speed_limit, construction)
speeds_range = list(range(0, 41))
risk_curve   = [predict(s, hour, is_highway, speed_limit, construction) for s in speeds_range]
sweet_spot   = next((s for s, r in zip(speeds_range, risk_curve) if r > 30), None)

if risk < 25:
    risk_color, risk_label, alert_class = "#00ff88", "CLEAR", "alert-low"
    alert_msg = "▲ ALL CLEAR — RISK WITHIN SAFE PARAMETERS"
elif risk < 55:
    risk_color, risk_label, alert_class = "#ff9500", "CAUTION", "alert-medium"
    alert_msg = "⚠ CAUTION — ELEVATED ENFORCEMENT PROBABILITY"
else:
    risk_color, risk_label, alert_class = "#ff3333", "DANGER", "alert-high"
    alert_msg = "✖ WARNING — HIGH TICKET RISK DETECTED"

st.markdown('<div class="hud-title">⬡ SpeedGuard GPS</div>', unsafe_allow_html=True)
st.markdown('<div class="hud-subtitle">Enforcement Risk Navigation System · California Corridor</div>', unsafe_allow_html=True)
st.markdown(f'<div class="{alert_class}">{alert_msg}</div>', unsafe_allow_html=True)
st.divider()

col_speed, col_gauge, col_map = st.columns([1.2, 1.3, 2], gap="large")

with col_speed:
    st.markdown('<div class="hud-section">◈ speed monitor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="speed-hud" style="color:{risk_color}">{your_speed}</div>', unsafe_allow_html=True)
    st.markdown('<div class="speed-unit">MPH CURRENT</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        vc = "danger" if speed_over > 15 else "warn" if speed_over > 7 else "safe"
        st.markdown(f'<div class="hud-card"><div class="hud-card-label">Over</div><div class="hud-card-value {vc}">+{speed_over}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="hud-card"><div class="hud-card-label">Limit</div><div class="hud-card-value">{speed_limit}</div></div>', unsafe_allow_html=True)
    with c3:
        rc = "danger" if risk >= 55 else "warn" if risk >= 25 else "safe"
        st.markdown(f'<div class="hud-card" style="border-top-color:{risk_color}"><div class="hud-card-label">Risk</div><div class="hud-card-value {rc}">{risk}%</div></div>', unsafe_allow_html=True)

    if construction:
        st.markdown('<div class="warning-strip">⚠ CONSTRUCTION ZONE · 1.8x MULTIPLIER</div>', unsafe_allow_html=True)
    if sweet_spot is not None:
        st.markdown(f'<div class="coord-bar">▸ SAFE: STAY UNDER +{sweet_spot} MPH OVER</div>', unsafe_allow_html=True)

with col_gauge:
    st.markdown('<div class="hud-section">◈ risk gauge</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=risk,
        number={"suffix": "%", "font": {"size": 30, "color": risk_color, "family": "Orbitron"}},
        gauge={
            "axis": {"range": [0,100], "tickcolor": "#0a3a5a", "tickfont": {"color": "#1a5a7a", "size": 9}},
            "bar": {"color": risk_color, "thickness": 0.22},
            "bgcolor": "#060e18", "borderwidth": 1, "bordercolor": "#0a3a5a",
            "steps": [{"range": [0,25], "color": "#001a0d"}, {"range": [25,55], "color": "#1a0f00"}, {"range": [55,100], "color": "#1a0000"}],
            "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.8, "value": risk}
        },
        title={"text": f"<span style='font-size:10px;color:#1a5a7a;font-family:monospace;letter-spacing:3px'>{risk_label}</span>"},
    ))
    fig_gauge.update_layout(height=220, margin=dict(t=30,b=0,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d8e8")
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('<div class="hud-section">◈ risk curve</div>', unsafe_allow_html=True)
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=speeds_range, y=risk_curve, mode="lines", line=dict(color="#00cfff", width=2), fill="tozeroy", fillcolor="rgba(0,207,255,0.05)"))
    fig_curve.add_trace(go.Scatter(x=[speed_over], y=[risk], mode="markers", marker=dict(size=10, color=risk_color, line=dict(width=2, color="#fff"))))
    fig_curve.add_hline(y=30, line_dash="dot", line_color="#0a3a5a", annotation_text="30%", annotation_font_color="#1a5a7a", annotation_font_size=9)
    fig_curve.update_layout(
        height=190, margin=dict(t=10,b=30,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
        xaxis=dict(title="mph over", color="#1a5a7a", gridcolor="#060e18", zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(title="risk %", color="#1a5a7a", gridcolor="#0a1a2a", range=[0,105], zeroline=False, tickfont=dict(size=9)),
        font=dict(family="Share Tech Mono", color="#1a5a7a")
    )
    st.plotly_chart(fig_curve, use_container_width=True)

with col_map:
    st.markdown('<div class="hud-section">◈ click map to set location — enforcement heatmap</div>', unsafe_allow_html=True)

    # Default location
    if "selected_lat" not in st.session_state:
        st.session_state.selected_lat = 36.7783
    if "selected_lng" not in st.session_state:
        st.session_state.selected_lng = -119.4179

    @st.cache_data
    def make_heatmap():
        np.random.seed(42)
        n = 400
        lats = np.concatenate([np.random.normal(34.05,0.7,n), np.random.normal(37.77,0.5,n), np.random.normal(38.58,0.4,n)])
        lngs = np.concatenate([np.random.normal(-118.24,0.9,n), np.random.normal(-122.42,0.7,n), np.random.normal(-121.49,0.5,n)])
        tickets = np.random.beta(2,4,n*3)
        return pd.DataFrame({"lat": lats, "lng": lngs, "ticket": tickets})

    heatmap_df = make_heatmap()

    m = folium.Map(location=[st.session_state.selected_lat, st.session_state.selected_lng],
                   zoom_start=6, tiles="CartoDB dark_matter")

    from folium.plugins import HeatMap
    HeatMap(heatmap_df[["lat","lng","ticket"]].values.tolist(), radius=24, blur=20, min_opacity=0.35,
            gradient={"0.2":"#003366","0.5":"#ff9500","0.8":"#ff2200"}).add_to(m)

    # Selected location marker
    folium.CircleMarker(
        location=[st.session_state.selected_lat, st.session_state.selected_lng],
        radius=12, color=risk_color, fill=True, fill_color=risk_color,
        fill_opacity=0.9, tooltip=f"📍 Selected location | Risk: {risk}%"
    ).add_to(m)

    folium.Marker(
        location=[st.session_state.selected_lat, st.session_state.selected_lng],
        tooltip=f"Risk: {risk}%",
        icon=folium.DivIcon(html=f'<div style="font-family:monospace;font-size:10px;color:{risk_color};background:#060e18;border:1px solid {risk_color};padding:2px 6px;border-radius:4px;white-space:nowrap;">RISK: {risk}%</div>')
    ).add_to(m)

    map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"])

    # Update location on click
    if map_data and map_data.get("last_clicked"):
        st.session_state.selected_lat = round(map_data["last_clicked"]["lat"], 4)
        st.session_state.selected_lng = round(map_data["last_clicked"]["lng"], 4)
        st.rerun()

    st.markdown(
        f'<div class="coord-bar">◈ SELECTED: {st.session_state.selected_lat}°N {abs(st.session_state.selected_lng)}°W &nbsp;|&nbsp; CORRIDOR: CA-99 / I-5 / I-80 &nbsp;|&nbsp; STATUS: {risk_label}</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.markdown('<p style="font-family:monospace;font-size:10px;color:#0a2a3a;letter-spacing:0.15em;text-align:center;">⬡ SPEEDGUARD GPS v2.0 · FOR EDUCATIONAL USE ONLY · ALWAYS OBEY POSTED SPEED LIMITS</p>', unsafe_allow_html=True)