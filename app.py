import streamlit as st
import cv2
import numpy as np
import datetime
import requests
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
from ultralytics import YOLO
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import time

# Windows alarm sound for 5-minute traffic notifications.
try:
    import winsound
except ImportError:
    winsound = None


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SMART TRAFFIC AI",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# SESSION STATE - MUST BE BEFORE ANY ACCESS
# =========================================================

DEFAULT_STATE = {
    "predicted": False,
    "notification": False,
    "last_check": 0.0,
    "next_check": 0.0,
    "prediction_count": 0,
    "last_congestion": 0,
    "last_prediction_time": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# FUTURISTIC GLACIER UI
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
        linear-gradient(
            rgba(2,10,25,.76),
            rgba(2,10,25,.88)
        ),
        url("https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=2200&q=90");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        padding-top: 1.3rem;
        max-width: 1450px;
    }

    .stApp p,
    .stApp span,
    .stApp label,
    .stApp small {
        color: #ffffff !important;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0,0,0,.8);
    }

    .main-header {
        font-size: 2.65rem;
        font-weight: 900;
        text-align: center;
        color: #ffffff !important;
        text-shadow:
            0 0 10px #38bdf8,
            0 0 25px #0284c7,
            0 3px 10px rgba(0,0,0,.9);
        margin-bottom: 4px;
    }

    .sub-header {
        text-align: center;
        color: #ffffff !important;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 22px;
    }

    .card {
        background: rgba(4,18,38,.88);
        border: 1px solid rgba(125,211,252,.42);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow:
            0 8px 35px rgba(0,0,0,.55),
            inset 0 0 20px rgba(56,189,248,.04);
        backdrop-filter: blur(14px);

        /* Keep plain text inside custom HTML cards bright white.
           Raw text nodes do not inherit the p/span CSS rules. */
        color: #ffffff !important;
    }

    .card *,
    .card h1,
    .card h2,
    .card h3,
    .card h4,
    .card h5,
    .card h6,
    .card p,
    .card span,
    .card div,
    .card li,
    .card small,
    .card b,
    .card strong {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
    }

    /* Preserve the intended colors for special status elements. */
    .active-text {
        color: #86efac !important;
    }

    .route-box,
    .route-box *,
    .alternative-box,
    .alternative-box *,
    .alert-box,
    .alert-box *,
    .info-box,
    .info-box *,
    .monitor-box,
    .monitor-box * {
        color: #ffffff !important;
    }

    .route-box {
        background: rgba(6,78,59,.78);
        border: 1px solid #34d399;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 0 20px rgba(52,211,153,.12);
    }

    .alternative-box {
        background: rgba(120,53,15,.72);
        border: 1px solid #fb923c;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .alert-box {
        background: rgba(127,29,29,.82);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 0 22px rgba(239,68,68,.16);
    }

    .info-box {
        background: rgba(8,47,73,.82);
        border: 1px solid #38bdf8;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .monitor-box {
        background: rgba(4,18,38,.94);
        border: 1px solid #38bdf8;
        border-radius: 17px;
        padding: 19px;
        margin-bottom: 15px;
        box-shadow: 0 0 25px rgba(56,189,248,.16);
    }

    .timer {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 900;
        color: #ffffff !important;
        letter-spacing: 2px;
        margin: 7px 0;
    }

    .active-text {
        text-align: center;
        color: #86efac !important;
        font-weight: 900;
    }

    section[data-testid="stSidebar"] {
        background: rgba(2,12,27,.98);
        border-right: 1px solid rgba(125,211,252,.35);
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div,
    .stTimeInput input {
        background: rgba(255,255,255,.97) !important;
        color: #111827 !important;
        border: 2px solid rgba(56,189,248,.45) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .stTextInput input::placeholder {
        color: #64748b !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg,#06b6d4,#2563eb);
        color: #ffffff !important;
        font-weight: 900;
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 11px;
        padding: 12px;
        box-shadow: 0 0 20px rgba(37,99,235,.35);
    }

    .stButton > button p {
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(5,22,45,.90);
        border: 1px solid rgba(125,211,252,.35);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,.4);
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: #ffffff !important;
    }

    .stCaption,
    div[data-testid="stCaptionContainer"] {
        color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PYTORCH MODEL
# =========================================================

class TrafficModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x)


@st.cache_resource
def load_dl_model():
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.rand(1000, 4) * [50, 24, 50, 45]

    y = (
        X[:, 0] * 70
        + X[:, 1] * 120
        + X[:, 2] * 30
        - X[:, 3] * 5
        + 300
    ).reshape(-1, 1)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    model = TrafficModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    for _ in range(100):
        optimizer.zero_grad()
        prediction = model(x_tensor)
        loss = criterion(prediction, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, scaler_x, scaler_y


dl_model, scaler_x, scaler_y = load_dl_model()


# =========================================================
# YOLO
# =========================================================

@st.cache_resource
def load_yolo():
    try:
        return YOLO("yolov8n.pt")
    except Exception:
        return None


yolo = load_yolo()


# =========================================================
# GEOCODING
# =========================================================

@st.cache_data(ttl=3600)
def geocode(place):
    try:
        if not place or not place.strip():
            return None

        geolocator = Nominatim(
            user_agent="smart_traffic_ai_expo",
            timeout=8,
        )

        query = place.strip()
        location = geolocator.geocode(
            f"{query}, India",
            country_codes="in",
        )

        if location:
            return (float(location.latitude), float(location.longitude))

        location = geolocator.geocode(query)

        if location:
            return (float(location.latitude), float(location.longitude))

    except Exception:
        pass

    return None


# =========================================================
# WEATHER
# =========================================================

@st.cache_data(ttl=300)
def get_weather(lat, lon):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}"
            f"&longitude={lon}"
            "&current=temperature_2m,rain"
        )

        response = requests.get(url, timeout=6)
        response.raise_for_status()

        current = response.json()["current"]

        temperature = float(current.get("temperature_2m", 28.0))
        rainfall = float(current.get("rain", 0.0))

        return temperature, rainfall

    except Exception:
        return 28.0, 0.0


# =========================================================
# REAL ROAD ROUTES - OSRM
# =========================================================

def get_routes(start, destination):
    try:
        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{start[1]},{start[0]};"
            f"{destination[1]},{destination[0]}"
            "?overview=full"
            "&geometries=geojson"
            "&alternatives=true"
            "&steps=false"
        )

        response = requests.get(url, timeout=15)
        response.raise_for_status()

        data = response.json()
        routes = []

        for route in data.get("routes", []):
            geometry = route.get("geometry", {})
            coordinates = geometry.get("coordinates", [])

            if not coordinates:
                continue

            routes.append(
                {
                    "coordinates": coordinates,
                    "distance": float(route["distance"]) / 1000.0,
                    "duration": float(route["duration"]) / 60.0,
                }
            )

        routes.sort(key=lambda item: item["duration"])

        return routes[:2]

    except Exception:
        return []


# =========================================================
# SMART ROUTE SCORING
# =========================================================

def calculate_route(route, route_index, total_routes, congestion, accident):
    # OSRM gives the normal travel time.
    # We estimate extra delay using predicted congestion.
    if total_routes == 1:
        route_traffic = congestion
    elif route_index == 0:
        route_traffic = congestion
    else:
        # Alternative route receives a different estimated traffic load.
        route_traffic = max(
            0,
            min(
                100,
                int(round(congestion * 0.68)),
            ),
        )

    if accident and route_index == 0:
        route_traffic = min(100, route_traffic + 15)

    delay = (route_traffic / 100.0) * 25.0

    if accident and route_index == 0:
        delay += 8.0

    predicted_time = route["duration"] + delay

    # Smart decision:
    # predicted travel time is the strongest factor,
    # followed by traffic and distance.
    score = (
        predicted_time * 1.00
        + route_traffic * 0.20
        + route["distance"] * 0.015
    )

    return {
        **route,
        "traffic": int(route_traffic),
        "delay": round(delay, 1),
        "predicted": round(predicted_time, 1),
        "score": round(score, 2),
    }


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-header">❄️🚦 SMART TRAFFIC AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-header">'
    "Predict • Warn • Recommend • Optimize — "
    "Intelligent Traffic Management System"
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ CONTROL PANEL")

input_type = st.sidebar.radio(
    "Vehicle Input",
    ["📝 Manual Inputs", "🎥 Live Camera (YOLOv8)"],
)

if input_type == "📝 Manual Inputs":
    manual_count = st.sidebar.slider(
        "🚗 Current Vehicles",
        min_value=0,
        max_value=100,
        value=35,
    )
else:
    manual_count = 0

st.sidebar.subheader("🚨 Safety & Traffic")

is_emergency = st.sidebar.checkbox(
    "🚑 Emergency Corridor",
    value=False,
)

accident = st.sidebar.checkbox(
    "⚠️ Accident / Road Hazard",
    value=False,
)

st.sidebar.subheader("📍 Journey")

start_name = st.sidebar.text_input(
    "Start Location",
    value="Hyderabad",
)

destination_name = st.sidebar.text_input(
    "Destination",
    value="Vijayawada",
)

target_time = st.sidebar.time_input(
    "Target Time",
    value=datetime.datetime.now().time(),
)

st.sidebar.caption(
    "Tip: Use city names or specific places in India."
)


# =========================================================
# LOCATIONS
# =========================================================

start_location = geocode(start_name)
destination_location = geocode(destination_name)

if not start_location:
    st.error(
        f"❌ Could not find start location: **{start_name}**. "
        "Please enter a valid Indian location."
    )
    st.stop()

if not destination_location:
    st.error(
        f"❌ Could not find destination: **{destination_name}**. "
        "Please enter a valid Indian location."
    )
    st.stop()


# =========================================================
# WEATHER
# =========================================================

temperature, rainfall = get_weather(
    start_location[0],
    start_location[1],
)


# =========================================================
# VEHICLE DETECTION
# =========================================================

detected_count = manual_count

if input_type == "🎥 Live Camera (YOLOv8)":
    run_camera = st.sidebar.checkbox(
        "📷 Turn On Camera",
        value=False,
    )

    if run_camera:
        camera_placeholder = st.empty()
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        if ret and yolo is not None:
            try:
                results = yolo(
                    frame,
                    verbose=False,
                )[0]

                target_classes = {
                    "car",
                    "motorcycle",
                    "bus",
                    "truck",
                }

                detected_count = 0

                for box in results.boxes:
                    class_id = int(box.cls[0])
                    class_name = yolo.names.get(
                        class_id,
                        "",
                    )

                    if class_name in target_classes:
                        detected_count += 1

                annotated_frame = results.plot()

                cv2.putText(
                    annotated_frame,
                    f"Vehicles: {detected_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                camera_placeholder.image(
                    cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB,
                    ),
                    caption=(
                        f"YOLOv8 Vehicle Detection — "
                        f"{detected_count} vehicles"
                    ),
                )

            except Exception as exc:
                st.warning(
                    f"YOLO detection issue: {exc}"
                )

        elif ret:
            camera_placeholder.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                ),
                caption="Camera frame",
            )

        else:
            st.error(
                "❌ Camera could not be opened. "
                "Use Manual Inputs for the expo demo."
            )

        cap.release()


# =========================================================
# JOURNEY INFORMATION
# =========================================================

j1, j2, j3 = st.columns(3)

with j1:
    st.metric(
        "📍 START",
        start_name,
    )

with j2:
    st.metric(
        "🏁 DESTINATION",
        destination_name,
    )

with j3:
    st.metric(
        "🕐 TARGET TIME",
        target_time.strftime("%I:%M %p"),
    )


# =========================================================
# WEATHER
# =========================================================

w1, w2, w3 = st.columns(3)

with w1:
    st.metric(
        "🌡️ Temperature",
        f"{temperature:.1f} °C",
    )

with w2:
    st.metric(
        "🌧️ Rain",
        f"{rainfall:.1f} mm",
    )

with w3:
    st.metric(
        "🚗 Vehicles Detected",
        str(detected_count),
    )


# =========================================================
# RUN BUTTON
# =========================================================

st.markdown("---")
st.subheader("🔮 AI TRAFFIC ANALYSIS")

predict_button = st.button(
    "🚀 RUN AI TRAFFIC PREDICTION",
    type="primary",
)

if predict_button:
    st.session_state.predicted = True
    st.session_state.notification = False
    st.session_state.last_check = time.time()
    st.session_state.next_check = time.time() + 300
    st.session_state.prediction_count += 1


# =========================================================
# NO PREDICTION YET
# =========================================================

if not st.session_state.predicted:
    st.markdown(
        """
        <div class="card">
        <h2>🚦 SMART TRAFFIC AI</h2>
        <p>
        Enter your journey in the left control panel and run
        the AI prediction.
        </p>
        <br>
        🗺️ Real road routing using OSRM<br><br>
        🟢 Smart recommended route<br><br>
        🟠 Alternative route<br><br>
        🔮 Traffic Ahead prediction<br><br>
        🚦 Congestion prediction<br><br>
        ⏱️ Predicted delay<br><br>
        🧠 AI Decision Center<br><br>
        🔔 Automatic 5-minute monitoring<br><br>
        🚗 YOLOv8 vehicle detection
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # =====================================================
    # AUTOMATIC 5-MINUTE RECHECK
    # =====================================================

    now = time.time()

    if st.session_state.next_check <= 0:
        st.session_state.next_check = now + 300

    if now >= st.session_state.next_check:
        monitoring_congestion = st.session_state.last_congestion
        st.session_state.notification = True
        st.session_state.last_check = now
        st.session_state.next_check = now + 300
        st.session_state.prediction_count += 1

        # Small deterministic demo variation.
        # This represents a new AI observation cycle.
        variation = int(
            np.random.default_rng(
                st.session_state.prediction_count
            ).integers(-3, 4)
        )

        detected_count = max(
            0,
            int(detected_count + variation),
        )

        # =================================================
        # 5-MINUTE ALARM / TRAFFIC-AHEAD NOTIFICATION
        # =================================================

        traffic_ahead_text = (
            "🚨 TRAFFIC AHEAD DETECTED — Slow traffic is predicted on your route!"
            if monitoring_congestion >= 40
            else
            "🔔 AI TRAFFIC UPDATE — Traffic conditions checked. Route is currently normal."
        )

        st.toast(
            traffic_ahead_text,
            icon="🚨" if monitoring_congestion >= 40 else "🔔",
        )

        # Real Windows alarm. This is more reliable than browser autoplay.
        if winsound is not None:
            try:
                if monitoring_congestion >= 75:
                    for freq, duration in [
                        (1100, 350),
                        (700, 350),
                        (1100, 500),
                    ]:
                        winsound.Beep(freq, duration)
                elif monitoring_congestion >= 40:
                    winsound.Beep(1000, 300)
                    time.sleep(0.12)
                    winsound.Beep(1250, 450)
                else:
                    winsound.Beep(850, 220)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass

        # Browser sound fallback. Browser autoplay may be blocked by policy.
        browser_freqs = (
            [(1100, 0.32), (700, 0.32), (1100, 0.45)]
            if monitoring_congestion >= 75
            else
            [(1000, 0.30), (1250, 0.45)]
            if monitoring_congestion >= 40
            else
            [(850, 0.22)]
        )

        browser_beeps = ""
        offset = 0.05
        for freq, duration in browser_freqs:
            browser_beeps += (
                f"beep(ctx.currentTime + {offset:.2f}, {freq}, {duration:.2f});"
            )
            offset += duration + 0.12

        st.markdown(
            f"""
            <script>
            (() => {{
                try {{
                    const AudioContext = window.AudioContext ||
                                          window.webkitAudioContext;
                    if (!AudioContext) return;

                    const ctx = new AudioContext();

                    const beep = (when, frequency, duration) => {{
                        const osc = ctx.createOscillator();
                        const gain = ctx.createGain();

                        osc.type = "sine";
                        osc.frequency.setValueAtTime(frequency, when);

                        gain.gain.setValueAtTime(0.0001, when);
                        gain.gain.exponentialRampToValueAtTime(0.30, when + 0.02);
                        gain.gain.exponentialRampToValueAtTime(
                            0.0001, when + duration
                        );

                        osc.connect(gain);
                        gain.connect(ctx.destination);
                        osc.start(when);
                        osc.stop(when + duration + 0.03);
                    }};

                    if (ctx.state === "suspended") {{
                        ctx.resume().then(() => {{
                            {browser_beeps}
                        }});
                    }} else {{
                        {browser_beeps}
                    }}
                }} catch (error) {{
                    console.log("Browser notification sound blocked:", error);
                }}
            }})();
            </script>
            """,
            unsafe_allow_html=True,
        )


    # =====================================================
    # AI PREDICTION
    # =====================================================

    raw_input = np.array(
        [
            [
                detected_count,
                target_time.hour,
                rainfall,
                temperature,
            ]
        ],
        dtype=float,
    )

    scaled_input = scaler_x.transform(raw_input)

    with torch.no_grad():
        output = dl_model(
            torch.tensor(
                scaled_input,
                dtype=torch.float32,
            )
        ).numpy()

    predicted_volume = float(
        scaler_y.inverse_transform(output)[0][0]
    )

    predicted_volume = max(
        0.0,
        predicted_volume,
    )

    if accident:
        predicted_volume += 1500

    congestion = min(
        100,
        max(
            0,
            int(
                round(
                    (predicted_volume / 4000.0)
                    * 100
                )
            ),
        ),
    )

    # =====================================================
    # STATUS + SIGNAL ACTION
    # =====================================================

    if is_emergency:
        status = "EMERGENCY"
        signal_action = (
            "GREEN CORRIDOR — PRIORITY SIGNAL"
        )

    elif congestion < 40:
        status = "LOW"
        signal_action = (
            "STANDARD SIGNAL — 30 sec"
        )

    elif congestion < 75:
        status = "MODERATE"
        signal_action = (
            "GREEN EXTENSION — +15 sec"
        )

    else:
        status = "HIGH"
        signal_action = (
            "MAX GREEN EXTENSION — +45 sec & REROUTE"
        )

    # =====================================================
    # REAL ROUTES
    # =====================================================

    routes = get_routes(
        start_location,
        destination_location,
    )

    if not routes:
        st.error(
            "❌ OSRM could not return a real road route "
            "right now. Please try again in a few seconds."
        )
        st.stop()

    smart_routes = [
        calculate_route(
            route,
            index,
            len(routes),
            congestion,
            accident,
        )
        for index, route in enumerate(routes)
    ]

    recommended = min(
        smart_routes,
        key=lambda item: item["score"],
    )

    alternatives = [
        item
        for item in smart_routes
        if item is not recommended
    ]

    alternative = (
        alternatives[0]
        if alternatives
        else None
    )

    distance = recommended["distance"]
    base_duration = recommended["duration"]
    predicted_duration = recommended["predicted"]
    delay = recommended["delay"]

    if alternative:
        time_saved = round(
            max(
                0.0,
                alternative["predicted"]
                - recommended["predicted"],
            ),
            1,
        )
    else:
        time_saved = 0.0

    # Save latest values.
    st.session_state.last_congestion = congestion
    st.session_state.last_prediction_time = (
        datetime.datetime.now().strftime(
            "%I:%M:%S %p"
        )
    )

    # =====================================================
    # NOTIFICATION
    # =====================================================

    if st.session_state.notification:
        notification_class = (
            "alert-box"
            if congestion >= 60
            else "info-box"
        )

        st.markdown(
            f"""
            <div class="{notification_class}">
            <b>🚨 AI TRAFFIC AHEAD ALERT</b><br><br>
            ⏱️ 5-minute monitoring cycle completed.<br>
            🚦 Traffic ahead prediction:
            <b>{congestion}% congestion</b><br>
            📍 Predicted traffic zone:
            <b>{max(1.0, distance * 0.35):.1f} km ahead</b><br>
            ⏱️ Expected delay:
            <b>{delay:.1f} min</b><br>
            🧠 Smart route automatically recalculated.<br>
            🗺️ Recommended route:
            <b>{distance:.1f} km</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.notification = False

    # =====================================================
    # TOP METRICS
    # =====================================================

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "🚗 Vehicles",
            str(detected_count),
        )

    with m2:
        st.metric(
            "🔮 Predicted Volume",
            f"{int(predicted_volume)} / hr",
        )

    with m3:
        st.metric(
            "🚦 Congestion",
            f"{congestion}%",
        )

    with m4:
        st.metric(
            "⏱️ Expected Delay",
            f"{delay:.1f} min",
        )

    # =====================================================
    # MAP + DECISION CENTER
    # =====================================================

    map_column, decision_column = st.columns(
        [1.45, 1]
    )

    # =====================================================
    # MAP
    # =====================================================

    with map_column:
        st.subheader(
            "🗺️ LIVE TRAFFIC & SMART ROUTES"
        )

        center_lat = (
            start_location[0]
            + destination_location[0]
        ) / 2.0

        center_lon = (
            start_location[1]
            + destination_location[1]
        ) / 2.0

        traffic_map = folium.Map(
            location=[
                center_lat,
                center_lon,
            ],
            zoom_start=7,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # Start marker
        folium.Marker(
            start_location,
            tooltip="📍 START",
            popup=start_name,
            icon=folium.Icon(
                color="blue",
                icon="play",
            ),
        ).add_to(traffic_map)

        # Destination marker
        folium.Marker(
            destination_location,
            tooltip="🏁 DESTINATION",
            popup=destination_name,
            icon=folium.Icon(
                color="green",
                icon="flag",
            ),
        ).add_to(traffic_map)

        # Draw every available route.
        for route in smart_routes:
            is_best = route is recommended

            points = [
                (point[1], point[0])
                for point in route["coordinates"]
            ]

            folium.PolyLine(
                points,
                color=(
                    "#22c55e"
                    if is_best
                    else "#f97316"
                ),
                weight=8 if is_best else 6,
                opacity=0.92,
                dash_array=(
                    None
                    if is_best
                    else "10,8"
                ),
                tooltip=(
                    "🟢 SMART RECOMMENDED ROUTE"
                    if is_best
                    else "🟠 ALTERNATIVE ROUTE"
                ),
                popup=(
                    f"Route — "
                    f"{route['distance']:.1f} km — "
                    f"{route['predicted']:.0f} min"
                ),
            ).add_to(traffic_map)

        # Traffic ahead marker on recommended route.
        route_coordinates = recommended["coordinates"]

        if len(route_coordinates) > 4:
            traffic_index = int(
                len(route_coordinates) * 0.35
            )

            traffic_index = max(
                1,
                min(
                    len(route_coordinates) - 2,
                    traffic_index,
                ),
            )

            point = route_coordinates[traffic_index]

            traffic_point = [
                point[1],
                point[0],
            ]

            traffic_color = (
                "red"
                if congestion >= 75
                else "orange"
                if congestion >= 40
                else "green"
            )

            folium.Marker(
                traffic_point,
                tooltip="🔮 TRAFFIC AHEAD",
                popup=(
                    f"🔮 Traffic Ahead<br>"
                    f"Predicted congestion: "
                    f"{congestion}%<br>"
                    f"Expected delay: "
                    f"{delay:.1f} min"
                ),
                icon=folium.Icon(
                    color=traffic_color,
                    icon="warning-sign",
                ),
            ).add_to(traffic_map)

            folium.Circle(
                traffic_point,
                radius=(
                    900
                    if congestion >= 75
                    else 650
                    if congestion >= 40
                    else 450
                ),
                color=traffic_color,
                fill=True,
                fill_opacity=0.20,
            ).add_to(traffic_map)

        # Vehicle indicators along route.
        if len(route_coordinates) > 10:
            for fraction in (0.25, 0.50, 0.75):
                point = route_coordinates[
                    int(
                        len(route_coordinates)
                        * fraction
                    )
                ]

                folium.Marker(
                    [point[1], point[0]],
                    tooltip="🚗 Vehicle Traffic",
                    icon=folium.DivIcon(
                        html="""
                        <div style="
                            font-size:21px;
                            text-shadow:
                            0 0 5px white;
                        ">🚗</div>
                        """
                    ),
                ).add_to(traffic_map)

        st_folium(
            traffic_map,
            width=None,
            height=500,
            returned_objects=[],
        )

    # =====================================================
    # AI DECISION CENTER
    # =====================================================

    with decision_column:
        st.subheader(
            "🧠 AI DECISION CENTER"
        )

        if is_emergency:
            st.markdown(
                """
                <div class="alert-box">
                <b>🚑 EMERGENCY CORRIDOR ACTIVE</b><br><br>
                Priority green corridor recommendation is active.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif congestion >= 75:
            st.markdown(
                f"""
                <div class="alert-box">
                <b>🚨 HIGH TRAFFIC PREDICTED</b><br><br>
                🔮 Traffic ahead detected.<br><br>
                📍 Approx. distance ahead:
                <b>{max(1.0, distance * .35):.1f} km</b><br><br>
                ⏱️ Expected delay:
                <b>{delay:.1f} min</b><br><br>
                🛣️ Alternative can save:
                <b>{time_saved:.1f} min</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif congestion >= 40:
            st.markdown(
                f"""
                <div class="info-box">
                <b>⚠️ MODERATE TRAFFIC PREDICTED</b><br><br>
                Expected delay:
                <b>{delay:.1f} minutes</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                """
                <div class="info-box">
                <b>🟢 TRAFFIC IS CURRENTLY NORMAL</b><br><br>
                AI found no major congestion risk
                on the recommended route.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # RECOMMENDED ROUTE
        # -------------------------------------------------

        traffic_label = (
            "HIGH"
            if recommended["traffic"] >= 75
            else "MODERATE"
            if recommended["traffic"] >= 40
            else "LOW"
        )

        save_line = (
            f"✅ Saves <b>{time_saved:.1f} min</b> "
            "compared with alternative"
            if alternative
            else
            "ℹ️ OSRM returned only one route"
        )

        st.markdown(
            f"""
            <div class="route-box">
            <b>🟢 RECOMMENDED ROUTE — SMART DECISION</b>
            <br><br>
            📍 {start_name}<br>
            ↓<br>
            🏁 {destination_name}<br><br>

            📏 Distance:
            <b>{distance:.1f} km</b><br>

            ⏱️ Normal Time:
            <b>{base_duration:.0f} min</b><br>

            🔮 Predicted Time:
            <b>{predicted_duration:.0f} min</b><br>

            🚦 Traffic:
            <b>{traffic_label} ({recommended["traffic"]}%)</b><br>

            ⏱️ Predicted Delay:
            <b>{delay:.1f} min</b><br>

            🧠 Smart Score:
            <b>{recommended["score"]:.2f}</b><br><br>

            {save_line}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # ALTERNATIVE ROUTE
        # -------------------------------------------------

        if alternative:
            alternative_label = (
                "HIGH"
                if alternative["traffic"] >= 75
                else "MODERATE"
                if alternative["traffic"] >= 40
                else "LOW"
            )

            st.markdown(
                f"""
                <div class="alternative-box">
                <b>🟠 ALTERNATIVE ROUTE</b><br><br>

                📏 Distance:
                <b>{alternative["distance"]:.1f} km</b><br>

                ⏱️ Normal Time:
                <b>{alternative["duration"]:.0f} min</b><br>

                🔮 Predicted Time:
                <b>{alternative["predicted"]:.0f} min</b><br>

                🚦 Traffic:
                <b>{alternative_label}
                ({alternative["traffic"]}%)</b><br>

                ⏱️ Predicted Delay:
                <b>{alternative["delay"]:.1f} min</b><br>

                🧠 Smart Score:
                <b>{alternative["score"]:.2f}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # GAUGE
        # -------------------------------------------------

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=congestion,
                number={"suffix": "%"},
                title={
                    "text": "Congestion Level"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "steps": [
                        {
                            "range": [0, 40],
                            "color": "#14532d",
                        },
                        {
                            "range": [40, 75],
                            "color": "#854d0e",
                        },
                        {
                            "range": [75, 100],
                            "color": "#7f1d1d",
                        },
                    ],
                    "bar": {
                        "color": "#38bdf8"
                    },
                },
            )
        )

        gauge.update_layout(
            height=280,
            margin=dict(
                l=15,
                r=15,
                t=50,
                b=10,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

    # =====================================================
    # 5-MINUTE MONITORING
    # =====================================================

    st.subheader(
        "🔔 5-MINUTE PREDICTIVE MONITORING"
    )

    remaining = max(
        0,
        int(
            st.session_state.next_check
            - time.time()
        ),
    )

    minutes, seconds = divmod(
        remaining,
        60,
    )

    st.markdown(
        f"""
        <div class="monitor-box">
        <b>🔔 NEXT AI TRAFFIC CHECK</b>

        <div class="timer">
        ⏱️ {minutes:02d}:{seconds:02d}
        </div>

        <div class="active-text">
        🟢 MONITORING ACTIVE
        </div>

        <div style="
            text-align:center;
            margin-top:9px;
        ">
        AI will automatically re-check traffic
        conditions after 5 minutes.
        </div>

        <div style="
            text-align:center;
            margin-top:8px;
            color:#cbd5e1 !important;
        ">
        Prediction cycle:
        {st.session_state.prediction_count}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Last AI prediction: "
        + st.session_state.last_prediction_time
    )

    # Browser refresh every second.
    # This keeps the countdown alive and triggers the
    # 5-minute server-side recheck automatically.
    st.markdown(
        """
        <script>
        setTimeout(function () {
            window.parent.location.reload();
        }, 1000);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # SIGNAL CONTROL
    # =====================================================

    st.subheader(
        "🚦 AUTONOMOUS SIGNAL CONTROL"
    )

    if is_emergency:
        st.success(
            "🚑 Emergency priority: "
            "GREEN CORRIDOR recommendation active."
        )

    elif congestion >= 75:
        st.error(
            "🔴 HIGH TRAFFIC: "
            "Extend green signal by 45 seconds "
            "and recommend rerouting."
        )

    elif congestion >= 40:
        st.warning(
            "🟠 MODERATE TRAFFIC: "
            "Extend green signal by 15 seconds."
        )

    else:
        st.success(
            "🟢 LOW TRAFFIC: "
            "Standard 30-second signal cycle."
        )

    # =====================================================
    # WEATHER / AI STATUS
    # =====================================================

    st.subheader("🌦️ LIVE ENVIRONMENT & AI STATUS")

    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric(
            "🌡️ Temperature",
            f"{temperature:.1f} °C",
        )

    with e2:
        st.metric(
            "🌧️ Rain",
            f"{rainfall:.1f} mm",
        )

    with e3:
        st.metric(
            "🧠 AI Status",
            status,
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "❄️ SMART TRAFFIC AI • "
    "YOLOv8 + PyTorch + Weather API + "
    "OpenStreetMap / OSRM • "
    "Expo Demonstration"
)
