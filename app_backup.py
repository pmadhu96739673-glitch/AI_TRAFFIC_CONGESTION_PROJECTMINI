import streamlit as st
import cv2
import pandas as pd
import numpy as np
import datetime
import requests
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
from ultralytics import YOLO

# PyTorch Deep Learning
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SMART TRAFFIC MANAGEMENT SYSTEM",
    page_icon="🚦",
    layout="wide"
)

# --- CLEAN MODERN STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00f2fe;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 25px;
    }
    .card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 10px;
        border-radius: 8px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- PYTORCH DEEP LEARNING MODEL ---
class TrafficModel(nn.Module):
    def __init__(self):
        super(TrafficModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.fc(x)

@st.cache_resource
def load_dl_model():
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Synthetic Data Training
    X = np.random.rand(1000, 4) * [50, 24, 50, 45] # Vehicles, Hour, Rain, Temp
    y = (X[:, 0] * 70 + X[:, 1] * 120 + X[:, 2] * 30 - X[:, 3] * 5 + 300).reshape(-1, 1)
    
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y)
    
    model = TrafficModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(torch.tensor(X_s, dtype=torch.float32)), torch.tensor(y_s, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model, scaler_X, scaler_y

dl_model, scaler_X, scaler_y = load_dl_model()

# --- YOLO MODEL ---
@st.cache_resource
def load_yolo():
    try:
        return YOLO("yolov8n.pt")
    except:
        return None

yolo = load_yolo()

# --- WEATHER API ---
def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,rain"
        res = requests.get(url, timeout=3).json()
        return res['current']['temperature_2m'], res['current']['rain']
    except:
        return 28.0, 0.0

# --- HEADER ---
st.markdown('<div class="main-header">🚦 SMART TRAFFIC MANAGEMENT & PREDICTION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Real-Time Computer Vision & Deep Learning Traffic Control</div>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Control Panel")
input_type = st.sidebar.radio("Select Input Source:", ["🎥 Live Camera (CV Detection)", "📝 Manual Inputs"])

st.sidebar.subheader("🚨 Emergency Settings")
is_emergency = st.sidebar.checkbox("Emergency Corridor Mode (Ambulance)")
accident = st.sidebar.checkbox("Road Hazard / Accident Simulation")

st.sidebar.subheader("📍 Location Settings")
city_name = st.sidebar.text_input("City Junction:", "Hyderabad")
target_time = st.sidebar.time_input("Target Time:", datetime.datetime.now().time())

# Fetch Weather & Coordinates
geolocator = Nominatim(user_agent="traffic_app", timeout=5)
try:
    loc = geolocator.geocode(city_name)
    lat, lon = (loc.latitude, loc.longitude) if loc else (17.3850, 78.4867)
except:
    lat, lon = 17.3850, 78.4867

temp, rain = get_weather(lat, lon)

# Store Vehicle Count
detected_count = 0

# --- INPUT SECTION ---
col_in1, col_in2 = st.columns([1, 1])

with col_in1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Vehicle Input Source")
    
    if input_type == "🎥 Live Camera (CV Detection)":
        st.write("Start camera to count vehicles in real-time using YOLOv8:")
        run_cam = st.checkbox("Turn On Camera")
        cam_placeholder = st.empty()
        
        if run_cam and yolo:
            cap = cv2.VideoCapture(0)
            target_classes = {"car", "motorcycle", "bus", "truck"}
            
            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera not found.")
                    break
                
                results = yolo.track(frame, persist=True, verbose=False)
                count = 0
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if yolo.names[cls_id] in target_classes:
                            count += 1
                
                detected_count = count
                annotated = results[0].plot()
                cv2.putText(annotated, f"Detected Vehicles: {count}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cam_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            st.info("Check 'Turn On Camera' to detect vehicles.")
    else:
        detected_count = st.slider("Enter Vehicle Count Manually:", 0, 100, 20)
        
    st.markdown('</div>', unsafe_allow_html=True)

with col_in2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. Environmental Data")
    st.write(f"**City:** {city_name} (Lat: {lat:.2f}, Lon: {lon:.2f})")
    st.write(f"**Temperature:** {temp} °C")
    st.write(f"**Live Rainfall:** {rain} mm")
    st.write(f"**Selected Time:** {target_time.strftime('%I:%M %p')}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION TRIGGER BUTTON ---
st.markdown("---")
st.subheader("3. AI Traffic Analysis & Signal Control")

# PREDICTION BUTTON
predict_btn = st.button("🚀 PREDICT TRAFFIC & OPTIMIZE SIGNALS")

# Initialize session state for button persistence
if "predicted" not in st.session_state:
    st.session_state.predicted = False

if predict_btn:
    st.session_state.predicted = True

# --- PREDICTION RESULTS DISPLAY ---
if st.session_state.predicted:
    # Run PyTorch Model Prediction
    raw_in = np.array([[detected_count, target_time.hour, rain, temp]])
    scaled_in = scaler_X.transform(raw_in)
    
    with torch.no_grad():
        out = dl_model(torch.tensor(scaled_in, dtype=torch.float32)).numpy()
        pred_volume = float(scaler_y.inverse_transform(out)[0][0])
        
    if accident:
        pred_volume += 1500

    congestion_lvl = min(int((pred_volume / 4000) * 100), 100)

    # Status Determination
    if congestion_lvl < 40:
        status = "LOW"
        color = "green"
        signal_action = "Standard Signal Timing (30 sec)"
    elif congestion_lvl < 75:
        status = "MODERATE"
        color = "orange"
        signal_action = "Medium Extension (+15 sec Green Light)"
    else:
        status = "HIGH"
        color = "red"
        signal_action = "Maximum Extension (+45 sec Green Light) & Rerouting"

    # Display Metrics
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Detected Live Vehicles", f"{detected_count} Units")
    res_col2.metric("Predicted Traffic Volume", f"{int(pred_volume)} vehicles/hr")
    res_col3.metric("Congestion Level", f"{congestion_lvl}% ({status})")
    res_col4.metric("Recommended Signal Action", signal_action)

    # Visual Layout (Map + Gauge)
    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.subheader("📍 Junction Heatmap")
        map_obj = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Circle(
            location=[lat, lon],
            radius=1200,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,
            popup=f"{city_name} Traffic: {status}"
        ).add_to(map_obj)
        st_folium(map_obj, width=500, height=300)

    with m_col2:
        st.subheader("📊 Congestion Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=congestion_lvl,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00c6ff"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(0, 255, 0, 0.3)"},
                    {'range': [40, 75], 'color': "rgba(255, 165, 0, 0.3)"},
                    {'range': [75, 100], 'color': "rgba(255, 0, 0, 0.3)"}
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Final Decision Output Box
    st.markdown("### 🚦 Autonomous Decision Output")
    if is_emergency:
        st.error("🚨 EMERGENCY VEHICLE DETECTED: Activating Green Corridor! All other signals turned RED.")
    elif status == "HIGH":
        st.error(f"🚨 HIGH TRAFFIC AT {city_name.upper()}: Increasing Green Light Duration by 45 Seconds.")
    elif status == "MODERATE":
        st.warning(f"⚠️ MODERATE TRAFFIC AT {city_name.upper()}: Increasing Green Light Duration by 15 Seconds.")
    else:
        st.success(f"🟢 NORMAL TRAFFIC AT {city_name.upper()}: Regular Signal Cycle Active.")