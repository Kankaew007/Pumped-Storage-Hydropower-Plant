
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# ------------------ เตรียมข้อมูล ------------------ #
df = pd.read_csv("load_demand_sample.csv")
df['Hour'] = df['Time'].str.slice(0, 2).astype(int)

df['Weekday'] = [1]*24
df['Period'] = pd.cut(df['Hour'], bins=[-1, 5, 11, 17, 23],
                      labels=["night", "morning", "afternoon", "evening"])

le = LabelEncoder()
df['Period_encoded'] = le.fit_transform(df['Period'])

X = df[['Hour', 'Weekday', 'Period_encoded']]
y = df['Load_MW']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted_Load'] = model.predict(X)

# ------------------ Streamlit Dashboard ------------------ #
st.title("Intelligent Control System for Pumped-Storage Hydroelectric Power Plant")

# ตั้งค่าขนาดอ่าง
st.sidebar.header("📏 กำหนดขนาดอ่างน้ำ")
upper_width = st.sidebar.number_input("ความกว้างอ่างบน (m)", value=100)
upper_length = st.sidebar.number_input("ความยาวอ่างบน (m)", value=100)
upper_depth = st.sidebar.number_input("ความลึกอ่างบน (m)", value=30)

lower_width = st.sidebar.number_input("ความกว้างอ่างล่าง (m)", value=80)
lower_length = st.sidebar.number_input("ความยาวอ่างล่าง (m)", value=80)
lower_depth = st.sidebar.number_input("ความลึกอ่างล่าง (m)", value=25)

height_between = st.sidebar.number_input("ความสูงระหว่างอ่าง (m)", value=100)

upper_volume = upper_width * upper_length * upper_depth  # m³
lower_volume = lower_width * lower_length * lower_depth  # m³

# ------------------ จำลองระบบโรงไฟฟ้า ------------------ #
class PumpedStoragePlant:
    def __init__(self, upper_size, lower_size, height):
        self.upper = 70  # % เริ่มต้น
        self.lower = 30  # % เริ่มต้น
        self.power = 0
        self.mode = "พักระบบ"
        self.history = []
        self.upper_size = upper_size
        self.lower_size = lower_size
        self.height = height
        self.energy_generated = 0  # kWh ที่ผลิตสะสม

    def calc_potential_energy(self):
        water_volume = self.upper / 100 * self.upper_size  # m³
        mass = water_volume * 1000  # kg
        g = 9.81  # m/s²
        potential_energy_joule = mass * g * self.height
        return potential_energy_joule / 3.6e6  # J → kWh

    def pump(self):
        if self.lower > 5 and self.upper < 100:
            self.lower -= 2
            self.upper += 2
            self.power = -10
            self.mode = "สูบน้ำ"
        else:
            self.idle()

    def generate(self):
        if self.upper > 5 and self.lower < 100:
            self.lower += 3
            self.upper -= 3
            self.power = 15
            self.mode = "ผลิตไฟฟ้า"
            self.energy_generated += 15/1  # กำลัง (kW) × ชั่วโมง (1 ชม.) = kWh
        else:
            self.idle()

    def idle(self):
        self.power = 0
        self.mode = "พักระบบ"

    def run(self, load):
        if load > 120:
            self.generate()
        elif load < 100:
            self.pump()
        else:
            self.idle()
        self.history.append((self.upper, self.lower, self.power, self.mode, self.calc_potential_energy(), self.energy_generated))

# ------------------ เริ่มจำลอง ------------------ #
mode = st.radio("โหมดการทำงาน", ["เลือกเวลาเอง", "จำลองทั้งวัน", "จำลอง Real-Time"])

plant = PumpedStoragePlant(upper_volume, lower_volume, height_between)

if mode == "เลือกเวลาเอง":
    hour = st.slider("เลือกเวลา (ชั่วโมง)", 0, 23, 12)
    row = df[df['Hour'] == hour].iloc[0]
    load = row['Predicted_Load']
    plant.run(load)
    status = plant.history[-1]

    st.metric("เวลา", f"{hour}:00")
    st.metric("โหลดที่คาดการณ์", f"{load:.2f} MW")
    st.metric("สถานะระบบ", status[3])
    st.metric("ระดับน้ำอ่างบน", f"{status[0]:.2f}%")
    st.metric("ระดับน้ำอ่างล่าง", f"{status[1]:.2f}%")
    st.metric("พลังงานที่ผลิตสะสม", f"{status[5]:.2f} kWh")

elif mode == "จำลองทั้งวัน":
    for i in range(24):
        load = df['Predicted_Load'][i]
        plant.run(load)

    uppers = [h[0] for h in plant.history]
    lowers = [h[1] for h in plant.history]
    powers = [h[2] for h in plant.history]
    modes = [h[3] for h in plant.history]
    energies = [h[5] for h in plant.history]

    st.subheader("📊 กราฟระดับน้ำและพลังงานต่อชั่วโมง")
    fig, ax = plt.subplots()
    ax.plot(df['Hour'], uppers, label='Upper reservoir (%)')
    ax.plot(df['Hour'], lowers, label='Lower reservoir (%)')
    ax.plot(df['Hour'], powers, label='Energy (MW)')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📋 ตารางสถานะรายชั่วโมง")
    summary = pd.DataFrame({
        "ชั่วโมง": df['Hour'],
        "โหลดคาดการณ์": df['Predicted_Load'].round(2),
        "สถานะ": modes,
        "อ่างบน": uppers,
        "อ่างล่าง": lowers,
        "พลังงานที่ผลิตสะสม (kWh)": energies
    })
    st.dataframe(summary)

else:  # Real-Time mode
    st_autorefresh(interval=5000, key="simulate_hour")

    if "simulated_hour" not in st.session_state:
        st.session_state.simulated_hour = 0
        st.session_state.uplog = []
        st.session_state.lowlog = []
        st.session_state.energylog = []

    st.session_state.simulated_hour = (st.session_state.simulated_hour + 1) % 24

    hour = st.session_state.simulated_hour
    row = df[df['Hour'] == hour].iloc[0]
    load = row['Predicted_Load']
    plant.run(load)
    status = plant.history[-1]

    st.session_state.uplog.append(status[0])
    st.session_state.lowlog.append(status[1])
    st.session_state.energylog.append(status[5])

    st.metric("⏰ เวลา (Real-Time)", f"{hour}:00")
    st.metric("โหลดที่คาดการณ์", f"{load:.2f} MW")
    st.metric("สถานะระบบ", status[3])
    st.metric("ระดับน้ำอ่างบน", f"{status[0]:.2f}%")
    st.metric("ระดับน้ำอ่างล่าง", f"{status[1]:.2f}%")
    st.metric("พลังงานที่ผลิตสะสม", f"{status[5]:.2f} kWh")

    st.subheader("📈 กราฟระดับน้ำและพลังงานสะสม (Real-Time)")

    fig, ax1 = plt.subplots()
    ax1.plot(range(len(st.session_state.uplog)), st.session_state.uplog, label="Upper reservoir (%)", color="blue")
    ax1.plot(range(len(st.session_state.lowlog)), st.session_state.lowlog, label="Lower reservoir (%)", color="green")
    ax1.set_ylabel("Water level (%)", color="blue")
    ax1.set_xlabel("Update interval: 5 seconds per cycle)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(range(len(st.session_state.energylog)), st.session_state.energylog, label="Stored energy (kWh)", color="red")
    ax2.set_ylabel("Stored energy (kWh)", color="red")

    st.pyplot(fig)
