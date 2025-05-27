import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit_autorefresh import st_autorefresh
from pymodbus.client.sync import ModbusTcpClient
import io

client = ModbusTcpClient('127.0.0.1', port=1502)

# ------------------ Load and Predict Load ------------------ #
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

# ------------------ Streamlit Layout ------------------ #
st.set_page_config(layout="wide")
st.title("🔋 Pumped Storage Hydropower Simulation with AI + Modbus TCP")

st.sidebar.header("🧲 Reservoir Parameters")
uw = st.sidebar.number_input("Upper Width (m)", 100)
ul = st.sidebar.number_input("Upper Length (m)", 100)
ud = st.sidebar.number_input("Upper Depth (m)", 30)
lw = st.sidebar.number_input("Lower Width (m)", 100)
ll = st.sidebar.number_input("Lower Length (m)", 100)
ld = st.sidebar.number_input("Lower Depth (m)", 30)
height_between = st.sidebar.number_input("Height Between Reservoirs (m)", 100)

upper_volume = uw * ul * ud
lower_volume = lw * ll * ld

# ------------------ Simulation Class ------------------ #
class PumpedStoragePlant:
    def __init__(self, upper_size, lower_size, height):
        self.upper = 70
        self.lower = 30
        self.power = 0
        self.mode = "Idle"
        self.history = []
        self.upper_size = upper_size
        self.lower_size = lower_size
        self.height = height
        self.energy_generated = 0

    def water_volume_per_percent(self, is_upper=True):
        return (self.upper_size if is_upper else self.lower_size) / 100

    def calc_energy_kwh(self, volume_m3):
        return (volume_m3 * 1000 * 9.81 * self.height) / 3.6e6

    def send_to_plc(self, mode, power):
        mode_value = {"Idle": 0, "Pumping": 1, "Generating": 2}.get(mode, 0)
        try:
            client = ModbusTcpClient('127.0.0.1', port=1502)
            if client.connect():
                client.write_register(0, mode_value, unit=1)
                client.write_register(1, int(power), unit=1)
                read = client.read_holding_registers(0, 2, unit=1)
                if not read.isError():
                    st.success(f"📱 Sent to PLC: MODE={mode}({mode_value}), POWER={power} MW")
                    st.info(f"📅 Confirmed Read: MODE={read.registers[0]}, POWER={read.registers[1]}")
                else:
                    st.warning("⚠️ Could not confirm values from PLC.")
                client.close()
            else:
                st.error("❌ Failed to connect to Modbus server.")
        except Exception as e:
            st.error(f"❌ Modbus Error: {e}")

    def pump(self):
        if self.lower > 5 and self.upper < 100:
            vol = self.water_volume_per_percent(False) * 2
            self.upper += (vol / self.upper_size) * 100
            self.lower -= (vol / self.lower_size) * 100
            self.power = -10
            self.mode = "Pumping"
            self.send_to_plc(self.mode, self.power)
        else:
            self.idle()

    def generate(self):
        if self.upper > 5 and self.lower < 100:
            vol = self.water_volume_per_percent(True) * 3
            self.upper -= (vol / self.upper_size) * 100
            self.lower += (vol / self.lower_size) * 100
            self.power = 15
            self.mode = "Generating"
            self.energy_generated += self.calc_energy_kwh(vol)
            self.send_to_plc(self.mode, self.power)
        else:
            self.idle()

    def idle(self):
        self.power = 0
        self.mode = "Idle"
        self.send_to_plc(self.mode, self.power)

    def run(self, load):
        if load > 120:
            self.generate()
        elif load < 100:
            self.pump()
        else:
            self.idle()
        self.history.append((self.upper, self.lower, self.power, self.mode, self.energy_generated))

# ------------------ Main Control ------------------ #
mode = st.radio("🔘 Select Simulation Mode", ["Manual Mode", "Simulate Full Day", "Real-Time Mode"])
plant = PumpedStoragePlant(upper_volume, lower_volume, height_between)

# ----------- Manual Mode ----------- #
if mode == "Manual Mode":
    hour = st.slider("Select Hour", 0, 23, 12)
    load = df['Predicted_Load'][hour]
    plant.run(load)
    u, l, p, m, e = plant.history[-1]

    st.metric("Hour", f"{hour}:00")
    st.metric("Mode", m)
    st.metric("Upper Reservoir", f"{u:.2f}%")
    st.metric("Lower Reservoir", f"{l:.2f}%")
    st.metric("Energy Stored", f"{e:.2f} kWh")

# ----------- Simulate Full Day ----------- #
elif mode == "Simulate Full Day":
    for i in range(24):
        plant.run(df['Predicted_Load'][i])

    uppers = [h[0] for h in plant.history]
    lowers = [h[1] for h in plant.history]
    powers = [h[2] for h in plant.history]
    modes = [h[3] for h in plant.history]
    energies = [h[4] for h in plant.history]

    df_result = pd.DataFrame({
        "Hour": df['Hour'],
        "Predicted Load": df['Predicted_Load'].round(2),
        "Mode": modes,
        "Upper Level (%)": uppers,
        "Lower Level (%)": lowers,
        "Power (MW)": powers,
        "Energy Stored (kWh)": energies
    })

    st.subheader("📈 Daily Performance Graph")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_result["Hour"], y=uppers, name="Upper Level", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df_result["Hour"], y=lowers, name="Lower Level", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df_result["Hour"], y=powers, name="Power", mode="lines+markers"))
    fig.update_layout(title="Simulation Result (24hr)", xaxis_title="Hour", yaxis_title="Value")
    st.plotly_chart(fig)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False, sheet_name="Simulation")
    st.download_button("📅 Download Excel", data=buffer.getvalue(),
                       file_name="simulation_full_day.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.dataframe(df_result)

# ----------- Real-Time Mode ----------- #
else:
    st_autorefresh(interval=5000, key="realtime_auto")
    if "sim_hour" not in st.session_state:
        st.session_state.sim_hour = 0
        st.session_state.log = []

    hour = st.session_state.sim_hour
    load = df['Predicted_Load'][hour]
    plant.run(load)
    st.session_state.log.append(plant.history[-1])
    st.session_state.sim_hour = (hour + 1) % 24

    logs = st.session_state.log
    uplog = [s[0] for s in logs]
    lowlog = [s[1] for s in logs]
    powerlog = [s[2] for s in logs]
    modelog = [s[3] for s in logs]
    energylog = [s[4] for s in logs]

    st.subheader("📱 Real-Time Monitoring")
    st.metric("Hour", f"{hour}:00")
    st.metric("Mode", modelog[-1])
    st.metric("Upper Reservoir", f"{uplog[-1]:.2f}%")
    st.metric("Lower Reservoir", f"{lowlog[-1]:.2f}%")
    st.metric("Energy Stored", f"{energylog[-1]:.2f} kWh")

    fig_rt = go.Figure()
    fig_rt.add_trace(go.Scatter(x=list(range(len(uplog))), y=uplog, mode="lines", name="Upper (%)"))
    fig_rt.add_trace(go.Scatter(x=list(range(len(lowlog))), y=lowlog, mode="lines", name="Lower (%)"))
    fig_rt.add_trace(go.Scatter(x=list(range(len(powerlog))), y=powerlog, mode="lines", name="Power (MW)"))
    fig_rt.update_layout(title="Real-Time Timeline", xaxis_title="Step", yaxis_title="Value")
    st.plotly_chart(fig_rt)

    df_export = pd.DataFrame({
        "Step": list(range(len(logs))),
        "Upper (%)": uplog,
        "Lower (%)": lowlog,
        "Power (MW)": powerlog,
        "Mode": modelog,
        "Energy (kWh)": energylog
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name="RealTime")
    st.download_button("📤 Download Excel", data=buffer.getvalue(),
                       file_name="realtime_simulation.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
