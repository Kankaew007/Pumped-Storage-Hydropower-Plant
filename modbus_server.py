from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore.store import ModbusSequentialDataBlock

# ------------------ ตั้งค่าพอร์ต (เปลี่ยนได้ตามต้องการ) ------------------ #
PORT = 1520  # ใช้พอร์ตอื่นแทน 1502 เพื่อหลีกเลี่ยงปัญหา bind

# ------------------ กำหนดค่าตั้งต้นของ Register ------------------ #
# Register 0: MODE (0 = Idle, 1 = Pumping, 2 = Generating)
# Register 1: POWER (signed 16-bit int)

store = ModbusSlaveContext(
    hr=ModbusSequentialDataBlock(0, [0, 0]),  # Holding Registers เริ่มที่ index 0
    zero_mode=True
)

context = ModbusServerContext(slaves=store, single=True)

# ------------------ เริ่มต้น TCP Server ------------------ #
if __name__ == "__main__":
    print(f"🟢 Modbus TCP Server กำลังรันอยู่ที่พอร์ต {PORT}...")
    StartTcpServer(context, address=("0.0.0.0", PORT))