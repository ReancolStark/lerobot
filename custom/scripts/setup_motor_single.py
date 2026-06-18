
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig

motor = "shoulder_lift"  # 改成要设置的: shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper
leader = SO100Leader(SO100LeaderConfig(port="COM6"))

try:
    leader.bus.setup_motor(motor)
    print(f"'{motor}' motor id set to {leader.bus.motors[motor].id}")
finally:
    if leader.bus.is_connected:
        leader.bus.disconnect(disable_torque=False)