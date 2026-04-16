#!/usr/bin/env python3
"""
---turn it on startup---
sudo systemctl start robot

__turn on the cameras__
python3 pi_stream --camera 0 --port 8000 &

--open the video camera streams--
http://10.42.0.1:8000/

"""

import serial
import struct
import time
from adafruit_servokit import ServoKit

# ============================================================
# CONFIG
# ============================================================

PORT = "/dev/ttyAMA0"
BAUD = 115200

ESC_LEFT_CH  = 14
ESC_RIGHT_CH = 15
ESC_NEUTRAL  = 90
ESC_MIN      = 30
ESC_MAX      = 150

# add to CONFIG section
SERVO_UPDATE_INTERVAL = 0.02   # only update servos every 20ms
ESC_UPDATE_INTERVAL   = 0.01   # only update ESCs every 10ms
THROTTLE_RAMP         = 1.0    # max throttle change per update, lower = slower
SPEED_LIMIT = 0.3

DEADBAND_LOW  = 1350
DEADBAND_HIGH = 1650
ARM_SPEED = 2
WRIST_SPEED = 2

STARTUP_DELAY    = 1.0
HOMING_SPEED     = 30
HOMING_STEP_TIME = 0.02

servos = {
    "arm_left":    {"ch": 6, "angle": 90.0, "min": 0, "max": 180, "invert": False, "home": 90},
    "arm_right":   {"ch": 7, "angle": 90.0, "min": 0, "max": 180, "invert": True,  "home": 90},
    "wrist_left":  {"ch": 1, "angle": 90.0, "min": 0, "max": 180, "invert": False, "home": 90},
    "wrist_right": {"ch": 5, "angle": 90.0, "min": 0, "max": 180, "invert": True,  "home": 90},
    "bucket":      {"ch": 0, "angle": 90.0, "min": 0, "max": 180, "invert": False, "home": 90},
    "camera_1":    {"ch": 8, "angle": 90.0, "min": 0, "max": 180, "invert": False, "home": 90},
}

# ============================================================
# INIT
# ============================================================

try:
    kit = ServoKit(channels=16)
    print("Servo HAT initialized!")
except Exception as e:
    print(f"HAT error: {e}")
    exit()

# ============================================================
# HELPERS
# ============================================================

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def apply_deadband(val, low, high):
    """Returns 0 if within deadband, otherwise maps to -100/+100."""
    if low <= val <= high:
        return 0.0
    elif val > high:
        return map_range(val, high, 2000, 0, 100)
    else:
        return map_range(val, 1000, low, -100, 0)

def set_servo(name, angle):
    s = servos[name]
    s["angle"] = clamp(angle, s["min"], s["max"])
    out = s["angle"]
    if s["invert"]:
        out = s["max"] - (out - s["min"])
    kit.servo[s["ch"]].angle = int(out)

def set_esc(ch, value):
    kit.servo[ch].angle = int(clamp(value, ESC_MIN, ESC_MAX))

def neutral_escs():
    kit.servo[ESC_LEFT_CH].angle  = ESC_NEUTRAL
    kit.servo[ESC_RIGHT_CH].angle = ESC_NEUTRAL

def release_all():
    # send neutral to ESCs first and wait before releasing
    kit.servo[ESC_LEFT_CH].angle  = ESC_NEUTRAL
    kit.servo[ESC_RIGHT_CH].angle = ESC_NEUTRAL
    time.sleep(0.5)  # hold neutral so ESC registers it
    kit.servo[ESC_LEFT_CH].angle  = ESC_NEUTRAL
    kit.servo[ESC_RIGHT_CH].angle = ESC_NEUTRAL
    for s in servos.values():
        kit.servo[s["ch"]].angle = ESC_NEUTRAL

def fast_startup():
    print("Fast startup: initializing ESCs + servos")

    # Immediately send neutral to ESCs
    kit.servo[ESC_LEFT_CH].angle  = ESC_NEUTRAL
    kit.servo[ESC_RIGHT_CH].angle = ESC_NEUTRAL

    # Set all servos instantly to home (NO slow sweep)
    for name, s in servos.items():
        s["angle"] = float(s["home"])
        out = s["angle"]
        if s["invert"]:
            out = s["max"] - (out - s["min"])
        kit.servo[s["ch"]].angle = int(out)

    # Short, consistent arming pulse (ESCs typically need ~0.2–0.5s)
    time.sleep(0.3)

    print("Startup complete.\n")

def startup_home():
    print(f"Waiting {STARTUP_DELAY}s before homing...")
    release_all()
    kit.servo[ESC_LEFT_CH].angle  = ESC_NEUTRAL; kit.servo[ESC_RIGHT_CH].angle = ESC_NEUTRAL
    #time.sleep(STARTUP_DELAY)

    print("Homing servos...")
    step = HOMING_SPEED * HOMING_STEP_TIME
    still_moving = True
    while still_moving:
        still_moving = False
        for name, s in servos.items():
            target = float(s["home"])
            if abs(s["angle"] - target) < step:
                s["angle"] = target
            else:
                still_moving = True
                s["angle"] += step if s["angle"] < target else -step
            out = s["angle"]
            if s["invert"]:
                out = s["max"] - (out - s["min"])
            kit.servo[s["ch"]].angle = int(clamp(out, s["min"], s["max"]))
        time.sleep(HOMING_STEP_TIME)
    print("Homing complete.\n")

# ============================================================
# MAIN
# ============================================================

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
    buf = bytearray()
    last_print   = 0
    valid_signal = False
    _kill_until = 0
    last_servo_update = 0
    last_esc_update   = 0
    current_throttle  = 0.0
    target_throttle = 0.0


    fast_startup()

    print("RC Controller running...")
    print("SWA (CH7): 1000=OFF  2000=ON | Ctrl+C to stop\n")

    try:
        while True:
            data = ser.read(32)
            if not data:
                continue
            buf.extend(data)

            while len(buf) >= 32:
                # Find iBus header
                idx = buf.find(bytes([0x20, 0x40]))
                if idx < 0:
                    buf = buf[-1:]
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < 32:
                    break

                frame = buf[:32]

                # Validate checksum
                chk = 0xFFFF
                for b in frame[:30]:
                    chk -= b
                if chk != (frame[30] | frame[31] << 8):
                    buf = buf[1:]
                    continue

                channels = [struct.unpack_from("<H", frame, 2 + i*2)[0] for i in range(14)]
                buf = buf[32:]

                if not all(900 <= c <= 2100 for c in channels[:10]):
                    continue

                if not valid_signal:
                    print("RC signal acquired!")
                    valid_signal = True

                ch1_steer    = channels[0]   # left/right
                ch2_throttle = channels[1]   # forward/reverse
                ch3_bucket   = channels[2]   # bucket
                ch5_camera   = channels[4]   # camera knob
                ch6_wrist    = channels[5]   #wrist movement
                ch7_kill     = channels[6]   # SWA kill switch
                ch8_reverse  = channels[7]   #when on reverse
                ch9_arms     = channels[8]   #arms
                ch10_bucket = channels[9]  #brake for the wheels

                # Kill switch: 2000 = ON, 1000 = OFF
                is_enabled = valid_signal and ch7_kill > 1500

                if not is_enabled:
                    neutral_escs()
                    # stop all servos too
                    for s in servos.values():
                        kit.servo[s["ch"]].angle = int(s["home"])
                    current_throttle = 0.0
                    target_throttle  = 0.0
                    status = "DISABLED"
                else:
                    status = "ACTIVE  "
                    current_time = time.time()

                    if current_time < _kill_until:
                        neutral_escs()
                        target_throttle = 0.0
                        current_throttle = 0.0
                        status = "CH10 KILL"
                    else:
                        # throttle ramp
                        raw_throttle = apply_deadband(ch2_throttle, DEADBAND_LOW, DEADBAND_HIGH) * SPEED_LIMIT

                        if ch8_reverse > 1500:
                            target_throttle = raw_throttle  # flip direction
                        else:
                            target_throttle = -raw_throttle
                        
                        steering = apply_deadband(ch1_steer, DEADBAND_LOW, DEADBAND_HIGH) * SPEED_LIMIT

                        if abs(target_throttle - current_throttle) < THROTTLE_RAMP:
                            current_throttle = target_throttle
                        elif target_throttle > current_throttle:
                            current_throttle += THROTTLE_RAMP
                        else:
                            current_throttle -= THROTTLE_RAMP

                        if current_time - last_esc_update > ESC_UPDATE_INTERVAL:
                            last_esc_update = current_time
                            left_power  = clamp(current_throttle + steering, -100 * SPEED_LIMIT, 100 * SPEED_LIMIT)
                            right_power = clamp(current_throttle - steering, -100 * SPEED_LIMIT, 100 * SPEED_LIMIT)

                            if abs(current_throttle) < 1.0 and abs(steering) < 1.0:
                                set_esc(ESC_LEFT_CH,  ESC_NEUTRAL)
                                set_esc(ESC_RIGHT_CH, ESC_NEUTRAL)
                                current_throttle = 0.0
                            else:
                                set_esc(ESC_LEFT_CH,  map_range(left_power,  -100, 100, ESC_MIN, ESC_MAX))
                                set_esc(ESC_RIGHT_CH, map_range(right_power, -100, 100, ESC_MIN, ESC_MAX))
                    # servos always run when active
                    if current_time - last_servo_update > SERVO_UPDATE_INTERVAL:
                        last_servo_update = current_time

                        cam_angle = map_range(ch5_camera, 1000, 2000, 0, 180)
                        set_servo("camera_1", cam_angle)

                        if ch3_bucket < 1400:
                            set_servo("bucket", servos["bucket"]["angle"] - ARM_SPEED)
                            print(f"CH3 bucket:{ch3_bucket} angle:{servos['bucket']['angle']:.1f}")
                        elif ch3_bucket > 1600:
                            set_servo("bucket", servos["bucket"]["angle"] + ARM_SPEED)
                            print(f"CH3 bucket:{ch3_bucket} angle:{servos['bucket']['angle']:.1f}")

                        if ch9_arms < 1400:
                            set_servo("arm_left",  servos["arm_left"]["angle"]  - ARM_SPEED)
                            set_servo("arm_right", servos["arm_right"]["angle"] - ARM_SPEED)
                        elif ch9_arms > 1600:
                            set_servo("arm_left",  servos["arm_left"]["angle"]  + ARM_SPEED)
                            set_servo("arm_right", servos["arm_right"]["angle"] + ARM_SPEED)

                        if ch6_wrist < 1400:
                            set_servo("wrist_left",  servos["wrist_left"]["angle"]  - WRIST_SPEED)
                            set_servo("wrist_right", servos["wrist_right"]["angle"] - WRIST_SPEED)
                        elif ch6_wrist > 1600:
                            set_servo("wrist_left",  servos["wrist_left"]["angle"]  + WRIST_SPEED)
                            set_servo("wrist_right", servos["wrist_right"]["angle"] + WRIST_SPEED)

                now = time.time()
                if now - last_print > 0.2:
                    print(
                        f"[{status}] "
                        f"Thr:{ch2_throttle} Str:{ch1_steer} | "
                        f"Cam:{ch5_camera} Kill:{ch7_kill}"
                    )
                    last_print = now

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        release_all()
        ser.close()
        print("Done.")

if __name__ == "__main__":
    main()  