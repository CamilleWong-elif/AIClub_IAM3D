import time
import curses
from adafruit_servokit import ServoKit

# SETUP

kit = ServoKit(channels=16)

servos = {
    
    "arm_left":    {"ch": 0, "angle": 0.0, "min": 0, "max": 170, "speed": 180, "invert": False},
    "arm_right":   {"ch": 1, "angle": 0.0, "min": 0, "max": 170, "speed": 180, "invert": False},

    "bucket":      {"ch": 2, "angle": 0.0, "min": 0, "max": 160, "speed": 180, "invert": False},

    "wrist_left":  {"ch": 3, "angle": 0.0, "min": 0, "max": 150, "speed": 120, "invert": False},
    "wrist_right": {"ch": 4, "angle": 0.0, "min": 0, "max": 150, "speed": 120, "invert": False},

    "camera_1":    {"ch": 5, "angle": 0.0, "min": 0, "max": 180, "speed": 120, "invert": False},
    "camera_2":    {"ch": 6, "angle": 0.0, "min": 0, "max": 180, "speed": 120, "invert": False},
}

def clamp(name, angle):
    return max(servos[name]["min"], min(servos[name]["max"], angle))


def move_servo(name, direction, dt):
    s = servos[name]

    if s["invert"]:
        direction = -direction

    s["angle"] += direction * s["speed"] * dt
    s["angle"] = clamp(name, s["angle"])

    kit.servo[s["ch"]].angle = int(s["angle"])


def move_group(names, direction, dt):
    for n in names:
        move_servo(n, direction, dt)


def release_all():
    for name in servos:
        kit.servo[servos[name]["ch"]].angle = None


def initialize_servos():
    for name in servos:
        kit.servo[servos[name]["ch"]].angle = int(servos[name]["angle"])


def display_servo_map(stdscr, start_row=0):
    stdscr.addstr(start_row, 0, "==== IAM3D SERVO CHANNEL MAP ====")
    row = start_row + 2

    sorted_servos = sorted(servos.items(), key=lambda x: x[1]["ch"])

    for name, data in sorted_servos:
        line = (
            f"Channel {data['ch']:2d}  |  "
            f"{name:12s}  |  "
            f"Angle: {data['angle']:6.1f}  |  "
            f"Min: {data['min']:3d}  Max: {data['max']:3d}"
        )
        stdscr.addstr(row, 0, line)
        row += 1

    return row + 1


# MAIN CONTROL LOOP (MULTI-INPUT)


HELP_TEXT = """
Controls:
Arms:        I / K
Bucket:      J / L
Wrist:       U / O
Camera 1:    1 / 2
Camera 2:    3 / 4

R = release all
Q = quit
"""


def main(stdscr):
    curses.cbreak()
    curses.noecho()
    stdscr.nodelay(True)
    stdscr.keypad(True)

    initialize_servos()

    stdscr.clear()
    next_row = display_servo_map(stdscr, start_row=0)
    stdscr.addstr(next_row, 0, HELP_TEXT)
    stdscr.refresh()

    quit_program = False
    last_time = time.time()

    # Multi-input tracking: key -> expire_time
    active_keys = {}
    HOLD_TIMEOUT = 0.20  # seconds (increase if keys "drop")

    while not quit_program:
        now = time.time()
        dt = now - last_time
        last_time = now

        ch = stdscr.getch()

        if ch != -1:
            try:
                k = chr(ch).lower()
            except ValueError:
                k = None

            if k == "q":
                quit_program = True
            elif k == "r":
                release_all()
            elif k is not None:
                # key-repeat refreshes this; expiry simulates key release
                active_keys[k] = now + HOLD_TIMEOUT

        # Expire keys that are no longer repeating
        for k in list(active_keys.keys()):
            if now > active_keys[k]:
                del active_keys[k]

           # Apply ALL active keys
       
        # Arms (synced)
        if "i" in active_keys:
            move_group(["arm_left", "arm_right"], +1, dt)
        if "k" in active_keys:
            move_group(["arm_left", "arm_right"], -1, dt)

        # Bucket
        if "j" in active_keys:
            move_servo("bucket", -1, dt)
        if "l" in active_keys:
            move_servo("bucket", +1, dt)

        # Wrist (synced)
        if "u" in active_keys:
            move_group(["wrist_left", "wrist_right"], +1, dt)
        if "o" in active_keys:
            move_group(["wrist_left", "wrist_right"], -1, dt)

        # Camera 1
        if "1" in active_keys:
            move_servo("camera_1", -1, dt)
        if "2" in active_keys:
            move_servo("camera_1", +1, dt)

        # Camera 2
        if "3" in active_keys:
            move_servo("camera_2", -1, dt)
        if "4" in active_keys:
            move_servo("camera_2", +1, dt)

        time.sleep(0.01)

    release_all()

# PROGRAM ENTRY

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    finally:
        print("Program ended.")


