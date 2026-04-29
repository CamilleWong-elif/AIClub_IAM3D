#!/usr/bin/env python3
"""
Interactive Throttle Tuner
==========================
W / S  — increase / decrease MAX_THROTTLE_PCT by 0.01
D / A  — increase / decrease RAMP_TIME_SEC by 0.25
Q      — quit
"""

import threading
import time
import sys
import tty
import termios
from adafruit_servokit import ServoKit

# ============================================================
# STARTING VALUES
# ============================================================

MAX_THROTTLE_PCT = 0.15   # start at 15%
RAMP_TIME_SEC    = 2.0    # start at 2 seconds

# ============================================================
# CONFIG
# ============================================================

ESC_LEFT_CH  = 14
ESC_RIGHT_CH = 15
ESC_NEUTRAL  = 90
ESC_MIN      = 30
ESC_MAX      = 150

kit = ServoKit(channels=16)

# ============================================================
# HELPERS
# ============================================================

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def neutral_escs():
    kit.servo[ESC_LEFT_CH].angle = ESC_NEUTRAL
    kit.servo[ESC_RIGHT_CH].angle = ESC_NEUTRAL

def set_esc(ch, value):
    kit.servo[ch].angle = int(clamp(value, ESC_MIN, ESC_MAX))

def get_key():
    """Read a single keypress without enter"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def print_status(current_throttle):
    max_power = MAX_THROTTLE_PCT * 100
    ramp = max_power / (RAMP_TIME_SEC * 50)
    print(
        f"\r[Max:{MAX_THROTTLE_PCT*100:.0f}% | "
        f"Ramp:{RAMP_TIME_SEC:.2f}s | "
        f"Throttle:{current_throttle:+.1f}/{max_power:.0f} | "
        f"rate:{ramp:.3f}/loop]  ",
        end="", flush=True
    )

# ============================================================
# MOTOR LOOP (runs in background thread)
# ============================================================

running          = True
current_throttle = 0.0
target_throttle  = 0.0   # 0 = stopped, positive = forward

def motor_loop():
    global current_throttle
    while running:
        max_power = MAX_THROTTLE_PCT * 100
        ramp      = max_power / (RAMP_TIME_SEC * 50)

        diff = target_throttle - current_throttle
        if abs(diff) < ramp:
            current_throttle = target_throttle
        elif diff > 0:
            current_throttle += ramp
        else:
            current_throttle -= ramp

        if abs(current_throttle) < 0.5:
            neutral_escs()
            current_throttle = 0.0
        else:
            esc_val = map_range(current_throttle, -max_power, max_power, ESC_MIN, ESC_MAX)
            set_esc(ESC_LEFT_CH,  esc_val)
            set_esc(ESC_RIGHT_CH, esc_val)

        print_status(current_throttle)
        time.sleep(0.02)  # ~50hz

# ============================================================
# MAIN
# ============================================================

def main():
    global MAX_THROTTLE_PCT, RAMP_TIME_SEC, target_throttle, running

    print("Arming ESCs...")
    neutral_escs()
    time.sleep(2)

    print("\nInteractive Throttle Tuner")
    print("  W / S  — max throttle  +0.01 / -0.01")
    print("  D / A  — ramp time     +0.25 / -0.25 sec")
    print("  SPACE  — toggle forward throttle on/off")
    print("  Q      — quit\n")

    t = threading.Thread(target=motor_loop, daemon=True)
    t.start()

    throttle_on = False

    try:
        while True:
            key = get_key().lower()

            if key == 'w':
                MAX_THROTTLE_PCT = round(min(1.0, MAX_THROTTLE_PCT + 0.01), 2)
                print(f"\n↑ Max throttle: {MAX_THROTTLE_PCT*100:.0f}%")

            elif key == 's':
                MAX_THROTTLE_PCT = round(max(0.01, MAX_THROTTLE_PCT - 0.01), 2)
                print(f"\n↓ Max throttle: {MAX_THROTTLE_PCT*100:.0f}%")

            elif key == 'd':
                RAMP_TIME_SEC = round(min(10.0, RAMP_TIME_SEC + 0.25), 2)
                print(f"\n→ Ramp time: {RAMP_TIME_SEC:.2f}s")

            elif key == 'a':
                RAMP_TIME_SEC = round(max(0.25, RAMP_TIME_SEC - 0.25), 2)
                print(f"\n← Ramp time: {RAMP_TIME_SEC:.2f}s")

            elif key == ' ':
                throttle_on = not throttle_on
                target_throttle = MAX_THROTTLE_PCT * 100 if throttle_on else 0.0
                print(f"\n{'▶ THROTTLE ON' if throttle_on else '■ THROTTLE OFF'}")

            elif key == 'q':
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        running = False
        neutral_escs()
        print(f"\nFinal values:")
        print(f"  MAX_THROTTLE_PCT = {MAX_THROTTLE_PCT}")
        print(f"  RAMP_TIME_SEC    = {RAMP_TIME_SEC}")
        print("Copy these into your main robot code!")

if __name__ == "__main__":
    main()