"""
Simple live camera feed + recording for Raspberry Pi 5 + Camera Module 3.

Terminal controls:
    1 = Start recording
    2 = Stop recording
    3 = Save (shows file path + size)
    q = Quit

Usage:
    python camera.py
    python camera.py --resolution 1280 720 --fps 30
"""

import os
import sys
import cv2
import time
import argparse
import threading
from datetime import datetime


# ---------------------------------------------------------------------------
# Non-blocking keyboard input (works over SSH)
# ---------------------------------------------------------------------------
class KeyboardListener:
    def __init__(self):
        self.last_key = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self._running:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self.last_key = ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            while self._running:
                try:
                    line = input()
                    if line:
                        with self._lock:
                            self.last_key = line[0]
                except EOFError:
                    break

    def get_key(self):
        with self._lock:
            k = self.last_key
            self.last_key = None
            return k

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pi Camera live feed + recording")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        help="Width Height (default: 640 480)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Framerate (default: 30)")
    parser.add_argument("--save-dir", type=str, default="recordings",
                        help="Directory for saved recordings (default: recordings/)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (no preview window)")
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = args.resolution

    # ---- Init camera ----
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("ERROR: picamera2 not installed.")
        print("  Install with: sudo apt install python3-picamera2")
        sys.exit(1)

    print(f"Starting camera: {width}x{height} @ {args.fps}fps")
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (width, height)},
    )
    picam2.configure(video_config)
    picam2.start()
    time.sleep(1)  # warm up + autofocus

    # ---- State ----
    os.makedirs(args.save_dir, exist_ok=True)
    recording = False
    video_writer = None
    current_path = None
    last_saved_path = None

    kb = KeyboardListener()

    print("\n--- Camera ready ---")
    print("1 = Record | 2 = Stop | 3 = Save | q = Quit\n")

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    try:
        while True:
            # ---- Capture ----
            frame_rgb = picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # ---- FPS ----
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # ---- HUD ----
            status = f"FPS: {fps:.1f}"
            if recording:
                status += "  [REC]"
                cv2.circle(frame_bgr, (width - 30, 25), 8, (0, 0, 255), -1)
            cv2.putText(frame_bgr, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_bgr, "1=Rec 2=Stop 3=Save q=Quit", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ---- Record ----
            if recording and video_writer is not None:
                video_writer.write(frame_bgr)

            # ---- Display ----
            if not args.no_display:
                cv2.imshow("Camera", frame_bgr)
                cv2.waitKey(1)

            # ---- Keyboard ----
            key = kb.get_key()

            if key == "1" and not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_path = os.path.join(args.save_dir, f"rec_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(current_path, fourcc, args.fps, (width, height))
                recording = True
                print(f"[REC] Started: {current_path}")

            elif key == "2" and recording:
                recording = False
                video_writer.release()
                video_writer = None
                last_saved_path = current_path
                print(f"[STOP] Stopped: {last_saved_path}")

            elif key == "3":
                if last_saved_path and os.path.exists(last_saved_path):
                    size_mb = os.path.getsize(last_saved_path) / (1024 * 1024)
                    print(f"[SAVED] {last_saved_path} ({size_mb:.1f} MB)")
                elif recording:
                    print("[INFO] Stop recording first (press 2)")
                else:
                    print("[INFO] No recording yet. Press 1 to start.")

            elif key == "q":
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        kb.stop()
        if recording and video_writer is not None:
            video_writer.release()
            print(f"[SAVED] Auto-saved: {current_path}")
        picam2.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
