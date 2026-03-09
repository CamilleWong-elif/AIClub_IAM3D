"""
MJPEG streaming server for Raspberry Pi 5 + Camera Module 3.
Streams video over HTTP so a local machine can receive and process frames.

Runs on the Pi. No OpenCV needed — only Picamera2 (pre-installed).

Endpoints:
    http://<pi-ip>:8000/             → HTML page with live preview
    http://<pi-ip>:8000/stream.mjpg  → Raw MJPEG stream (used by live_yolo.py)

Terminal controls:
    1 = Start recording locally on Pi
    2 = Stop recording
    3 = Save (shows file path + size)
    q = Quit

Usage:
    python stream.py
    python stream.py --resolution 640 480 --port 8000
    python stream.py --resolution 1280 720 --port 8000
"""

import io
import os
import sys
import time
import logging
import argparse
import socketserver
import threading
from http import server
from datetime import datetime

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Non-blocking keyboard input
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
# Streaming output buffer (shared between encoder and HTTP handler)
# ---------------------------------------------------------------------------
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
PAGE_TEMPLATE = """\
<html>
<head><title>Pi Camera Stream</title></head>
<body>
<h1>Pi Camera Stream</h1>
<p>Connect live_yolo.py to: <code>http://{host}:{port}/stream.mjpg</code></p>
<img src="stream.mjpg" width="{width}" height="{height}" />
</body>
</html>
"""

# These get set in main() before server starts
stream_output = None
page_html = ""


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()

        elif self.path == "/index.html":
            content = page_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with stream_output.condition:
                        stream_output.condition.wait()
                        frame = stream_output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress per-request logs to keep terminal clean for controls
        pass


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pi Camera MJPEG stream server")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        help="Width Height (default: 640 480)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Framerate (default: 30)")
    parser.add_argument("--port", type=int, default=8000,
                        help="HTTP port (default: 8000)")
    parser.add_argument("--save-dir", type=str, default="recordings",
                        help="Directory for local recordings (default: recordings/)")
    return parser.parse_args()


def main():
    global stream_output, page_html

    args = parse_args()
    width, height = args.resolution

    # ---- Init camera ----
    try:
        from picamera2 import Picamera2
        from picamera2.encoders import MJPEGEncoder, H264Encoder, Quality
        from picamera2.outputs import FileOutput, FfmpegOutput
    except ImportError:
        print("ERROR: picamera2 not installed. sudo apt install python3-picamera2")
        sys.exit(1)

    print(f"Starting camera: {width}x{height} @ {args.fps}fps")
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"size": (width, height)},
    )
    picam2.configure(video_config)

    # ---- Set up MJPEG streaming ----
    stream_output = StreamingOutput()
    stream_encoder = MJPEGEncoder()
    stream_file_output = FileOutput(stream_output)

    picam2.start()
    time.sleep(1)  # warm up + autofocus

    # Start the MJPEG encoder for streaming
    picam2.start_encoder(stream_encoder, stream_file_output)

    # ---- Get Pi's IP for display ----
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        pi_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pi_ip = "localhost"

    page_html = PAGE_TEMPLATE.format(
        host=pi_ip, port=args.port, width=width, height=height
    )

    # ---- Start HTTP server in background ----
    http_server = StreamingServer(("0.0.0.0", args.port), StreamingHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    print(f"\n--- Streaming at http://{pi_ip}:{args.port}/stream.mjpg ---")
    print(f"--- Preview at http://{pi_ip}:{args.port}/ ---")

    # ---- Recording state ----
    os.makedirs(args.save_dir, exist_ok=True)
    recording = False
    rec_encoder = None
    rec_output = None
    current_path = None
    last_saved_path = None

    kb = KeyboardListener()
    print("\n1 = Record | 2 = Stop | 3 = Save | q = Quit\n")

    try:
        while True:
            time.sleep(0.05)

            key = kb.get_key()

            if key == "1" and not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_path = os.path.join(args.save_dir, f"rec_{timestamp}.mp4")
                rec_encoder = H264Encoder()
                rec_output = FfmpegOutput(current_path)
                picam2.start_encoder(rec_encoder, rec_output, quality=Quality.HIGH)
                recording = True
                print(f"[REC] Started: {current_path}")

            elif key == "2" and recording:
                picam2.stop_encoder(rec_encoder)
                recording = False
                rec_encoder = None
                rec_output = None
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
        if recording and rec_encoder:
            try:
                picam2.stop_encoder(rec_encoder)
                print(f"[SAVED] Auto-saved: {current_path}")
            except Exception:
                pass
        try:
            picam2.stop_encoder(stream_encoder)
        except Exception:
            pass
        picam2.stop()
        http_server.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()