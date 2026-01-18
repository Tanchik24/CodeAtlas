import os
import signal
import subprocess
import sys
import time

from app.config import get_config

config = get_config().api

def _kill_port(port: str) -> None:
    subprocess.call(["bash", "-lc", f"fuser -k {port}/tcp >/dev/null 2>&1 || true"])


def main() -> None:
    api_host = config.host
    api_port = config.port
    ui_port = config.ui_port

    _kill_port(api_port)
    _kill_port(ui_port)

    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "app.api_app:app",
        "--host", api_host,
        "--port", str(api_port)
    ]

    ui_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app/ui.py",
        "--server.address", api_host,
        "--server.port", str(ui_port),
    ]

    print("Starting FastAPI:", " ".join(api_cmd))
    api = subprocess.Popen(api_cmd)

    time.sleep(1.0)

    print("Starting Streamlit:", " ".join(ui_cmd))
    ui = subprocess.Popen(ui_cmd)

    def stop(*_):
        for p in (ui, api):
            if p.poll() is None:
                p.terminate()
        time.sleep(0.5)
        for p in (ui, api):
            if p.poll() is None:
                p.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while True:
        if api.poll() is not None or ui.poll() is not None:
            stop()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
