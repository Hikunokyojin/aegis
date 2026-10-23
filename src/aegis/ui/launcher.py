"""`aegis-ui`: starts the Streamlit web UI (aegis/ui/app.py) on localhost."""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).with_name("app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                              "--server.address", "localhost", "--browser.gatherUsageStats", "false",
                              *sys.argv[1:]]))


if __name__ == "__main__":
    main()
