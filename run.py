import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import run  # noqa: E402

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nFailed to start Avatar AI Studio:\n  {exc}\n")
        input("Press Enter to close…")
        raise SystemExit(1) from exc