import runpy
import sys
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_DIR / "saas"))
APP = DEPLOY_DIR / "saas" / "app.py"
runpy.run_path(str(APP), run_name="__main__")
