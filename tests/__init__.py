from pathlib import Path
import sys

# Add the root directory of your project to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))