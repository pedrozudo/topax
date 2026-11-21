#!/usr/bin/env python3
from pathlib import Path
import sys


EVALUATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_DIR))
from two_dimensional import design_main


if __name__ == "__main__":
    design_main("sigmund_2007")
