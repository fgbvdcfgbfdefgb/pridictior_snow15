#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitcoin_predictor.hardware import choose_model_profile, detect_hardware


if __name__ == "__main__":
    hardware = detect_hardware()
    print(json.dumps({"hardware": hardware, "selected_profile": choose_model_profile(hardware).to_dict()}, indent=2))
