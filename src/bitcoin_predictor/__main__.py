"""Small package entry point for the hardware probe."""

import json
from .hardware import choose_model_profile, detect_hardware

if __name__ == "__main__":
    resources = detect_hardware()
    print(json.dumps({"hardware": resources, "profile": choose_model_profile(resources).to_dict()}, indent=2))
