
import json
import os

FILE = "ultron_neural_core.json"

if not os.path.exists(FILE):
    print("Neural model file not found!")
    raise SystemExit

with open(FILE, "r", encoding="utf-8") as f:
    model = json.load(f)

print("\n========== NEURAL MODEL INSPECTOR ==========")

print("\nModel keys:")
for key in model:
    value = model[key]

    print(f"\n{key}: {type(value).__name__}")

    if isinstance(value, list):
        print("Length:", len(value))

        if value and isinstance(value[0], list):
            print("First row length:", len(value[0]))
            print("First row sample:", value[0][:5])
        else:
            print("Sample:", value[:10])

    elif isinstance(value, dict):
        print("Subkeys:", list(value.keys())[:15])

    else:
        print("Value:", value)

print("\n========== INSPECTION COMPLETE ==========")
