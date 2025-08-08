import json
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python convert.py <filename.jsonl>")
    sys.exit(1)

input_file = sys.argv[1]

if not input_file.endswith(".jsonl"):
    print("Error: Input file must have a .jsonl extension")
    sys.exit(1)

output_file = os.path.splitext(input_file)[0] + ".json"

with open(input_file, "r") as f:
    data = [json.loads(line) for line in f]

with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Converted {input_file} to {output_file}")
