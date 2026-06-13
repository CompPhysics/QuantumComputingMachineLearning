#!/usr/bin/env python3
"""Extract PNG figures from an executed notebook into a directory, in cell order.
Usage: extract_figures.py NOTEBOOK OUTDIR"""
import sys, os, json, base64
nb = json.load(open(sys.argv[1])); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
i = 0
for ci, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}):
            i += 1
            open(f"{outdir}/fig{i}.png", "wb").write(base64.b64decode(o["data"]["image/png"]))
            print(f"fig{i}.png  <- cell {ci}")
print(f"wrote {i} figures to {outdir}")
