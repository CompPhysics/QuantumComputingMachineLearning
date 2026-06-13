#!/usr/bin/env python3
"""Validate an executed notebook: no errors, >= min figures, anchor numbers present.
Usage: check_notebook.py NB [min_figs]"""
import sys, json
nb = json.load(open(sys.argv[1])); min_figs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
errs, figs, text = [], 0, []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error": errs.append((i, o.get("ename")))
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}): figs += 1
        if "text" in o: text.append("".join(o["text"]))
blob = "\n".join(text)
anchors = ["0.794697", "0.635548", "acceptance"]   # FCI(k=3), FCI(k=4), rodeo
missing = [a for a in anchors if a not in blob]
ok = not errs and figs >= min_figs and not missing
print(f"errors={errs or 'none'}  figures={figs} (min {min_figs})  missing_anchors={missing or 'none'}")
print("CHECK PASSED" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
