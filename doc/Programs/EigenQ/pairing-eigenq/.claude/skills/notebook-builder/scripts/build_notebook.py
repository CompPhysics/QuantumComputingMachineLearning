#!/usr/bin/env python3
"""Assemble a notebook from an ordered cell list. Import and call build().
Cells: list of ("md"|"code", source_string). Never hand-edit .ipynb JSON."""
import sys, nbformat as nbf

def build(cells, out_path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(s) if k == "md"
                   else nbf.v4.new_code_cell(s) for k, s in cells]
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3",
                                    "language": "python", "name": "python3"}
    nbf.write(nb, out_path)
    print(f"wrote {out_path} ({len(nb['cells'])} cells)")

if __name__ == "__main__":
    print("import build() from this module; pass cells=[('md'|'code', src), ...]")
