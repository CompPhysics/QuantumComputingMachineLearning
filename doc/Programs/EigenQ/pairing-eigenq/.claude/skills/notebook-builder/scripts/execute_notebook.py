#!/usr/bin/env python3
"""Execute a notebook in place. Usage: execute_notebook.py NB [timeout_s]."""
import sys, subprocess
nb = sys.argv[1]; timeout = sys.argv[2] if len(sys.argv) > 2 else "1200"
cmd = ["python3", "-m", "jupyter", "nbconvert", "--to", "notebook",
       "--execute", "--inplace", f"--ExecutePreprocessor.timeout={timeout}", nb]
sys.exit(subprocess.call(cmd))
