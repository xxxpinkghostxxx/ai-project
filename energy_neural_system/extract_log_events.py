import os
import re

LOG_PATH = os.path.join('logs', 'diagnostic_trace.txt')
EXTRACT_PATH = os.path.join('logs', 'diagnostic_trace_extract.txt')

patterns = [r'^\[WS ENERGY\]', r'^\[TRANSFER\]']

matches = []
try:
    with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if any(re.match(p, line) for p in patterns):
                matches.append(line)
            if len(matches) >= 1000:
                break
    with open(EXTRACT_PATH, 'w', encoding='utf-8') as f:
        for line in reversed(matches):
            f.write(line)
    print(f"Extracted {len(matches)} entries to {EXTRACT_PATH}")
except Exception as e:
    print(f"[ERROR] Could not extract log events: {e}") 