import os
import re

INPUT_LOG = os.path.join('logs', 'ai_system_20250520_194743.log')  # Change as needed
OUTPUT_LOG = os.path.join('logs', 'filtered_transfer.log')
PATTERN = r'\[TRANSFER\]'

try:
    with open(INPUT_LOG, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(OUTPUT_LOG, 'w', encoding='utf-8') as outfile:
        count = 0
        for line in infile:
            if re.search(PATTERN, line):
                outfile.write(line)
                count += 1
    print(f'Filtered {count} lines to {OUTPUT_LOG}')
except Exception as e:
    print(f'[ERROR] Could not filter log: {e}') 