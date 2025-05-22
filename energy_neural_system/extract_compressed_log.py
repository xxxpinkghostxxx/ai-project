import gzip
import os

LOG_PATH = os.path.join('logs', 'diagnostic_trace.json.gz')
OUTPUT_DIR = 'logs'
MAX_SIZE = int(1.8 * 1024 * 1024)  # 1.8MB

try:
    file_idx = 1
    out_path = os.path.join(OUTPUT_DIR, f'diag_extract_{file_idx}.txt')
    out_f = open(out_path, 'w', encoding='utf-8')
    current_size = 0
    with gzip.open(LOG_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            encoded = line if line.endswith('\n') else line + '\n'
            size = len(encoded.encode('utf-8'))
            if current_size + size > MAX_SIZE:
                out_f.close()
                file_idx += 1
                out_path = os.path.join(OUTPUT_DIR, f'diag_extract_{file_idx}.txt')
                out_f = open(out_path, 'w', encoding='utf-8')
                current_size = 0
            out_f.write(encoded)
            current_size += size
    out_f.close()
    print(f'Extraction complete: {file_idx} file(s) created in {OUTPUT_DIR}')
except Exception as e:
    print(f'[ERROR] Could not extract compressed log: {e}') 