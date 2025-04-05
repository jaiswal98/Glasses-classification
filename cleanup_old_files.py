import os
import time
from datetime import datetime, timedelta

OUTPUT_DIR = 'static/output'
DAYS_TO_KEEP = 30

def cleanup_old_files():
    now = time.time()
    cutoff = now - (DAYS_TO_KEEP * 86400)  # 30 days in seconds

    for folder in os.listdir(OUTPUT_DIR):
        folder_path = os.path.join(OUTPUT_DIR, folder)
        if os.path.isdir(folder_path) or folder.endswith('.zip'):
            try:
                if os.path.getmtime(folder_path) < cutoff:
                    if os.path.isdir(folder_path):
                        print(f"Deleting folder: {folder_path}")
                        os.system(f'rmdir /S /Q "{folder_path}"')  # For Windows
                    else:
                        print(f"Deleting file: {folder_path}")
                        os.remove(folder_path)
            except Exception as e:
                print(f"Error deleting {folder_path}: {e}")

if __name__ == "__main__":
    cleanup_old_files()
