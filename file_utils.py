import os
import cv2
from datetime import datetime
from config import RAW_DIR, DECODED_DIR, INFERRED_DIR

def ensure_directories():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DECODED_DIR, exist_ok=True)
    os.makedirs(INFERRED_DIR, exist_ok=True)



def saveFiles(raw,decode,infered):
    # 📁 Ensure directories exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DECODED_DIR, exist_ok=True)
    os.makedirs(INFERRED_DIR, exist_ok=True)

    # 🕒 create timestamp (unique filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")


    # 📁 paths
    raw_path = RAW_DIR + "/" + timestamp + ".jpg"
    decoded_path = DECODED_DIR + "/" + timestamp + ".jpg"
    inferred_path = INFERRED_DIR + "/" + timestamp + ".jpg"


    # 💾 Save original raw image
    with open(raw_path, "wb") as f:
        f.write(raw)

    # 💾 Save decoded image
    cv2.imwrite(decoded_path, decode)


    # 💾 Save infered image
    cv2.imwrite(inferred_path, infered)