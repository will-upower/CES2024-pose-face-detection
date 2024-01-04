import cv2
import numpy as np
import mmap
import os
import array
import time
import threading
size = 1280 * 720 * 3

# Used by orin_pub
#mmap_inf_path = '/home/nvidia/mmap_exchange/orin_to_v4h2.dat'
mmap_inf_path = '/home/nvidia/mmap_exchange/orin_to_v4h2.dat'
# Used by orin_sub
mmap_v4h_path = '/home/nvidia/mmap_exchange/v4h2_to_orin.dat'

# mmap to be used
mmap_file_inference = None
mmap_file_v4h = None
file_inference = None
file_v4h = None

thread_lock = threading.RLock()

# reads the mmap produced by inferencing. Used by orin_pub
def open_orin_inference_mmap():
    global mmap_file_inference, file_inference
    attempts = 0
    while True:
        try:
            if attempts < 5:
                file_inference = os.open(mmap_inf_path, os.O_RDONLY)
                mmap_file_inference = mmap.mmap(file_inference, 0, mmap.MAP_SHARED, mmap.PROT_READ)
            else:
                file_handle = os.open(mmap_inf_path, os.O_RDWR | os.O_CREAT | os.O_EXCL)
                os.truncate(file_handle, size)
                mmap_file_inference = mmap.mmap(file_handle, size, mmap.MAP_SHARED, mmap.PROT_READ)
                print(f"Warning: File not found, creating ad-hoc mmap buffer")
            return True
        except (FileNotFoundError, OSError) as e:
            print(f"Error: {e}")
            attempts += 1
        time.sleep(0.025)
            

# writes the mmap from v4h => orin. Used by orin_sub
def open_from_v4h2_mmap():
    global mmap_file_v4h, file_v4h    
    try:
        file_v4h = os.open(mmap_v4h_path, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
        os.truncate(file_v4h, size)
        mmap_file_v4h = mmap.mmap(file_v4h, size, mmap.MAP_SHARED, mmap.PROT_WRITE)
    except OSError as e:
        print(f"Error: {e}")
    
# closes the mmap from inferencing
def close_orin_inference_mmap():
    mmap_file_inference.close()
    try:
        os.close(file_inference)
    except (FileNotFoundError, OSError) as e:
        print(f"Error: {e}")

# closes the mmap from v4h => orin
def close_from_v4h2_mmap():
    mmap_file_v4h.close()
    try:
        os.close(file_inference)
    except (FileNotFoundError, OSError) as e:
        print(f"Error: {e}")

#orin publisher. Reads mmap, compresses data, and then returns a compressed image to send back to the v4h
def orin_pub_mmap(width=1280, height=720):
    with thread_lock:
        mmap_file_inference.seek(0)
        data = mmap_file_inference.read()
    if not data:
        print(f"Warning: mmap from inferencing is empty")
        numpy_array = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        numpy_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    
    compressed = cv2.imencode('.jpg', numpy_array, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1]
    return array.array('B', compressed.tobytes()) 
    
#orin subscriber. Takes the ros2 data, decompresses it, and then writes it to mmap
def orin_sub_mmap(data):
    numpy_array = np.frombuffer(data, np.uint8)
    numpy_array = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

    with thread_lock:
        mmap_file_v4h.seek(0)
        mmap_file_v4h.write(numpy_array.data)
