import cv2
import numpy as np
import mmap


def mmap_buffer(width, height):
    with open("orin_to_v4h2.dat", "rb") as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # Read and print the data
    data = mmapped_file.read()
    
    numpy_array = np.frombuffer(data, dtype=np.uint8)
    numpy_array = numpy_array.reshape((height, width, 3))
    
    return numpy_array


def convert_yuv_to_bgr(yuv_image):
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image

file_path = 'test.raw'
output_path = 'output.bmp'
jpeg_filename = "output.jpg"
width = 1920
height = 1080
while True:
    rgb_image = mmap_buffer(width,height)
    #cv2.imwrite(jpeg_filename, rgb_image)
    cv2.imshow("Image", rgb_image)
    cv2.waitKey(1)

cv2.destroyAllWindows()
