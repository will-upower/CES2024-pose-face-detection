from pathlib import Path
from threading import Lock, Thread
import numpy as np
import cv2
import os

import mmap


class Camera_mmap:

    def __init__(self, distort, mmap_filename, image_height, image_width, w = 1920, h = 1080):
        self.res        = (image_width,image_height)
        self.pose_res   = (640, 360)
        self.distort    = distort
        self.filename = mmap_filename
        self.pose_norm_mean         = np.ndarray(
            [self.pose_res[1] + 24, self.pose_res[0], 3], np.float32)
        self.pose_norm_std          = np.ndarray(
            [self.pose_res[1] + 24, self.pose_res[0], 3], np.float32)
        self.pose_norm_mean[:, :]   = np.array([0.485, 0.456, 0.406])
        self.pose_norm_std[:, :]    = 1 / np.array([0.229, 0.224, 0.225])


        ### 
        #   Image in is initialized here
        #   TODO: remove vid and replace
        #   with frames being read in from mmap buffer
        ###
        # try:
        #     vid = int(vid)
        # except BaseException:
        #     vid = Path(vid)
        #     vid = vid.as_posix()

        # vid         = cv2.VideoCapture(vid)
        # vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # vid.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # xres            = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # yres            = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # self.res        = (yres, xres)
        # self.vid        = vid

        xres            = image_width
        yres            = image_height

        self.lock       = Lock()
        self.started    = False
        self.thread     = Thread(target=self.update, args=())

        roi             = np.load(os.getcwd() + "/src/utils/roi.dat.npy")
        self.mapx       = np.load(os.getcwd() + "/src/utils/mapx.dat.npy")
        self.mapy       = np.load(os.getcwd() + "/src/utils/mapy.dat.npy")

        self.x, self.y, self.w, self.h  = roi

        # ret, frame      = self.vid.read()
        frame = self.read_image_from_mmap(mmap_filename)
        self.ret        = ret
        self.frame      = frame

        ret, frame, frame_rgb, pose_frame = self.__read()
        self.ret        = ret
        self.frame      = frame
        self.frame_rgb  = frame_rgb
        self.pose_frame = pose_frame


    def start(self):
        if self.started:
            return
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            # ret, frame  = self.vid.read()
            ret, frame = self.read_image_from_mmap(self.filename)
            with self.lock:
                self.ret    = ret
                if frame is not None:
                    self.frame  = frame
    
    def read_image_from_mmap(filename, img_height, img_width, bpp):
        try:
            with open(filename, "rb") as f:
                mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Read and print the data
            data = mmapped_file.read()
            
            numpy_array = np.frombuffer(data, dtype=np.uint8)
            img_frame = numpy_array.reshape(img_height, img_width, bpp)
            
            return True, img_frame
        except OSError as e:
            return False, None

    
    def __read(self):
        with self.lock:
            ret, frame  = self.ret, self.frame

            frameD      = frame.shape

            if not ret:
                return ret, None, None, None
            else:
                if self.distort:
                    # Undistortion            
                    frame       = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                    frame       = frame[self.y : self.y+self.h, self.x : self.x+self.w]
                    frame       = cv2.resize(frame, (frameD[1], frameD[0]))

                # Mirroring
                # frame       = cv2.flip(frame, 1)
                # frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Frame read from mmap is already RGB
                frame_rgb   = cv2.flip(frame, 1)
                frame_bgr   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                pose_frame  = cv2.resize(frame, self.pose_res)
                pose_frame  = cv2.copyMakeBorder(pose_frame, 12, 12, 0, 0, cv2.BORDER_CONSTANT)

            pose_frame  = pose_frame.astype(np.float32)
            pose_frame  *= (1 / 255.0)
            pose_frame  -= self.pose_norm_mean
            pose_frame  *= self.pose_norm_std
            pose_frame  = pose_frame.transpose(2, 0, 1)[np.newaxis]
            pose_frame  = np.ascontiguousarray(pose_frame)
            
        # return ret, frame, frame_rgb, pose_frame
        return ret, frame_bgr, frame_rgb, pose_frame


    def read(self):
        ret, frame, frame_rgb, pose_frame   = self.__read()        

        return ret, frame, frame_rgb, pose_frame

    def stop(self):
        if self.started == False:
            return
        self.started = False
        self.thread.join()
        # self.vid.release()

    def __del__(self, **kwargs):
        self.stop()
