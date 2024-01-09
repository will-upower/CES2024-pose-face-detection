from pathlib import Path
from threading import Lock, Thread
import numpy as np
import cv2
import os
import time

import mmap
mmap_v4h_path = '/home/nvidia/mmap_exchange/v4h2_to_orin.dat'

class Camera:

    def __init__(self, vid, distort, w = 1920, h = 1080, oclEnable = False):
        self.pose_res   = (640, 360)
        self.distort    = distort
        self.oclEnable  = oclEnable

        self.pose_norm_mean         = np.ndarray(
            [self.pose_res[1] + 24, self.pose_res[0], 3], np.float32)
        self.pose_norm_std          = np.ndarray(
            [self.pose_res[1] + 24, self.pose_res[0], 3], np.float32)
        self.pose_norm_mean[:, :]   = np.array([0.485, 0.456, 0.406])
        self.pose_norm_std[:, :]    = 1 / np.array([0.229, 0.224, 0.225])

        try:
            vid = int(vid)
        except BaseException:
            vid = Path(vid)
            vid = vid.as_posix()
        
        self.f_frame_BGR    = os.open(mmap_v4h_path, os.O_RDONLY)
        self.mmFrameBGR     = mmap.mmap(self.f_frame_BGR, 1280 * 720 * 3, access=mmap.ACCESS_READ, offset=0)
        ret, frame          = self.sharedMread()
        self.res            = (1080, 1920)
        self.ret            = ret
        self.frame          = frame
        # vid         = cv2.VideoCapture(vid)
        # vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # vid.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # xres            = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # yres            = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # self.res        = (yres, xres)
        # self.vid        = vid

        self.lock       = Lock()
        self.started    = False
        self.thread     = Thread(target=self.update, args=())

        roi             = np.load(os.getcwd() + "/src/utils/roi.dat.npy")
        self.mapx       = np.load(os.getcwd() + "/src/utils/mapx.dat.npy")
        self.mapy       = np.load(os.getcwd() + "/src/utils/mapy.dat.npy")

        self.x, self.y, self.w, self.h  = roi

        """
        # Still images test
        self.fname      = self.getTestImgList("/home/ubuntu/Downloads/testimages/out_raw_7/")
        self.testImg    = cv2.imread(self.fname[0])
        self.lock2      = Lock()
        """

        # ret, frame      = self.vid.read()
        # self.ret        = ret
        # self.frame      = frame

        ret, frame, frame_rgb, pose_frame = self.produce_frames()
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
            time.sleep(0.05)
            ret, frame  = self.sharedMread()
            with self.lock:
                self.ret    = ret
                if frame is not None:
                    self.frame  = frame

    def produce_frames(self):
        with self.lock:
            ret, frame  = self.ret, self.frame
            """
            # Still images test
            with self.lock2:
                frame       = self.testImg.copy()
            """
            frameD      = frame.shape

            if not ret:
                return ret, None, None, None
            
            if self.oclEnable:
                uMatFrame   = cv2.UMat(frame)

                if self.distort:
                    # Undistortion            
                    uMatFrame   = cv2.remap(uMatFrame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                    frame       = cv2.UMat.get(uMatFrame)
                    frame       = frame[self.y : self.y+self.h, self.x : self.x+self.w]
                    uMatFrame   = cv2.UMat(frame)
                    uMatFrame   = cv2.resize(uMatFrame, (frameD[1], frameD[0]))

                # Mirroring
                # uMatFrame       = cv2.flip(uMatFrame, 1)
                uMatFrame_rgb   = cv2.cvtColor(uMatFrame, cv2.COLOR_YUV2RGB_YUYV)
                uMatPose_frame  = cv2.resize(uMatFrame_rgb, self.pose_res)
                uMatPose_frame  = cv2.copyMakeBorder(uMatPose_frame, 12, 12, 0, 0, cv2.BORDER_CONSTANT)
                frame           = cv2.UMat.get(uMatFrame)
                frame_rgb       = cv2.UMat.get(uMatFrame_rgb)
                pose_frame      = cv2.UMat.get(uMatPose_frame)

            else:
                if self.distort:
                    # Undistortion            
                    frame       = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                    frame       = frame[self.y : self.y+self.h, self.x : self.x+self.w]
                    frame       = cv2.resize(frame, (frameD[1], frameD[0]))

                # Mirroring
                frame       = cv2.flip(frame, 1)
                frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_frame  = cv2.resize(frame_rgb, self.pose_res)
                pose_frame  = cv2.copyMakeBorder(pose_frame, 12, 12, 0, 0, cv2.BORDER_CONSTANT)

            pose_frame  = pose_frame.astype(np.float32)
            pose_frame  *= (1 / 255.0)
            pose_frame  -= self.pose_norm_mean
            pose_frame  *= self.pose_norm_std
            pose_frame  = pose_frame.transpose(2, 0, 1)[np.newaxis]
            pose_frame  = np.ascontiguousarray(pose_frame)
  
        return ret, frame, frame_rgb, pose_frame

    def read(self):
        ret, frame, frame_rgb, pose_frame   = self.produce_frames()        

        return ret, frame, frame_rgb, pose_frame


    def sharedMread(self):
        self.mmFrameBGR.seek(0)
        recImg  = np.frombuffer(self.mmFrameBGR, dtype=np.uint8).reshape((720, 1280, 3))
        return True, recImg  

    def stop(self):
        if self.started == False:
            return
        self.started = False
        self.thread.join()

    def __del__(self, **kwargs):
        os.close(self.f_frame_BGR)
        self.mmFrameBGR.close()
        self.stop()