import time

import numpy as np

import cv2
import torch

# from ..fwdnxt import FWDNXT
from .models.model_irse import IR_50 as Model
from .models.model_irse import l2_norm


class FaceIdentifier:
    INPUT_SIZE = 112

    def __init__(self, args, pose, addr_off=None, cmem=None):

        self.pose = pose

        self.inp_res = (FaceIdentifier.INPUT_SIZE, FaceIdentifier.INPUT_SIZE)
        self.fie = args.device == 'fie'
        self.cuda = args.device == 'gpu'
        self.model = None

        self.nid = 0
        self.nimages = 0

        self.ie_addroff = None
        self.ie_handle = None

        self.norm_mean = np.ndarray(
            [self.inp_res[1], self.inp_res[0], 3], np.float32)
        self.norm_std = np.ndarray(
            [self.inp_res[1], self.inp_res[0], 3], np.float32)
        self.norm_mean[:] = 127.5
        self.norm_std[:] = 1 / 128.0

        if args.device == 'fie':
            self.ie = FWDNXT()
            self.ie.SetFlag('options', 'Ck')

            if addr_off is not None:
                self.ie.SetFlag('addr_off', str(addr_off))

            # Check for bin file
            bin_path = args.bin_dir / 'face.bin'
            if not bin_path.is_file() or True:
                # Check for onnx file
                onnx_path = args.onnx_dir / 'face.onnx'
                if not onnx_path.is_file():
                    # Load model and generate onnx
                    self.load_model(args)
                    self.gen_onnx(onnx_path.as_posix())
                # Compile onnx
                onnx_path = onnx_path.as_posix()
                self.ie.Compile(
                    '%dx%dx3' % self.inp_res,
                    onnx_path,
                    bin_path.as_posix(),
                    1,
                    4
                )
            bin_path = bin_path.as_posix()

            # Initialize with model bin
            bitfile = args.bitfile if args.load else ''
            args.load = False
            result_size = self.ie.Init(bin_path, bitfile, cmem)
            self.result = np.ndarray(result_size, dtype=np.float32)
            self.ie_addroff = self.ie.GetInfo('addr_off')
            self.ie_handle = self.ie.get_handle()
        else:
            self.load_model(args)

        # Face ID model execution time
        self.time_model = 0

        # Preventing abnormal CPU load(100%)(It happens when face ID
        # is never running.
        if self.fie:
            dummyImg    = np.zeros((112,112,3))
            dummyImg    = np.array(dummyImg, dtype=np.float32)
            dummyImg    = dummyImg.transpose(2, 0, 1)[np.newaxis]
            dummyImg    = np.ascontiguousarray(dummyImg, dtype=np.float32)
            self.ie.PutInput(dummyImg, 0)
            self.ie.GetResult(self.result)

    def load_model(self, args):
        if self.model is not None:
            return

        model = Model(self.inp_res)
        state = torch.load(args.face_weights.as_posix(), map_location='cpu')
        model.load_state_dict(state)

        if self.cuda:
            model = model.cuda()

        model = model.eval()
        self.model = model
        return

    def gen_onnx(self, onnx_path):
        torch.onnx.export(
            self.model,
            torch.Tensor(
                1,
                3,
                *
                reversed(self.inp_res)),
            onnx_path)

    def __call__(self, image):
        img = np.array(image, dtype=np.float32)
        img -= self.norm_mean
        img *= self.norm_std
        img = img.transpose(2, 0, 1)[np.newaxis]
        img = np.ascontiguousarray(img, dtype=np.float32)

        if self.fie:
            while self.pose.nimages > 0:
                while self.pose.ie.GetResult(self.pose.result) is None:
                    time.sleep(0.001)
                self.pose.nimages -= 1
            self.pose.time_model    = time.time() - self.pose.startT
            self.pose.nid           = 0 # id of each image
            startT                  = time.time()
            self.ie.PutInput(img, 0)
            self.ie.GetResult(self.result)
            self.time_model = time.time() - startT
            feat = np.reshape(self.result, (1, 512))
        else:
            img = torch.Tensor(img)
            if self.cuda:
                img = img.cuda()
            feat = self.model(img)
            feat = feat.detach().cpu().numpy()

        feat = torch.Tensor(feat)
        feat = l2_norm(feat)
        feat = feat.numpy()

        return feat
