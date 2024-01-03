import numpy as np
import torch
from torch import nn
from torchvision.models import resnet
import time

from ..fwdnxt import FWDNXT


class SpeechIdentifier:

    def __init__(self, args, pose, addr_off=None, cmem=None):

        self.pose = pose

        self.inp_res = (224, 224)

        self.nid = 0
        self.nimages = 0

        self.fie = args.device == 'fie'
        self.cuda = args.device == 'gpu'
        self.model = None

        self.ie_addroff = None
        self.ie_handle = None

        if args.device == 'fie':
            self.ie = FWDNXT()
            self.ie.SetFlag('options', 'C')

            if addr_off is not None:
                self.ie.SetFlag('addr_off', str(addr_off))

            # Check for bin file
            bin_path = args.bin_dir / 'voice.bin'
            if not bin_path.is_file() or True:
                # Check for onnx file
                onnx_path = args.onnx_dir / 'voice.onnx'
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

        self.time_model = 0

    def load_model(self, args):
        if self.model is not None:
            return

        model = resnet.resnet18(num_classes=1251)
        state = torch.load(args.voice_weights.as_posix(), map_location='cpu')
        model.load_state_dict(state)
        model = nn.Sequential(*list(model.children())[:-1])

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
                reversed(
                    self.inp_res)),
            onnx_path)

    def __call__(self, image):
        if self.fie:
            while self.pose.nimages > 0:
                while self.pose.ie.GetResult(self.pose.result) is None:
                    time.sleep(0.001)
                self.pose.nimages -= 1
            self.pose.time_model    = time.time() - self.pose.startT
            self.pose.nid           = 0 # id of each image

            startT          = time.time()  
            self.ie.PutInput(image, 0)
            self.ie.GetResult(self.result)
            self.time_model = time.time() - startT
            feat = np.reshape(self.result, (1, 512))
        else:
            img = torch.Tensor(image)
            if self.cuda:
                img = img.cuda(non_blocking=True)
            feat = self.model(img.unsqueeze(0))
            feat = feat.data.cpu().numpy()
            feat = np.reshape(feat, (1, 512))
        return feat[0]
