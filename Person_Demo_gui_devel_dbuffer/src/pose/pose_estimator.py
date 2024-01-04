# from ..fwdnxt import FWDNXT
import time

import numpy as np

import cv2
try:
    import gc
    import pyopencl as cl
except ModuleNotFoundError:
    print('OpenCL not found')
    pass
from skimage.draw import line
from math import sqrt
import torch
import sys

from .models.pose_model import HRFPN34 as Model

jnt_info = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
            [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# kpt_info : length 17
kpt_info = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle']

jnt_map_info = {k: v for k, v in enumerate(jnt_info)}
jnt_map_info = sorted(jnt_map_info.items(), key=lambda x: x[1][0])

scale = 1.5

# Flag value which is indicating not valid value in pair matrix "pairs_np"
NOTVALID = -9000.0

# jm buffer size = 240 x 135 x 4bytes
JM_OCL_BUFFERSIZE   = 129600

# disabling garbage collection due to prevent pybind11 crash
# gc.disable()
# gc.set_debug(gc.DEBUG_COLLECTABLE)


class PoseEstimator:

    def __init__(self, args, addr_off=None, cmem=None):

        self.inp_res = (640, 384)
        self.fie = (args.device == 'fie')
        self.cuda = (args.device == 'gpu')
        self.model = None

        self.nid = 0
        self.nimages = 0

        self.ie_addroff = None
        self.ie_handle = None

        self.hm_thresh = args.hm_thresh
        self.jm_thresh = args.jm_thresh

        self.window = args.window / 100.0
        self.cl_available = args.ocl

        if args.device == 'fie':
            self.ie = FWDNXT()
            self.ie.SetFlag('options', 'C')
            # Non-blocking mode
            self.ie.SetFlag('blockingmode', '0')

            if addr_off is not None:
                self.ie.SetFlag('addr_off', str(addr_off))

            # Check for bin file
            bin_path = args.bin_dir / 'pose.bin'
            if not bin_path.is_file() or True:
                # Check for onnx file
                onnx_path = args.onnx_dir / 'pose.onnx'
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

        if self.cl_available:
            self.plat = cl.get_platforms()
            self.cl_dev_cpu = self.plat[0].get_devices(
                device_type=cl.device_type.CPU)
            if len(self.cl_dev_cpu) == 0:
                self.cl_dev_cpu = self.plat[1].get_devices(
                    device_type=cl.device_type.CPU)
            self.cl_dev_gpu = self.plat[0].get_devices(
                device_type=cl.device_type.GPU)
            if len(self.cl_dev_gpu) == 0:
                self.cl_dev_gpu = self.plat[1].get_devices(
                    device_type=cl.device_type.GPU)

            # select gpu or cpu for pose estimation post processing.
            self.cl_dev = self.cl_dev_gpu
            self.ctx = cl.Context(self.cl_dev)
            self.queue = cl.CommandQueue(self.ctx)
            src = open("./src/pose/getPairScore.cl", "r").read()
            self.prg = cl.Program(self.ctx, src).build()
            mf = cl.mem_flags
            # OpenCL buffer allocation
            # memory allocation is assumed max 20 points.
            self.jm_x_g = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, JM_OCL_BUFFERSIZE)
            self.jm_y_g = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, JM_OCL_BUFFERSIZE)
            # 8 bytes per 1 point
            self.ptx1_np_g = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, 160)
            self.ptx2_np_g = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, 160)
            # pts length
            self.pts_len_g = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, 8)
            # jm length
            self.jm_len_g = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, 8)
            # maxinum pairs = 20 by 20
            self.pairs_g = cl.Buffer(
                self.ctx, mf.WRITE_ONLY | mf.ALLOC_HOST_PTR, 1600)

        # Pose model execution time
        self.time_model = 0
        # Static ROI margin (2 : Half, 4 :Quarter)
        self.staticMarg = 4
        self.startT = time.time()

    def load_model(self, args):
        if self.model is not None:
            return

        model = Model(len(kpt_info) + (2 * len(jnt_info)))
        state = torch.load(args.pose_weights.as_posix(), map_location='cpu')
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

    def pose_model_fnc(self, pose_frame):
        img = pose_frame
        if self.fie:
            if self.nimages >= 1:
                while self.ie.GetResult(self.result) is None:
                    time.sleep(0.001)
                self.nimages -= 1
                self.time_model = time.time() - self.startT
            self.startT     = time.time()
            self.ie.PutInput(img, self.nid)
            self.nid += 1 # id of each image
            self.nimages += 1 # number of images loaded
            preds   = self.result.reshape(-1, self.inp_res[1] // 4, self.inp_res[0] // 4)
            hms     = preds[:17]
            jms     = preds[17:]
        else:
            img = torch.Tensor(img).float()

            if self.cuda:
                img = img.cuda(non_blocking=True)

            preds = self.model(img)[-1][0]
            hms = preds[:17].data.cpu().numpy()
            jms = preds[17:].data.cpu().numpy()

        # Bug fix : 90x154 -> 90x160
        hms = hms[:, 3:-3, :]
        jms = jms[:, 3:-3, :]
        return hms, jms

    def pose_postproc_fnc(self, ih, iw, hms, jms, people_prev):
        # Create heatmap to overlay
        heatmap = hms.copy()
        heatmap = heatmap.max(axis=0)
        heatmap = cv2.resize(heatmap, (iw, ih))
        heatmap *= 255
        heatmap = np.clip(heatmap, 0, 255)
        heatmap = heatmap.astype(np.uint8)
        # setting color of heatmap
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        heatmap[:, :, 2] = 0
        heatmap[:, :, 1] = 0

        kpt = [[] for _ in range(len(kpt_info))]

        # Calculate window dimensions
        size = self.window
        p, h, w = hms.shape
        factor = (1 - size) / self.staticMarg
        f1 = factor
        f2 = 1 - factor

        z1 = int(f1 * w)
        z2 = int(f2 * w)

        # Create masks
        hms_mask = np.ones(hms.shape)
        hms_mask = np.float32(hms_mask)

        hms_mask[:, :, 0:z1] = 0
        hms_mask[:, :, z2:-1] = 0

        jms_mask = np.ones(jms.shape)
        jms_mask = np.float32(jms_mask)
        jms_mask[:, :, 0:z1] = 0
        jms_mask[:, :, z2:-1] = 0

        # Go through all existing points and see if any are beyond boundary
        if people_prev is not None:
            for p in people_prev:
                # Count number of points in each person
                pts = int((p.size - np.count_nonzero(p == -1)) / 2)

                # If a person has less than 'threshold' points, they are
                # considered to be NOT fully in the frame
                threshold = 6
                if pts < threshold:
                    continue

                # Create mask for all coordinates of this person
                for x, y in p:
                    if (x == -1) or (y == -1):
                        continue
                    if (x / iw) < z1 or (x / iw) > z2:
                        # Create rectangular mask for this point
                        x1 = int(((x / iw) - .10) * w)
                        y1 = int(((y / ih) - .10) * h)
                        x2 = int(((x / iw) + .10) * w)
                        y2 = int(((y / ih) + .10) * h)
                        hms_mask[:, y1:y2, x1:x2] = 1
                        jms_mask[:, y1:y2, x1:x2] = 1

        # Apply mask
        hms = np.multiply(hms, hms_mask)
        jms = np.multiply(jms, jms_mask)

        for i, hm in enumerate(hms):
            hm = cv2.erode(hm, np.ones((5, 5)))
            hm = cv2.resize(hm, None, fx=scale, fy=scale)
            mask = np.uint8(hm > self.hm_thresh)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                M = cv2.moments(cnt)
                try:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                except Exception:
                    continue

                kpt[i].append((cx, cy))
                continue

        sh, sw = hms[0].shape
        sh, sw = scale * sh, scale * sw
        def rescale_kpt(x, y): return (int(x * iw / sw), int(y * ih / sh))
        scaled_kpt = [[rescale_kpt(x, y) for (x, y) in k] for k in kpt]

        people = []
        pairsList = {}

        if self.cl_available:
            mf = cl.mem_flags

        for i, (src, dst) in jnt_map_info:
            pts1 = kpt[src - 1]
            pts2 = kpt[dst - 1]
            jm_x = cv2.resize(jms[i], None, fx=scale, fy=scale)
            jm_y = cv2.resize(jms[i + len(jnt_info)], None, fx=scale, fy=scale)
            pairs = {}

            if self.cl_available:
                # preventing crash during gc in multi threading
                gc.disable()
                # pair matching procedure by opencl
                if (len(pts1) > 0) and (len(pts2) > 0):
                    pts1_np = np.asarray(pts1).astype(np.float32)
                    pts2_np = np.asarray(pts2).astype(np.float32)
                    pairs_np = np.full(
                        (len(pts1_np), len(pts2_np)), NOTVALID, dtype=np.float32)
                    # pts lenath
                    pts_len = np.zeros(2, dtype=np.int32)
                    # pts1 length
                    pts_len[0] = len(pts1_np)
                    # pts2 length
                    pts_len[1] = len(pts2_np)
                    # jm_x & jm_y length (same size)
                    jm_len = np.asarray(jm_x.shape).astype(np.int32)

                    # OpenCL buffer updates
                    if (jm_x.shape[0] * jm_x.shape[1] * 4 ) > JM_OCL_BUFFERSIZE :
                        print("ERR : Joint map overflow ", sys.getsizeof(jm_x), " ", jm_x.shape)

                    cl.enqueue_copy(self.queue, self.jm_x_g, jm_x)
                    cl.enqueue_copy(self.queue, self.jm_y_g, jm_y)
                    cl.enqueue_copy(self.queue, self.ptx1_np_g, pts1_np)
                    cl.enqueue_copy(self.queue, self.ptx2_np_g, pts2_np)
                    cl.enqueue_copy(self.queue, self.pts_len_g, pts_len)
                    cl.enqueue_copy(self.queue, self.jm_len_g, jm_len)
                    cl.enqueue_copy(self.queue, self.pairs_g, pairs_np)

                    # OpenCL kernel function execution
                    self.prg.getPairScore(
                        self.queue,
                        (len(pts1_np),
                         len(pts2_np),
                            1),
                        None,
                        self.ptx1_np_g,
                        self.ptx2_np_g,
                        self.jm_x_g,
                        self.jm_y_g,
                        self.pts_len_g,
                        self.jm_len_g,
                        self.pairs_g)
                    cl.enqueue_copy(self.queue, pairs_np, self.pairs_g)

                    # Gathers valid outputs. (If values are NOTVALID, this pair is
                    # not valid.)
                    m, n = np.where(pairs_np > NOTVALID)
                    for j in range(len(m)):
                        pairs[(m[j], n[j])] = pairs_np[(m[j], n[j])]

                    pairs = sorted(
                        pairs.items(),
                        key=lambda x: x[1],
                        reverse=True)
                # Enabling gc again after OpenCL execution.
                gc.enable()
            else:
                for m, pt1 in enumerate(pts1):
                    for n, pt2 in enumerate(pts2):
                        #pts = np.array(self.get_line(pt1, pt2))
                        rr, cc = self.get_line(pt1, pt2)

                        x1, y1 = pt1
                        x2, y2 = pt2
                        dx, dy = x2 - x1, y2 - y1
                        norm = sqrt(dx**2 + dy**2)
                        if norm > 0:
                            ux, uy = dx / norm, dy / norm
                        else:
                            continue
                        paf_score = jm_x[rr, cc] * ux + jm_y[rr, cc] * uy
                        score = paf_score.mean()
                        correct_ratio = len(np.where(paf_score > 0.40)[
                                            0]) / len(paf_score)
                        if correct_ratio < self.jm_thresh:
                            continue
                        pairs[(m, n)] = score * correct_ratio

                pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
            pairsList[str(i)] = pairs

        for i, (src, dst) in jnt_map_info:
            taken_pt1 = []
            taken_pt2 = []
            pairs = pairsList[str(i)]

            for (m, n), score in pairs:
                if m in taken_pt1 or n in taken_pt2:
                    continue

                for k, person in enumerate(people):
                    # If both slots are free, new person
                    if person[src - 1] == -1 and person[dst - 1] == n:
                        taken_pt1.append(m)
                        taken_pt2.append(n)
                        person[src - 1] = m
                        break
                    elif person[src - 1] == m and person[dst - 1] == -1:
                        taken_pt1.append(m)
                        taken_pt2.append(n)
                        person[dst - 1] = n
                        break
                    elif person[src - 1] == m and person[dst - 1] == n:
                        taken_pt1.append(m)
                        taken_pt2.append(n)
                        break
                else:
                    new_person = np.ones(len(kpt_info), dtype=np.int32)
                    new_person = new_person * -1
                    new_person[src - 1] = m
                    new_person[dst - 1] = n
                    taken_pt1.append(m)
                    taken_pt2.append(n)
                    people.append(new_person)

        people_pts = []
        while len(people) > 0:
            p1 = people.pop(0)

            if np.all(p1 == -1):
                continue

            for p2 in people:
                if np.any(p1 == p2) and np.any(p1[np.where(p1 == p2)] != -1):
                    zero_check = ((p1[np.where(p1 != p2)] + 1)
                                  * (p2[np.where(p1 != p2)] + 1))
                    if np.all(zero_check == 0):
                        for i in range(len(kpt_info)):
                            if p1[i] == -1:
                                p1[i] = p2[i]
                        p2[:] = -1
            people_pts.append(p1)
        people = people_pts

        people_pts = []
        for person in people:
            new_person = np.ones((len(kpt_info), 2), dtype=np.int32)
            new_person = new_person * -1
            for i in range(len(new_person)):
                if person[i] != -1:
                    new_person[i] = scaled_kpt[i][person[i]]
            people_pts.append(new_person)
        people = people_pts

        return people, heatmap

    # @profile

    def __call__(self, frame, pose_frame):
        """
        deleted
        """

    def __del__(self, **kwargs):
        while self.nimages > 0:
            while self.ie.GetResult(self.result) is None:
                time.sleep(0.001)
            self.nimages -= 1
        self.nid = 0 # id of each image

        try:
            self.fie.Free()
        except BaseException:
            pass

    def get_line(self, p1, p2):
        "Bresenham's line algorithm"
        x0, y0 = p1
        x1, y1 = p2
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        return line(y0, x0, y1, x1)
