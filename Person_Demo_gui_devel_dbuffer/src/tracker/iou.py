import numpy as np
from numba.pycc import CC

cc = CC('ciou')


@cc.export(
    'iou', 'f8[:, :](f4[:, :], f4[:, :], f4[:, :], f4[:, :], f4[:, :], f4[:, :], f4[:, :], f4[:, :])')
def iou(x11, y11, x12, y12, x21, y21, x22, y22):
    #x11, y11, x12, y12 = box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3]
    #x21, y21, x22, y22 = box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3]
    min_ymax = np.minimum(y12, np.transpose(y22))
    max_ymin = np.maximum(y11, np.transpose(y21))
    inter_h = np.maximum(np.zeros(min_ymax.shape), min_ymax - max_ymin)
    min_xmax = np.minimum(x12, np.transpose(x22))
    max_xmin = np.maximum(x11, np.transpose(x21))
    inter_w = np.maximum(np.zeros(min_xmax.shape), min_xmax - max_xmin)
    inter_a = inter_h * inter_w
    box1_a = (x12 - x11) * (y12 - y11)
    box2_a = (x22 - x21) * (y22 - y21)
    union = box1_a + np.transpose(box2_a) - inter_a
    return inter_a / np.add(union, 1e-4)


if __name__ == "__main__":
    cc.compile()
