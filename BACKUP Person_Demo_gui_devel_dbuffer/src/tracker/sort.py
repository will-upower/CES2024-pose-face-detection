import numpy as np
from scipy.optimize import linear_sum_assignment

from .ciou import iou
from .kalman import KalmanBoxTracker

KalmanBoxTracker.time_since_update = 1


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.2):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    lendet = len(detections)
    lentrk = len(trackers)

    if (lentrk == 0):
        return np.empty((0, 2), dtype=int), np.arange(
            lendet), np.empty((0, 4), dtype=int)
    if (lendet == 0):
        return np.empty((0, 2), dtype=int), np.empty(
            (0, 4), dtype=int), np.arange(lentrk)

    detections = detections.astype(np.float32)
    trackers = trackers.astype(np.float32)

    x11, y11, x12, y12 = np.split(detections, 4, axis=1)
    x21, y21, x22, y22 = np.split(trackers, 4, axis=1)
    iou_matrix = iou(x11, y11, x12, y12, x21, y21, x22, y22)

    iou_matrix[iou_matrix < iou_threshold] = 0.
    matched_indices = linear_sum_assignment(-iou_matrix)

    costs = iou_matrix[matched_indices]
    matches = matched_indices[0][np.where(
        costs)[0]], matched_indices[1][np.where(costs)[0]]
    unmatched_detections = np.where(
        np.in1d(range(lendet), matches[0], invert=True))[0]
    unmatched_trackers = np.where(
        np.in1d(range(lentrk), matches[1], invert=True))[0]

    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)

    return matches, unmatched_detections, unmatched_trackers


class Sort(object):

    def __init__(self, max_age=1, min_hits=3, max_people=100):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.conf = 0
        self.max_people = max_people

    def update(self, faces, isRecording):
        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = []
        del_trks = []
        for t, tracker in enumerate(self.trackers):
            pos = self.trackers[t].predict()[0]
            if np.any(np.isnan(pos)):
                del_trks.append(tracker)
            trks.append(pos)

        # Remove items that have NaN for position
        for item in del_trks:
            print(item)
            self.trackers.remove(item)

        trks = np.array(trks, dtype=np.int32)
        dets = []

        for (x1, y1, x2, y2, *_) in faces:
            dets.append([x1, y1, x2, y2])

        dets = np.array(dets, dtype=np.int32)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[0][np.where(matched[1] == t)[0]][0]
                trk.update(faces[d], isRecording)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(faces[i], self.max_people)
            self.trackers.append(trk)

        tracking_list = []
        del_list = []

        for i, trk in enumerate(self.trackers):
            bbox, direction, *_ = trk.get_state()
            bbox = bbox[0]

            # Mark tracks for deletion if they have not been updated in a long
            # time
            if(trk.time_since_update > self.max_age):
                del_list.append(trk)
            else:
                if((trk.time_since_update < KalmanBoxTracker.time_since_update) and
                        (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                    trk.visible = True
                    if trk.id == 0:
                        trk.id = KalmanBoxTracker.count + 1
                        KalmanBoxTracker.count += 1
                else:
                    trk.visible = False
                tracking_list.append(trk)

        for item in del_list:
            self.trackers.remove(item)
        return tracking_list
