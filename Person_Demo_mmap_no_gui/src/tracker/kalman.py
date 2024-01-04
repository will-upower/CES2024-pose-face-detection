import time
from collections import deque
from enum import Enum

import numpy as np

import cv2
from filterpy.kalman import KalmanFilter
from scipy.ndimage import median_filter
from scipy.spatial import distance

WAVING_DETECT_THRESH                = 0.1
FACEID_RETRY_FORCE_GESTURE_THRESH   = 40


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2
    y = bbox[1] + h / 2
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score is None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] +
                         w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] +
                         w / 2., x[1] + h / 2., score]).reshape((1, 5))


def normalize_pose(pose):
    pose = np.array(pose)

    xmin = pose[:, 0].min()
    xmax = pose[:, 0].max()
    ymin = pose[:, 1].min()
    ymax = pose[:, 1].max()

    w = xmax - xmin
    h = ymax - ymin

    pose[:, 0] = pose[:, 0] - xmin
    pose[:, 1] = pose[:, 1] - ymin

    pose[:, 0] = pose[:, 0] / w
    pose[:, 1] = pose[:, 1] / h

    return pose


class ObjTrkState(Enum):
    FACE_REC = 0
    VOICE_COUNT = 1
    VOICE_REC = 2
    POSE_REC = 3
    COMPLETED = 4
    IDLE = 5


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    time_since_update = 1

    def __init__(self, face, max_people):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf                     = KalmanFilter(dim_x = 7, dim_z = 4)

        self.kf.F                   = np.array([[1, 0, 0, 0, 1, 0, 0],
                                                  [0, 1, 0, 0, 0, 1, 0],
                                                  [0, 0, 1, 0, 0, 0, 1],
                                                  [0, 0, 0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H                   = np.array([[1, 0, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:]           *= 10.
        self.kf.P[4:, 4:]           *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1]           *= 0.01
        self.kf.Q[4:, 4:]           *= 0.01

        bbox = face[:4]
        self.kf.x[:4]               = convert_bbox_to_z(bbox)
        self.time_since_update      = 0
        self.id                     = 0
        self.history                = []
        self.hits                   = 0
        self.hit_streak             = 0
        self.age                    = 0

        self.uid                    = None
        # Face ID confidence
        self.conf                   = 0
        self.complete_feats         = False
        self.recording              = False
        # Face direction
        self.last_direction         = face[4]
        # Croped face image
        self.last_crop              = face[6]
        # Pose data
        self.last_pose              = face[5]

        self.run_voice              = False

        self.wave_states            = deque(maxlen=30)
        self.waved                  = False

        self.pose_timer             = None
        self.voice_timer            = None

        self.saved_pose             = None
        self.saved_voice            = None

        self.voice_recorded         = False
        self.pose_locked            = True
        self.voice_locked           = True
        self.pose_dist              = 0
        self.voice_dist             = 0

        self.pose_thresh            = 20
        self.voice_thresh           = 0.10

        self.pose_wait_time         = 5  # secs
        self.voice_wait_time        = 5  # secs + 5secs (record time)

        self.checked                = False

        self.visible                = False
        #YS change
        self.prev_forehand_angle    = 0

        self.instruction_state      = ObjTrkState.IDLE
        
        # Is voice tested?
        self.voice_checked          = False
        # Face ID test count
        self.faceIDTestCount        = 0
        self.faceIDTestValue        = np.zeros(max_people)
        self.faceIDConfOrder        = []
        self.faceConfOrderI         = 0

        # Last failed face position and size
        self.lastFacdPos            = (0, 0)
        self.lastFaceSize           = 0
        self.numFaceIDCheck         = 0

        self.faceIDRetryCnt         = 0
        self.faceIDRetryCoolDn      = 0
        self.faceIDRetryByG         = False

    def update(self, face, isRecording):
        """
        Updates the state vector with observed bbox.
        """
        bbox                = face[:4]
        pose                = face[5]

        self.last_direction = face[4]
        self.last_crop      = np.array(face[6])
        self.last_pose      = pose

        # wave detection
        # During recording, hand waving shall not work.
        # Hand waving detection shall work only when people see camera.(direction = 1)
        if isRecording != True and self.last_direction == 1:
            wave_state = self.get_wave_state(pose)
            if wave_state is None and len(self.wave_states) > 0:
                wave_state = self.wave_states[-1]

            if wave_state is not None:
                self.wave_states.append(wave_state)

            try:
                waves = np.absolute(
                    np.gradient(
                        median_filter(
                            self.wave_states,
                            size=5))).sum()
            except Exception as e:
                waves = 0

            if waves > 8 and not self.recording:
                self.waved  = True

            # Forced face id retry gesture(Rising left hand for 3 sec) detection
            # Face ID retry forcing triger has 3 sec cool down time.
            if self.faceIDRetryCoolDn <= 0:
                if self.get_lefthandup_state(pose) is True:
                    self.faceIDRetryCnt += 1
                    if self.faceIDRetryCnt > FACEID_RETRY_FORCE_GESTURE_THRESH:
                        self.faceIDRetryByG     = True
                        self.faceIDRetryCoolDn  = FACEID_RETRY_FORCE_GESTURE_THRESH
                else:
                    self.faceIDRetryCnt = 0
            else:
                self.faceIDRetryCoolDn  -= 1

        # Save pose based on timer
        if self.saved_pose is not None:
            self.check_pose(pose)

        if self.instruction_state == ObjTrkState.POSE_REC and self.pose_locked == False:
            self.instruction_state = ObjTrkState.COMPLETED

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > KalmanBoxTracker.time_since_update):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        kalman_return = self.history[-1]
        return kalman_return

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)[0].astype(np.int32), self.last_direction, self.last_crop

    def set_recording(self, status):
        if status == False:
            self.reset_wave_state()
            self.recording = False
            return

        if not self.checked and self.complete_feats and self.uid is not None:
            return False

        self.reset_wave_state()
        self.recording = True
        return True

    def get_wave_state(self, pose):

        # Image flip : 6/8/10 -> 5/7/9
        right_shoulder  = pose[5]
        right_elbow     = pose[7]
        right_wrist     = pose[9]

        if (np.any(right_shoulder == -1)    or
            np.any(right_elbow == -1)       or
            np.any(right_wrist == -1)):
            return None
        # if down, no wave
        # kaneeun modified : Dont need to rise elbow over shoulder.

        if (right_wrist[1] - right_elbow[1]) > 0:
            return None

        forearm         = right_wrist - right_elbow
        length          = np.linalg.norm(forearm)

        if length > 0:
            forearm = forearm / length

        arm             = right_elbow - right_shoulder
        length          = np.linalg.norm(arm)
        if length > 0:
            arm     = arm / length

        # kaneeun modified : Only detecting fore-arm waving
        state           = forearm[0] * forearm[1]
        
        #YS change
        delta_angle =self.prev_forehand_angle - (forearm[0] * forearm[1]) 

        if abs(delta_angle) > WAVING_DETECT_THRESH:
            self.prev_forehand_angle    = forearm[0] * forearm[1]

        else:
            return None

        # Threshold = Not should be 90(Value == 0) degree but should be
        # inclined to right side(Value > 0).
        if delta_angle  < 0:
            state   = -1
        else:
            state   = 1

        return state

    def get_lefthandup_state(self, pose):
        result          = False
        left_shoulder   = pose[6]
        left_elbow      = pose[8]
        left_wrist      = pose[10]

        if( np.any(left_shoulder == -1) or
            np.any(left_elbow == -1) or
            np.any(left_wrist == -1)):
            result  = False
        else:
            vecArmLow   = left_wrist - left_elbow
            vecArmLow   = vecArmLow / np.linalg.norm(vecArmLow)
            vecArmUp    = left_shoulder - left_elbow
            vecArmUp    = vecArmUp / np.linalg.norm(vecArmUp)
            angle       = np.dot(vecArmLow, vecArmUp.T)

            if (    left_wrist[1] < left_shoulder[1] and
                    angle > -0.5 and
                    angle < 0.75):
                result  = True

        return result

    def reset_wave_state(self):
        self.wave_states.clear()
        self.waved = False
        return

    def record_pose(self):
        if self.saved_pose is None and self.pose_timer is None:
            self.pose_timer = time.time()

    def record_voice(self):
        """Called when starting to record the voice
        Only works if voice has not been recorded yet and
        the voice timer has not been started yet. THe voice timer gets set
        to the current time when called
        """
        if not self.voice_recorded and self.voice_timer is None:
            self.voice_timer = time.time()

    def get_timer(self):
        if self.voice_timer is not None:
            t = time.time() - self.voice_timer
            t = self.voice_wait_time - t
            return '%d' % t if t > 0 else 'R'
        if self.pose_timer is not None:
            t = time.time() - self.pose_timer
            t = self.pose_wait_time - t
            return '%d' % t
        return None

    def check_pose(self, pose):
        pose1 = np.array(pose)
        pose2 = np.array(self.saved_pose)
        mask = np.where(pose2[:, 0] != -1) and np.where(pose2[:, 1] != -1)
        pose1 = pose1[mask]
        pose2 = pose2[mask]

        # estimateAffinePartial2D(pose1, pose2)
        M, x = cv2.estimateAffine2D(pose1, pose2)
        new_pose1 = cv2.transform(pose1[np.newaxis], M)[0]

        try:
            if np.any(np.isnan(new_pose1)) or np.any(np.isnan(pose2)):
                dist = np.inf
            else:
                dist = (new_pose1 - pose2) ** 2
                dist = np.sum(dist, axis=1) ** 0.5
                dist = np.mean(dist)
        except BaseException:
            dist = np.inf

        self.pose_dist = dist

        if dist < self.pose_thresh:
            self.pose_locked = False

        return self.pose_locked

    def check_voice(self, feat):
        assert self.saved_voice is not None, 'Ooops!'
        dist = distance.cosine(self.saved_voice, feat)

        self.voice_dist = dist

        if dist < self.voice_thresh:
            self.voice_locked   = False

        # Voice is tested
        self.voice_checked  = True

        return
    
    def unlock_voice(self):
        self.voice_locked = False
        self.voice_checked = True
        self.voice_timer = None
        self.voice_recorded = True
