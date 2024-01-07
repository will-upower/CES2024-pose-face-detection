import time
from collections import deque
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

import numpy as np

import cv2
from randomcolor import RandomColor
from src.proc_queue import ProcQueue

from .pose import jnt_info, kpt_info
from .tracker.kalman import ObjTrkState
from .utils.camera import Camera
from .utils.cli import CLI
from .utils.mic import Mic
import psutil
import math

import pdb

import os
import mmap
mmap_inf_path = '/home/nvidia/mmap_exchange/orin_to_v4h2.dat'

user_action = {
    0: 'Turn your face left/right',
    1: 'Get ready to record',
    2: 'Please record your voice.',
    3: 'Please make your own Pose',
    4: '',
    5: 'Hello! Wave your right hand.'
}

# Face ID checking = 1 time per position
FACEID_TESTCOUNT    = 1

# Add pauses to keep buffer from bottleneck
TIMEOUT_LENGTH = 15

class GUI:

    def __init__(self, args, db, pose_estimator, face_detector,
                 face_identifier, voice_identifier, tracker, proc_queue):

        self.simple = args.simple
        self.disable_speech = args.disable_speech

        # Setup video capture
        vid = Camera(args.vid, args.cam_dist, args.width, args.height, args.ocl)
        vid.start()
        yres, xres = vid.res

        colorgen = RandomColor(1)
        jt_color = colorgen.generate( luminosity='light', count=len(jnt_info), format_='rgbArray')
        bbox_color = [0, 255, 0]
        pt_color = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                    [255, 255, 0], [170, 255, 0], [85, 255, 0],
                    [0, 255, 0], [0, 255, 85], [0, 255, 170],
                    [0, 255, 255], [0, 170, 255], [0, 85, 255],
                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
                    [255, 0, 255], [255, 0, 170], [255, 0, 85]
                    ]

        self.time_taken = deque(maxlen=10)

        self.window = args.window / 100.0
        self.vid = vid
        self.name = 'Demo'
        self.textarea_sz = (xres - yres) // 2

        self.db = db
        self.tracker = tracker
        self.pose_estimator = pose_estimator
        self.face_detector = face_detector
        self.face_identifier = face_identifier
        self.voice_identifier = voice_identifier

        self.opacity = 0.7
        self.kt_radius = int(min(xres, yres) * 0.003)
        self.jt_radius = max(int(self.kt_radius * 0.8), 1)
        self.jt_color = jt_color
        self.pt_color = pt_color
        self.bbox_color = bbox_color
        self.disable_speech = args.disable_speech

        self.last_face_run = 0
        self.proc_queue = proc_queue
        self.lock = Lock()

        # assumption is render time is smaller than pose estimation time
        self.render_queue = Queue(maxsize=1)
        self.render_worker = Thread( target=self.render_loop, args=( self.render_queue,))
        self.render_worker.start()

        self.end = False
        self.recording = None
        self.recording_uid = None
        # Proc thread
        self.worker_posemodel = Thread(target=self.thread_posemodel, args=(self.proc_queue.queue_posemodel,))
        self.worker_posepost = Thread(target=self.thread_posepostproc, args=(self.proc_queue.queue_posepostproc,))
        self.worker_faceDtrack = Thread(target=self.thread_faced_tracking, args=(self.proc_queue.queue_faceD_track,))
        self.worker_faceID_voice_overlay = Thread(target=self.thread_faceID_voice, args=(self.proc_queue.queue_faceID_voice,))

        self.faceIDRetrySize    = args.faceretry_size
        self.faceIDRetryPos     = args.faceretry_pos
        self.faceIDRetryMaxC    = args.faceretry_maxc
        self.enFaceIDRetryByG   = not args.disable_retryfaceidbyforce

    # @profile
    def __call__(self):
        self.worker_posemodel.start()
        self.worker_posepost.start()
        self.worker_faceDtrack.start()
        self.worker_faceID_voice_overlay.start()

        while not self.end:
            ret, frame, frame_rgb, pose_frame = self.vid.read()
            if not ret:
                break

            try:
                self.proc_queue.queue_posemodel.put(
                    (frame, frame_rgb, pose_frame), timeout=TIMEOUT_LENGTH)
            except Exception as e:
                if self.end:
                    pass
                else:
                    print(e)
                    exit(-1)

            self.last_face_run -= 1

        self.vid.stop()
        self.render_worker.join()
        cv2.destroyAllWindows()

    def thread_posemodel(self, q1):
        while not self.end:
            frame, frame_rgb, pose_frame = q1.get()
            self.lock.acquire()
            hms, jms = self.pose_estimator.pose_model_fnc(pose_frame)
            self.lock.release()
            # HW Accelerator owner change : Face id, voice thread
            try:
                self.proc_queue.queue_posepostproc.put(
                    (frame, frame_rgb, hms, jms), timeout=TIMEOUT_LENGTH)
            except BaseException:
                if self.end:
                    pass

    def thread_posepostproc(self, q):
        people_prev = None
        while not self.end:
            frame, frame_rgb, hms, jms = q.get()
            ih, iw = frame.shape[:2]
            people, heatmap = self.pose_estimator.pose_postproc_fnc( ih, iw, hms, jms, people_prev)
            people_prev = people
            try:
                self.proc_queue.queue_faceD_track.put( (frame, frame_rgb, people, heatmap), timeout=TIMEOUT_LENGTH)
            except BaseException:
                if self.end:
                    pass

    def thread_faced_tracking(self, q):
        while not self.end:
            frame, frame_rgb, people, heatmap = q.get()
            faces, people = self.face_detector.get_faces_from_pose(
                frame_rgb, people)
            tracks = self.tracker.update(faces, self.recording)
            try:
                self.proc_queue.queue_faceID_voice.put(
                    (frame, tracks, people, heatmap), timeout=TIMEOUT_LENGTH)
            except Exception as e:
                if self.end:
                    pass

    def thread_faceID_voice(self, q1):
        start_time = time.time()
        while not self.end:
            frame, tracks, people, heatmap = q1.get()
            self.faceID_voice_overlay(frame, tracks, people, heatmap)
            self.time_taken.append(time.time() - start_time)
            start_time = time.time()
            # HW Accelerator owner change : Pose model thread

    def faceID_voice_overlay(self, frame, tracks, people, heatmap):
        if not self.simple:
            # variable for tracking number of users in DB
            self.numOfUsers     = self.db.number_of_face_db()
            # If recording, make sure that track exists
            if self.recording is not None:
                for trk in tracks:
                    if trk.id == self.recording:
                        break
                else:
                    if self.recording_uid is not None:
                        self.db.lock()
                        self.db.cleanup(self.recording_uid)
                        self.db.unlock()
                    self.recording_uid = None
                    self.recording = None

            # Check if any track is ready to be recorded
            if self.recording is None:
                for trk in tracks:
                    if trk.waved and trk.set_recording(True):
                        self.recording = trk.id

                        break

            if self.recording:
                self.db.lock()

                # Find track that is being recorded
                for trk in tracks:
                    if trk.id == self.recording:
                        break
                else:
                    # Recording should always match an existing track
                    raise ValueError

                # Get current state of track
                bbox, direction, aligned = trk.get_state()

                # Check if I need to run Face ID for this angle
                run_faceid = False
                # Record only frontal face.
                if direction == 1:
                    run_faceid = True

                # Condition of face ID execution
                # 1. Just waved right hand
                # 2. Catched fly
                # 3. 5 frames have passed since last face ID execution.
                # 4. Feature vectors for person are not all occupied.
                run_faceid = run_faceid and (self.last_face_run < 1) and np.any(
                    self.db.face_mask[trk.uid] == 0)


                if trk.uid is None:
                    assert trk.complete_feats == False, 'Logic Error'

                if trk.complete_feats:
                    if not self.disable_speech and (
                            trk.saved_voice is None or trk.run_voice):
                        trk.record_voice()
                        if((time.time() - trk.voice_timer) >= 0 and (time.time() - trk.voice_timer) < 5):
                            trk.instruction_state = ObjTrkState.VOICE_COUNT
                        else:
                            trk.instruction_state = ObjTrkState.VOICE_REC

                        # + 5 because of record time
                        if (time.time() - trk.voice_timer) >= (trk.voice_wait_time + 5):
                            if self.mic.data_available():
                                trk.voice_timer = None
                                trk.voice_recorded = True
                                spec = self.mic.read()
                                self.lock.acquire()
                                feat = self.voice_identifier(spec)
                                self.lock.release()
                                if trk.saved_voice is None:
                                    trk.saved_voice = feat
                                    self.db.set_voice(trk.uid, trk.saved_voice)
                                    trk.voice_locked = False
                                else:
                                    trk.check_voice(feat)
                                    trk.run_voice = False
                                trk.instruction_state = ObjTrkState.POSE_REC
                            else:
                                pass
                    elif trk.saved_pose is None:
                        trk.record_pose()
                        trk.instruction_state = ObjTrkState.POSE_REC
                        if (time.time() - trk.pose_timer) >= trk.pose_wait_time:
                            trk.pose_timer = None
                            trk.pose_locked = False
                            trk.saved_pose = np.array(trk.last_pose)
                            self.db.set_pose(trk.uid, trk.saved_pose)
                    else:
                        if self.disable_speech:
                            trk.unlock_voice()
                        self.recording = None
                        self.recording_uid = None
                        trk.set_recording(False)
                self.db.unlock()
            else:
                # find a track without id
                check_track         = self.last_face_run < 1
                needUIDRearrange    = False

                for trk in tracks:
                    if not check_track:
                        continue

                    # Face ID check condition
                    # 0. Only frontal face will be checked.
                    # 1. New track which is older than 5
                    # 2. Checked track but failed with condition
                    # 2-1. Track user id is not assigned.
                    # 2-2. Track user id is already checked but failed.
                    # 2-3. Face position is moved more than 25 pixels from failed position.
                    # 2-4. Face size is changed more than 10% since face id
                    # checking is failed.

                    bbox, direction, aligned = trk.get_state()
                    # Only for frontal face
                    if direction == 1:
                        # First condition checking
                        isNewFace           =   (   (trk.uid is None) and 
                                                    (trk.checked == False) and 
                                                    (trk.age > 5))

                        # Second condition checking
                        oldButMovedFace     = self.isFaceMoved( trk,
                                                            self.faceIDRetrySize, 
                                                            self.faceIDRetryPos, 
                                                            self.faceIDRetryMaxC)

                        forceByGesture      = self.isForcedRetry(trk, 0)

                        if (    (self.numOfUsers > 0) and
                                (   (isNewFace is True) or 
                                    (oldButMovedFace is True) or 
                                    (forceByGesture is True))):
                            self.last_face_run = 5

                            # Get feature vector for the current face
                            self.lock.acquire()
                            feat = self.face_identifier(aligned)
                            self.lock.release()
                            uids, confs = self.db.find_face(feat)

                            # if DB is empty
                            if uids is None:
                                pass
                            else:
                                # For first face id test, initialize conf value.
                                if trk.faceIDTestCount == 0:
                                    trk.faceIDTestValue[:]  = 0

                                trk.faceIDTestCount += 1

                                for i, confVal in enumerate(confs):
                                    trk.faceIDTestValue[uids[i]]    += confVal
                                
                                # Face ID check >= FACEID_TESTCOUNT times
                                if trk.faceIDTestCount >= FACEID_TESTCOUNT:
                                    trk.faceIDTestCount = 0
                                    # Make averages for all uids
                                    trk.faceIDTestValue = trk.faceIDTestValue / FACEID_TESTCOUNT
                                    trk.faceIDConfOrder = np.argsort(trk.faceIDTestValue)
                                    # Descending order
                                    trk.faceIDConfOrder = trk.faceIDConfOrder[::-1]
                                    trk.faceConfOrderI  = 0
                                    confMaxIdx          = trk.faceIDConfOrder[trk.faceConfOrderI]
                                    conf                = trk.faceIDTestValue[confMaxIdx]
                                    if trk.faceIDTestValue[confMaxIdx] > self.db.face_thresh:
                                        uid             = confMaxIdx
                                    else:
                                        # Face ID search failed.
                                        uid = None
                                        # For second chance, save position and size of face
                                        # when ID check is failed.
                                        x1, y1, x2, y2          = bbox
                                        trk.lastFacdPos         = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                        trk.lastFaceSize        = (x2 - x1) * (y2 - y1)
                                        trk.numFaceIDCheck      += 1                                        

                                    # If person exists, populate track
                                    # information
                                    if uid is not None:
                                        trk.uid             = uid
                                        trk.conf            = conf
                                        trk.saved_pose      = self.db.get_pose(uid)
                                        trk.saved_voice     = self.db.get_voice(uid)
                                        trk.complete_feats  = True
                                        trk.run_voice       = True

                                        # Voice is not tested yet.
                                        trk.voice_checked   = False
                                        trk.voice_dist      = 0

                                        needUIDRearrange    = True

                                        assert np.all(self.db.face_mask[uid] == 1)
                                    else:
                                        trk.conf = conf

                                    trk.checked = True
                                    break

                # Update uids : Compare uid/confidence with other tracks
                if needUIDRearrange is True:
                    self.rearrange_track_uids(tracks)
        try:
            self.render_queue.put((np.array(frame), np.array(people), tracks, heatmap), timeout=TIMEOUT_LENGTH)
        except Exception as e:
            if self.end:
                pass
            else:
                # Exit application when display is not working.
                print("Rendering thread is stuck. ", e)
                self.end    = True
                exit(-1)

        
    # Check face ID retry condition by face movement.
    def isFaceMoved(self, trk, threshSize, threshDist, maxRetry):
        result  = False

        if (    (trk.uid is None) and
                (trk.checked is True) and
                (trk.numFaceIDCheck < maxRetry)):
            bbox, direction, aligned    = trk.get_state()
            x1, y1, x2, y2              = bbox
            centerX     = int((x1 + x2) / 2)
            centerY     = int((y1 + y2) / 2)
            faceSize    = (x2 - x1) * (y2 - y1)
            centerDist  = math.sqrt(pow(centerX - trk.lastFacdPos[0], 2) +
                                        pow(centerY - trk.lastFacdPos[1], 2))
            sizeDiff    = abs(faceSize - trk.lastFaceSize) / trk.lastFaceSize
            # Size change or position change of face is bigger than threshold
            # than retry face id.
            result      = (centerDist >= threshDist) or (sizeDiff >= threshSize)

        return result

    # Check forced face ID retry condition
    def isForcedRetry(self, trk, maxRetry):
        result  = False

        if (    (self.enFaceIDRetryByG is True) and
                (trk.uid is None) and
                (trk.checked is True) and
                ((maxRetry == 0) or (trk.numFaceIDCheck < maxRetry)) and
                (trk.faceIDRetryByG is True)):
            result              = True
            trk.faceIDRetryByG  = False

        return result

    # Re-arrange uid in conjuction with confidence value
    def rearrange_track_uids(self, tracks):

        isUpdated = False

        while isUpdated is True:
            isUpdated = False

            for i, trk_i in enumerate(tracks):
                if trk_i.uid is None:
                    continue
                for j, trk_j in enumerate(tracks):
                    if i == j or trk_j.uid is None:
                        continue
                    else:
                        if trk_i.uid == trk_j.uid:
                            isUpdated = True
                            # If I's confidence is bigger than j's confidence
                            if trk_i.conf >= trk_j.uid:
                                # Update j's uid
                                # If there is no more candidate
                                if trk_j.faceConfOrderI >= (
                                        len(trk_j.faceIDConfOrder) - 1):
                                    # This track shall be "Unknown"
                                    trk_j.uid = None
                                    trk_j.conf = 0
                                # Else, update to next confidence uid
                                else:
                                    trk_j.faceConfOrderI += 1
                                    uid = trk_j.faceIDConfOrder[trk_j.faceConfOrderI]
                                    conf = trk_j.faceIDTestValue[uid]
                                    if conf > self.db.face_thresh:
                                        trk_j.uid = uid
                                        trk_j.conf = conf
                                        trk_j.saved_pose = self.db.get_pose(
                                            uid)
                                        trk_j.saved_voice = self.db.get_voice(
                                            uid)
                                    else:
                                        trk_j.uid = None
                                        trk_j.conf = 0
                            # If I's confidence is smaller than j's confidence
                            else:
                                # Update i's uid
                                # If there is no more candidate
                                if trk_i.faceConfOrderI >= (
                                        len(trk_i.faceIDConfOrder) - 1):
                                    # This track shall be "Unknown"
                                    trk_i.uid = None
                                    trk_i.conf = 0
                                # Else, update to next confidence uid
                                else:
                                    trk_i.faceConfOrderI += 1
                                    uid = trk_i.faceIDConfOrder[trk_i.faceConfOrderI]
                                    conf = trk_i.faceIDTestValue[uid]
                                    if conf > self.db.face_thresh:
                                        trk_i.uid = uid
                                        trk_i.conf = conf
                                        trk_i.saved_pose = self.db.get_pose(
                                            uid)
                                        trk_i.saved_voice = self.db.get_voice(
                                            uid)
                                    else:
                                        trk_i.uid = None
                                        trk_i.conf = 0
                            break

    def render_loop(self, q):
        #cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        """
        cv2.setWindowProperty(
            self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        """
        file_inference = os.open(mmap_inf_path, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
        os.truncate(file_inference, 1280 * 720 * 3)
        mmap_file_inference = mmap.mmap(file_inference, 1280 * 720 * 3, mmap.MAP_SHARED, mmap.PROT_WRITE)

        while True:
            frame, people, tracks, heatmap = q.get()
            key = self.render_overlay(frame, people, tracks, heatmap, mmap_file_inference)
            #if key == ord('q'):
            #    self.end = True
            #    break
        #cv2.destroyAllWindows()

    def __del__(self):
        self.end = True
        self.vid.stop()

        self.render_worker.join()

    # @profile
    def render_overlay(self, frame, people, tracks, heatmap, mmap_file_inference):
        disFrame    = frame.copy()
        overlay     = disFrame.copy()

        for person in people:
            # Draw pose
            _person = person[person[:, 0] != -1]
            _person = _person[_person[:, 1] != -1]
            x1 = min(_person[:, 0])
            x2 = max(_person[:, 0])
            y1 = min(_person[:, 1])
            y2 = max(_person[:, 1])

            for idx, (i, j) in enumerate(jnt_info):
                x1, y1, x2, y2 = None, None, None, None
                if (person[i - 1] != -1).all():
                    x1, y1 = person[i - 1]
                    cv2.circle(overlay, (x1, y1), self.kt_radius,
                               self.pt_color[i - 1], -1)
                if (person[j - 1] != -1).all():
                    x2, y2 = person[j - 1]
                    cv2.circle(overlay, (x2, y2), self.kt_radius,
                               self.pt_color[j - 1], -1)
                if x1 and x2 and y1 and y2:
                    cv2.line(overlay, (x1, y1), (x2, y2),
                             self.jt_color[idx], self.jt_radius, cv2.LINE_AA)

        instMsgIndex = 6
        for trk in tracks:
            if trk.instruction_state.value < instMsgIndex:
                instMsgIndex = trk.instruction_state.value

            if not trk.visible:
                continue
            # Draw face
            bbox, direction, *_ = trk.get_state()

            if trk.uid is not None and trk.uid >= 0:
                name = self.db.get_name(trk.uid)  # fine to be not thread safe
                trk_id = '%s' % name
            else:
                trk_id = 'anon' if not trk.checked else 'unknown'

            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2),
                          self.bbox_color, 2, cv2.LINE_AA)

            if self.simple:
                fill_color = (0, 255, 0)
            elif trk.waved:
                fill_color = (255, 0, 0)
            elif trk.recording:
                fill_color = (0, 0, 255)
            else:
                fill_color = (0, 255, 0)

            cv2.rectangle(disFrame, (x1, y1), (x2, y2), fill_color, -1, cv2.LINE_AA)

            w = (x2 - x1) // 3
            sx = x1
            kpt_radius = w // 3

            anglebar = [0, 0, 0]

            if direction >= 0:
                anglebar[direction] = 1

            for i, val in enumerate(anglebar):
                color = (0, 255, 0) if val == 1 else (255, 255, 255)

                if direction == -1:
                    color = (0, 0, 255)

                cv2.rectangle(
                    overlay, (sx, y2), (sx + w, y2 + kpt_radius), color, 2, cv2.LINE_AA)

                sx += w

            cv2.rectangle(overlay, (sx, y2), (x2 + 1, y2 +
                                              kpt_radius), color, 2, cv2.LINE_AA)
            cv2.rectangle(overlay, (sx, y2), (x2 + 1, y2 +
                                              kpt_radius), color, -1, cv2.LINE_AA)

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            radius = int(max((x2 - x1), (y2 - y1)) // 2)

            timer = trk.get_timer()

            if timer is not None:
                font_scale = 3
                font_thickness = 2
                font = cv2.FONT_HERSHEY_DUPLEX
                text_size, baseline = cv2.getTextSize(
                    timer, font, font_scale, font_thickness)
                x, y = int(cx - text_size[0] //
                           2), int(cy + text_size[1] // 2)
                cv2.putText(overlay, timer, (x, y), font, font_scale,
                            (255, 255, 255), font_thickness, cv2.LINE_AA)

            if not self.simple:
                font_scale = 1.2
                font_thickness = 2
                font = cv2.FONT_HERSHEY_DUPLEX
                text_size, baseline = cv2.getTextSize(
                    trk_id, font, font_scale, font_thickness)

                x1, y1 = x1, cy - radius - baseline - 5
                cv2.rectangle(overlay, (x1, y1 + baseline),
                              (x1 + text_size[0], y1 - text_size[1]), (0, 0, 0), -1)
                cv2.putText(overlay, trk_id, (x1, y1), font, font_scale,
                            self.bbox_color, font_thickness, cv2.LINE_AA)

                # Face ID conf, Voice/Pose status display
                font_scale = 0.7
                font_thickness = 1
                y1 += 30

                # Face ID confidence display
                font_scale = 0.8
                font_thickness = 1
                faceIDConf = "Face ID Conf : {:0.2f}".format(
                    np.float32(trk.conf))
                # If confidence is lower than 0.5, text color is red.
                # Otherwise, green.
                if trk.conf <= 0.5:
                    txtColor = [0, 0, 255]
                else:
                    txtColor = [0, 255, 0]

                cv2.putText(overlay, faceIDConf, (x2, y1), font, font_scale,
                            txtColor, font_thickness, cv2.LINE_AA)

                # Pose/Voice status text display
                y1 += 20
                if trk.voice_locked:
                    if trk.saved_voice is None:
                        voiceMsg = "Voice is not recorded."
                    else:
                        if trk.voice_checked == False:
                            voiceMsg = "Voice check ready"
                        else:
                            voiceMsg = "Voice ID NG : {:0.2f}/{:0.2f}".format(
                                trk.voice_dist, trk.voice_thresh)

                    cv2.putText(overlay, voiceMsg, (x2, y1), font, font_scale,
                                [0, 0, 255], font_thickness, cv2.LINE_AA)
                else:
                    if trk.voice_dist == 0:
                        voiceMsg = "Voice is recorded."
                    else:
                        voiceMsg = "Voice ID Pass : {:0.2f}/{:0.2f}".format(
                            trk.voice_dist, trk.voice_thresh)
                    cv2.putText(overlay, voiceMsg, (x2, y1), font, font_scale,
                                [0, 255, 0], font_thickness, cv2.LINE_AA)

                y1 += 20
                if trk.pose_locked:
                    if trk.saved_pose is None:
                        poseMsg = "Pose is not captured."
                    else:
                        poseMsg = "Pose check ready"
                    cv2.putText(overlay, poseMsg, (x2, y1), font, font_scale,
                                [0, 0, 255], font_thickness, cv2.LINE_AA)
                else:
                    poseMsg = "Pose ID Pass"
                    cv2.putText(overlay, poseMsg, (x2, y1), font, font_scale,
                                [0, 255, 0], font_thickness, cv2.LINE_AA)

        if len(self.time_taken) > 0:
            time_taken = np.mean(self.time_taken)
            fps = 1 / time_taken if time_taken > 0 else 0
        else:
            fps = 0

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        font_thickness = 2

        font_scale = 1.0
        infoOffset = 800

        # FPS
        fps = "FPS          : {:02d}".format(int(fps))
        cv2.putText(overlay, fps, (50, infoOffset), font, font_scale,
                    self.bbox_color, font_thickness, cv2.LINE_AA)
        infoOffset += 40

        # Add number of users in DB
        user_in_db = "People in DB : {:03d}".format(self.numOfUsers)
        cv2.putText(overlay, user_in_db, (50, infoOffset), font,
                    font_scale, self.bbox_color, font_thickness, cv2.LINE_AA)
        infoOffset += 40

        # CPU occupancy
        cpuOccup = "CPU          : {:03.2f} %".format(psutil.cpu_percent())
        cv2.putText(overlay, cpuOccup, (50, infoOffset), font,
                    font_scale, self.bbox_color, font_thickness, cv2.LINE_AA)
        infoOffset += 40

        # Pose model execution time
        poseMTime = "Pose model  : {:04.2f} ms".format(
            self.pose_estimator.time_model * 1000)
        cv2.putText(overlay, poseMTime, (50, infoOffset), font,
                    font_scale, self.bbox_color, font_thickness, cv2.LINE_AA)
        infoOffset += 40

        # Face model execution time
        faceIDMTime = "Face model  : {:04.2f} ms".format(
            self.face_identifier.time_model * 1000)
        cv2.putText(overlay, faceIDMTime, (50, infoOffset), font,
                    font_scale, self.bbox_color, font_thickness, cv2.LINE_AA)
        infoOffset += 40

        # Face model execution time
        if not self.disable_speech:
            voiceMTime = "Voice model : {:04.2f} ms".format(
                self.voice_identifier.time_model * 1000)
            cv2.putText(
                overlay,
                voiceMTime,
                (50, infoOffset),
                font,
                font_scale,
                self.bbox_color,
                font_thickness,
                cv2.LINE_AA)

        disFrame    = cv2.addWeighted(  overlay,
                                    self.opacity,
                                    disFrame,
                                    1 - self.opacity,
                                    0)

        # add heatmap
        disFrame    = cv2.addWeighted(disFrame, 1, heatmap, .8, 0)       

        # write file to mmap buffer
        mmap_file_inference.seek(0)
        mmap_file_inference.write(disFrame.data)

        # debug output for display connected to orin
        cv2.imshow(self.name, disFrame)

        return cv2.waitKey(5)
