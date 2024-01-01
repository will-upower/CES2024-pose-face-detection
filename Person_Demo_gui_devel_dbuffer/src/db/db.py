import shutil
import time
from threading import Lock
import numpy as np

# Number of feature vectors per person
NUM_FEATURE_VEC = 7

class DB:

    def __init__(self, args):
        self._lock = Lock()
        self.face_thresh = args.face_thresh

        def get_mmap(fname, dtype, shape, init_val=0):
            f = args.db_dir / fname
            init = False if f.is_file() else True
            mode = 'r+' if f.is_file() else 'w+'

            if not init:
                shutil.copy(f.as_posix(), f.as_posix() + '.bak')

            f = np.memmap(f.as_posix(), dtype=dtype, mode=mode, shape=shape)

            if init:
                f[:] = init_val

            return f

        self.name_map = get_mmap(
            'name_map.dat', 'S10', (args.max_people,), '')
        self.face_data = get_mmap(
            'face_data.dat', np.float32, (args.max_people, NUM_FEATURE_VEC, 512))
        self.face_mask = get_mmap(
            # 'face_mask.dat', np.bool, (args.max_people, NUM_FEATURE_VEC))
            'face_mask.dat', bool, (args.max_people, NUM_FEATURE_VEC))
        self.pose_data = get_mmap(
            'pose_data.dat', np.float32, (args.max_people, 17, 2), -1)
        self.voice_data = get_mmap(
            'voice_data.dat', np.float32, (args.max_people, 512))
        self.voice_mask = get_mmap(
            # 'voice_mask.dat', np.bool, (args.max_people,))
            'voice_mask.dat', bool, (args.max_people,))
        self.last_time = get_mmap(
            'last_time.dat', np.float32, (args.max_people,))

        for i in range(args.max_people):
            self.cleanup(i)

    def lock(self):
        return self._lock.acquire()

    def unlock(self):
        return self._lock.release()

    def find_face(self, feat):
        # Find index of all valid embeddings
        used_slots = np.where(self.face_mask.sum(1) == NUM_FEATURE_VEC)[0]

        if len(used_slots) > 0:
            # Get all available embeddings
            vectors = self.face_data[used_slots]

            # Compute cosine distance
            dist        = np.dot(feat, vectors.reshape(-1, 512).T)
            # Bug fix : uid ignorance bug fix
            dist        = np.reshape(dist, (-1, NUM_FEATURE_VEC))
            sorted_dist = np.max(dist, axis = 1)

            # Create a sorted list of indicies based on max value
            ind = np.argsort(sorted_dist)

            # Decending order
            ind = ind[::-1]

            uids = []
            confs = []

            for i in ind:
                # If the ID threshold is great enough, place in the list
                uids.append(used_slots[i])
                confs.append(sorted_dist[i])

            return uids, confs

        # No items in DB or no match found
        return None, 0

    def add_face(self, uid, feat):
        # If uid is None, find free slot
        if uid is None:
            free_list = np.where(self.face_mask.sum(1) == 0)[0]

            # If no free slot, eject oldest slot
            if len(free_list) == 0:
                free_list = [np.argmin(self.last_time)]

            uid = free_list[0]
            self.delete_entry(uid)

        # Find empty face feature slot.
        for i, mask in enumerate(self.face_mask[uid]):
            if mask == 0:
                break

        if mask == 0:
            self.face_data[uid, i] = np.array(feat)
            self.face_mask[uid, i] = 1
            self.last_time[uid] = time.time()

        return uid

    def number_of_face_db(self):
        used_slots = np.where(self.face_mask.sum(1) == NUM_FEATURE_VEC)[0]
        return np.size(used_slots)

    def cleanup(self, uid):
        # Delete if entry is incomplete
        incomplete = False
        if (np.any(self.face_mask[uid] == 0) or
            np.all(self.pose_data[uid] == -1) or
                np.any(self.voice_mask[uid] == 0)):
            incomplete = True
        if incomplete:
            self.delete_entry(uid)
        return

    def delete_entry(self, uid):
        try:
            uid = int(uid)
        except BaseException:
            return

        if uid < 0 or uid > len(self.name_map):
            return
        self.name_map[uid] = ''
        self.face_data[uid][:] = 0
        self.face_mask[uid][:] = 0
        self.pose_data[uid][:] = -1
        self.voice_data[uid][:] = 0
        self.voice_mask[uid] = 0
        self.last_time[uid] = 0

    def get_name(self, uid):
        name = self.name_map[uid]
        if len(name) == 0:
            return '%04d' % uid
        return name.decode('utf-8')

    def get_pose(self, uid):
        pose = self.pose_data[uid]
        if np.all(pose == -1):
            pose = None
        return np.array(pose)

    def get_voice(self, uid):
        if self.voice_mask[uid] == 0:
            return None
        return np.array(self.voice_data[uid])

    def set_pose(self, uid, pose):
        self.pose_data[uid] = np.array(pose)

    def set_voice(self, uid, voice):
        self.voice_data[uid] = np.array(voice)
        self.voice_mask[uid] = 1

    def __del__(self, **kwargs):
        # Save when exiting gracefully
        self.name_map.flush()
        self.face_data.flush()
        self.face_mask.flush()
        self.pose_data.flush()
        self.voice_data.flush()
        self.voice_mask.flush()
        self.last_time.flush()
        return
