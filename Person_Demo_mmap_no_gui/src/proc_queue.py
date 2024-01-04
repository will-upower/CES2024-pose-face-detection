from queue import Queue


class ProcQueue:
    def __init__(self):
        self.queue_posemodel = Queue(maxsize=1)
        self.queue_posepostproc = Queue(maxsize=1)
        self.queue_faceD_track = Queue(maxsize=1)
        self.queue_faceID_voice = Queue(maxsize=1)
        self.queue_people = Queue(maxsize=1)
        self.queue_heatmap = Queue(maxsize=1)
