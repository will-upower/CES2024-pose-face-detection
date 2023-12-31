try:
    from src.fwdnxt import FWDNXT
except BaseException:
    print("FWDNXT Module not loaded")
import argparse
from pathlib import Path
from src.db import DB
from src.face import FaceDetector, FaceIdentifier
# kaneeun fly catch game
from src.flygame import FlyCatchGame
from src.gui import GUI
from src.pose import PoseEstimator
from src.proc_queue import ProcQueue
from src.speech import SpeechIdentifier
from src.tracker import Sort


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid', default='0', type=str,
                        help='Camera ID or Video File')
    parser.add_argument(
        '--width',
        default=1920,
        type=int,
        help='Video Capture Width')
    parser.add_argument(
        '--height',
        default=1080,
        type=int,
        help='Video Capture Height')
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Run in simple mode. No faceid or voice')
    parser.add_argument(
        '--max-people',
        default=1000,
        type=int,
        help='Max number of entries in DB')
    parser.add_argument(
        '--face-thresh',
        default=0.5,
        type=float,
        help='Threshold for face matching in DB')
    parser.add_argument(
        '--hm-thresh',
        default=0.3,
        type=float,
        help='Threshold to control keypoint extraction from heatmap')
    parser.add_argument(
        '--jm-thresh',
        default=0.9,
        type=float,
        help='Threshold to control keypoint extraction from jointmap')
    parser.add_argument(
        '--device',
        default='fie',
        choices=[
            'cpu',
            'gpu',
            'fie'],
        help='Where to run models')
    parser.add_argument(
        '--db-dir',
        default=Path('./db'),
        type=Path,
        help='directory to store database files')
    parser.add_argument(
        '--onnx-dir',
        default=Path('./onnx'),
        type=Path,
        help='directory to store generated onnx files')
    parser.add_argument(
        '--bin-dir',
        default=Path('./bin'),
        type=Path,
        help='directory to store generated bin files')
    parser.add_argument(
        '--pose-weights',
        default=Path('./pretrained/pose_model.pth'),
        type=Path,
        help='Weights for Pose Estimation Model')
    parser.add_argument(
        '--face-weights',
        default=Path('./pretrained/face_model.pth'),
        type=Path,
        help='Weights for Face Identification Model')
    parser.add_argument(
        '--voice-weights',
        default=Path('./pretrained/voice_model.pth'),
        type=Path,
        help='Weights for Voice Identification Model')
    parser.add_argument(
        '--clache_enable',
        action='store_true',
        help='Add histogram equalization to improve the contrast of image')
    parser.add_argument(
        '--cam_dist',
        action='store_true',
        help='Remove distortion in camera')
    parser.add_argument(
        '--bitfile',
        default='./bitfiles/AC511_prog_non_lin_4_cluster.tgz')
    parser.add_argument('--load', action='store_true', help='Load bitfile')
    parser.add_argument(
        '--ocl',
        action='store_true',
        help='Enable OpenCL Optimization')
    parser.add_argument(
        '--window',
        default=30,
        type=int,
        help='Center crop window size as a percentage of frame')
    parser.add_argument(
        '--disable-speech',
        action='store_true',
        help="Disable Voice Detection")
    parser.add_argument(
        '--enable-fly',
        action='store_true',
        help="Enable the fly game")
    parser.add_argument(
        '--faceretry-size',
        default=0.2,
        type=float,
        help='Face ID retry condition (Size change)')
    parser.add_argument(
        '--faceretry-pos',
        default=50,
        type=int,
        help='Face ID retry condition (Position change in pixel)')
    parser.add_argument(
        '--faceretry-maxc',
        default=5,
        type=int,
        help='Face ID retry condition (Maximum retry count)')
    parser.add_argument(
        '--disable-retryfaceidbyforce',
        action='store_true',
        help='Disable face id retry by gesture')
    parser.add_argument(
        '--oclFD',
        action='store_true',
        help='Enable OpenCL aided face detection')

    args = parser.parse_args()

    # Database
    db                  = DB(args)

    # Inter-thread data queue
    proc_queue          = ProcQueue()

    # Pose estimation model
    pose_estimator      = PoseEstimator(args)

    # Face detector model
    face_detector       = FaceDetector(args)

    # Face identifier
    face_identifier     = FaceIdentifier(   args,
                                            pose_estimator,
                                            pose_estimator.ie_addroff,
                                            pose_estimator.ie_handle)

    # Voice identifier
    voice_identifier    = SpeechIdentifier( args,
                                            pose_estimator,
                                            face_identifier.ie_addroff,
                                            face_identifier.ie_handle)

    # Tracker
    tracker             = Sort(max_age=15 if args.device == 'fie' else 30, min_hits=1, max_people=args.max_people)

    # Game
    flyGame             = FlyCatchGame()

    # GUI
    gui                 = GUI(              args, 
                                            db, 
                                            pose_estimator, 
                                            face_detector,
                                            face_identifier, 
                                            voice_identifier, 
                                            tracker, 
                                            proc_queue,
                                            flyGame)
    gui()


if __name__ == "__main__":
    main()
