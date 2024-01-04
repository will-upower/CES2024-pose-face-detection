# Continental Demo for Frankfurt Auto Show.



This application demonstrates the ability to run multiple network on the Inference Engine side-by-side. Runs pose estimation, face identification and voice identification to emulate a secret pose + voice based locking system. 

# Version Tracking Information
Please see this document for tracking features
https://continental.sharepoint.com/teams/team_10003507/_layouts/15/Doc.aspx?OR=teams&action=edit&sourcedoc={6533B310-6FC5-4F41-90EF-A700D176ECDE}

## How clone sources
This repository uses git LFS support, in order to install, please follow the instructions [here](https://github.com/git-lfs/git-lfs/wiki/Installation)

**It is important to note that after installing git-lfs you must run the command 'git lfs install' otherwise clone will only clone the lfs pointer files**

**The application behaves in the following way:**

1. User walks in, application uses pose estimation network to find people and detect faces. Users are initially assigned a green bounding box, 'anonymous' id and are in a locked state for both voice and pose.
2. User waves at the camera using their right <u>forearm</u> (*a wave is currently programmed to be only recognized if the elbow is above the shoulder level*), application starts data recording process for the waved user. If multiple users wave at the same time, they will be processed one by one. User being recorded is assigned a red bounding box, an id and continue to be in a locked state. Users waiting to be recorded are assigned a blue bounding box instead and continue to stay anonymous till they start being recorded. 
3. At the beginning of the recording process, after the first face feature is extracted, the user is checked against the database to find potential matches. 
   1. If the user already exists in the database, they are given 5 seconds before recording audio for 5 seconds. If the audio matches the entry in database for that user, their audio lock is set to an unlocked state. The pose is continuously matched every frame with the pose stored in database till the pose matches, after which the pose lock is set to an unlocked state. 
   2. If the user wasn't found in the database, application first records and extracts features for three angles of their face. Application then proceeds to wait for 5 seconds before recording audio for 5 seconds. Finally, the application waits for another 5 seconds before recording the pose. After the recording process is over the user is reverted to a green bounding box, has an id and is in the unlocked state. 
4. If no user has waved at the camera, application tries to identify a random individual on the frame by checking their face against the database.



## Setup

1. Setup compiler and PICO framework as described in https://github.com/FWDNXT/sdk
2. ```pip3 install -r requirements.txt```
3. ```cd tracker; python3 iou.py```



## Running the Demo

```
usage: main.py [-h] [--vid VID] [--width WIDTH] [--height HEIGHT] [--simple]
               [--max-people MAX_PEOPLE] [--face-thresh FACE_THRESH]
               [--hm-thresh HM_THRESH] [--jm-thresh JM_THRESH]
               [--device {cpu,gpu,fie}] [--db-dir DB_DIR]
               [--onnx-dir ONNX_DIR] [--bin-dir BIN_DIR]
               [--pose-weights POSE_WEIGHTS] [--face-weights FACE_WEIGHTS]
               [--voice-weights VOICE_WEIGHTS] [--bitfile BITFILE] [--load]

optional arguments:
  -h, --help            show this help message and exit
  --vid VID             Camera ID or Video File
  --width WIDTH         Video Capture Width
  --height HEIGHT       Video Capture Height
  --simple              Run in simple mode. No faceid or voice
  --max-people MAX_PEOPLE
                        Max number of entries in DB
  --face-thresh FACE_THRESH
                        Threshold for face matching in DB
  --hm-thresh HM_THRESH
                        Threshold to control keypoint extraction from heatmap
  --jm-thresh JM_THRESH
                        Threshold to control keypoint extraction from jointmap
  --device {cpu,gpu,fie}
                        Where to run models
  --db-dir DB_DIR       directory to store database files
  --onnx-dir ONNX_DIR   directory to store generated onnx files
  --bin-dir BIN_DIR     directory to store generated bin files
  --pose-weights POSE_WEIGHTS
                        Weights for Pose Estimation Model
  --face-weights FACE_WEIGHTS
                        Weights for Face Identification Model
  --voice-weights VOICE_WEIGHTS
                        Weights for Voice Identification Model
  --bitfile BITFILE
  --load                Load bitfile

```



## Design

1. The overall application is made of 5 threads : Camera, Audio, Render, Command Line and Application
2. The application uses a set of numpy memmap files as DB. This DB is backed up every time the application is run (in case of any catastrophic failure).
3. Camera thread runs in a loop, grabs frame from the camera device, and runs the preprocessing steps on the frame. Application thread can acquire lock and grab latest frame whenever ready. This minimizes some of the camera read time and preprocessing time.
4. Audio thread runs in a loop, maintains a buffer of 5 sec audio data. When data is requested, stops recording audio, generates spectrogram and waits till the spectrogram is read by the application thread and the resumes populating the audio buffer.
5. Command Line Interface (CLI) thread waits for user to interact to with the terminal, processes the request (name user, list user or delete user) in a loop.
6. Render thread waits for the application thread to push frames and data, renders and displays them. <u>It is assumed the render thread can finish one cycle faster than the application thread</u>. 
7. Application thread runs in a loop, reads frame from the camera, runs pose estimation, extracts bounding box from the pose estimation output and updates tracker state. Then proceeds to check if any of the tracks in the frame want to be recorded and if not, tries to identify people in the frame and assign them an id. If someone wants to be recorded, the application thread runs the recording process till either the user is done recording or leaves the frame.



## Software Flow

![](/home/dawood/Workspace/fwdnxt/conti_demo/docs/program_flow_misc_threads.png)

------

![](/home/dawood/Workspace/fwdnxt/conti_demo/docs/program_flow_main_thread_part1.png)

![](/home/dawood/Workspace/fwdnxt/conti_demo/docs/program_flow_main_thread_part2.png)
