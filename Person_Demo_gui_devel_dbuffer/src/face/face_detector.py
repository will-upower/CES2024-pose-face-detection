import time
from enum import IntEnum
import numpy as np
import cv2

class Face(IntEnum):
    '''
    Parts we care for face detection from pose
    '''
    nose = 0
    leye = 1
    reye = 2
    lear = 3
    rear = 4
    lshoulder = 5
    rshoulder = 6


class FaceDetector:
    '''
    Detect face bbox from pose kpts.
    Can be extended to use custom detector if needed.
    '''

    def __init__(self, args):
        # kaneeun adaptive hist-eq test
        self.clache         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.clache_enable  = args.clache_enable
        
        self.faceDByCV      = (args.ocl and args.oclFD)

        if self.faceDByCV is True:
            cv2.ocl.setUseOpenCL(True)
            self.face_cascade   = cv2.CascadeClassifier('./src/face/haarcascade_frontalface_default.xml')
        pass

    def get_faces_from_pose(self, frame, people):
        faces = []

        if self.faceDByCV is True:
            frameW      = frame.shape[1]
            frameH      = frame.shape[0]

        for person in people:

            found = False
            direction = 0
            angle = None

            nose = (person[Face.nose] != -1).all()
            lear = (person[Face.lear] != -1).all()
            rear = (person[Face.rear] != -1).all()
            leye = (person[Face.leye] != -1).all()
            reye = (person[Face.reye] != -1).all()
            lshoulder = (person[Face.lshoulder] != -1).all()
            rshoulder = (person[Face.rshoulder] != -1).all()

            if nose and lear and rear:
                # Front view
                x1, x2 = person[Face.lear][0], person[Face.rear][0]
                if x1 > x2:
                    x1, x2 = x2, x1

                cx = (x1 + x2) / 2
                cy = person[Face.nose][1]

                wd = np.linalg.norm(person[Face.lear] - person[Face.rear])

                v = person[Face.rear] - person[Face.lear]
                u = (1, 0)
                dot = v[0] * u[0] + v[1] * u[1]
                det = v[0] * u[1] - v[1] * u[0]
                rad = np.arctan2(det, dot)
                angle = 180 - np.degrees(rad)

                x1 = cx - (wd / 2)
                x2 = cx + (wd / 2)

                y1 = cy - (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))
                y2 = cy + (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))

                ang = 360 - angle if angle > 180 else angle
                ang = ang / 180
                y2 += wd * min(ang, 0.10)

                found = True

            elif nose and leye and rear:
                x1, x2 = person[Face.leye][0], person[Face.rear][0]
                if x1 > x2:
                    x1, x2 = x2, x1
                cx = person[Face.rear][0] + person[Face.nose][0]
                cy = person[Face.nose][1]
                cx = cx / 2

                wd = np.linalg.norm(person[Face.leye] - person[Face.rear])
                wd = wd

                v = person[Face.rear] - person[Face.leye]
                u = (1, 0)
                dot = v[0] * u[0] + v[1] * u[1]
                det = v[0] * u[1] - v[1] * u[0]
                rad = np.arctan2(det, dot)
                angle = 180 - np.degrees(rad)

                x1 = cx - (wd / 2)
                x2 = cx + (wd / 2) + wd * 0.10

                y1 = cy - (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))
                y2 = cy + (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))

                ang = 360 - angle if angle > 180 else angle
                ang = ang / 180

                y2 += wd * min(ang, 0.20)

                found = True

            elif nose and reye and lear:
                x1, x2 = person[Face.reye][0], person[Face.lear][0]
                if x1 > x2:
                    x1, x2 = x2, x1
                cx = person[Face.lear][0] + person[Face.nose][0]
                cy = person[Face.nose][1]
                cx = cx / 2

                wd = np.linalg.norm(person[Face.reye] - person[Face.lear])
                wd = wd

                v = person[Face.reye] - person[Face.lear]
                u = (1, 0)
                dot = v[0] * u[0] + v[1] * u[1]
                det = v[0] * u[1] - v[1] * u[0]
                rad = np.arctan2(det, dot)
                angle = 180 - np.degrees(rad)

                x1 = cx - (wd / 2) - wd * 0.10
                x2 = cx + (wd / 2)

                y1 = cy - (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))
                y2 = cy + (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))

                ang = 360 - angle if angle > 180 else angle
                ang = ang / 180

                y2 += wd * min(ang, 0.20)
                found = True

            elif nose and reye and rear:
                x1, x2 = person[Face.nose][0], person[Face.rear][0]
                if x1 > x2:
                    x1, x2 = x2, x1
                cx = (x1 + x2) / 2
                cy = person[Face.nose][1]

                v = person[Face.rear] - person[Face.reye]
                u = (1, 0)
                dot = v[0] * u[0] + v[1] * u[1]
                det = v[0] * u[1] - v[1] * u[0]
                rad = np.arctan2(det, dot)
                angle = 180 - np.degrees(rad)

                wd = np.linalg.norm(person[Face.rear] - person[Face.nose])

                x1 = cx - (wd / 2)
                x2 = cx + (wd / 2) + wd * 0.05

                wd = wd * 1.2
                y1 = cy - (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))
                y2 = cy + (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))

                ang = 360 - angle if angle > 180 else angle
                ang = ang / 180

                y2 += wd * min(ang * 1.4, 0.20)

                found = True

            elif nose and leye and lear:
                x1, x2 = person[Face.nose][0], person[Face.lear][0]
                if x1 > x2:
                    x1, x2 = x2, x1
                cx = (x1 + x2) / 2
                cy = person[Face.nose][1]

                v = person[Face.leye] - person[Face.lear]
                u = (1, 0)
                dot = v[0] * u[0] + v[1] * u[1]
                det = v[0] * u[1] - v[1] * u[0]
                rad = np.arctan2(det, dot)
                angle = 180 - np.degrees(rad)

                wd = np.linalg.norm(person[Face.lear] - person[Face.nose])

                x1 = cx - (wd / 2) - wd * 0.05
                x2 = cx + (wd / 2)

                y1 = cy - (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))
                y2 = cy + (wd / 2) * max(abs(np.cos(rad)), abs(np.sin(rad)))

                ang = 360 - angle if angle > 180 else angle
                ang = ang / 180

                y2 += wd * min(ang * 1.4, 0.20)

                found = True

            if found:
                u, v = (person[Face.nose][0] - x1), (person[Face.nose][0] - x2)
                if v != 0:
                    direction = u / v
                else:
                    direction = 0
                direction = np.absolute(direction)
                if direction < 1:
                    direction = -90 * (direction - 1)
                elif direction > 1:
                    direction = 1 / direction
                    direction = 90 * (direction - 1)

                direction *= -1
                anglecut = [(-75, -30), (-30, 30), (30, 75)]
                for i in range(len(anglecut)):
                    if anglecut[i][0] < direction <= anglecut[i][1]:
                        direction = i
                        break
                else:
                    direction = -1

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if (x2 - x1) == 0 or (y2 - y1) == 0:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                sz = max(w, h)
                x1 = cx - sz // 2
                x2 = cx + sz // 2
                y1 = cy - sz // 2
                y2 = cy + sz // 2

                if self.faceDByCV is True:
                    pfW         = x2 - x1
                    pfH         = y2 - y1
                    mx1         = max(x1 - int(pfW / 2), 0)
                    mx2         = min(x2 + int(pfW / 2), frameW - 1)
                    my1         = max(y1 - int(pfH / 2), 0)
                    my2         = min(y2 + int(pfH / 2), frameH - 1)
                    subFrame    = frame[my1 : my2 + 1, mx1 : mx2 + 1, :]
                
                    uMSubFrame  = cv2.UMat(subFrame)
                    gray        = cv2.cvtColor(uMSubFrame, cv2.COLOR_BGR2GRAY)
                    gray        = cv2.resize(gray, (int(subFrame.shape[1] / 2), int(subFrame.shape[0] / 2)))
                    # Detect the faces with OpenCV OCL
                    cvFacesPos  = self.face_cascade.detectMultiScale(   gray,
                                                                        1.1,
                                                                        4,
                                                                        minSize = (int(pfW * 0.45), int(pfH * 0.45)))

                    if len(cvFacesPos) > 0:
                        cvFacesPos[:, 2]    = cvFacesPos[:,0] + cvFacesPos[:,2]
                        cvFacesPos[:, 3]    = cvFacesPos[:,1] + cvFacesPos[:,3]
                        cvFacesPos          = cvFacesPos * 2
                        cvFacesPos[:, 0]    += mx1
                        cvFacesPos[:, 2]    += mx1
                        cvFacesPos[:, 1]    += my1
                        cvFacesPos[:, 3]    += my1
                        cvFacesPosC         = np.zeros((cvFacesPos.shape[0], 2)).astype(np.uint16)
                        cvFacesPosC[:, 0]   = (cvFacesPos[:, 0] + cvFacesPos[:, 2]) / 2
                        cvFacesPosC[:, 1]   = (cvFacesPos[:, 1] + cvFacesPos[:, 3]) / 2

                        pcx                 = (x1 + x2) / 2
                        pcy                 = (y1 + y2) / 2
                        dist                = np.sqrt(pow(cvFacesPosC[:, 0] - pcx, 2) + pow(cvFacesPosC[:, 1] - pcy, 2))
                        if min(dist) < ((x2 - x1) / 2):
                            minDistI        = dist.argmin()
                            x1, y1, x2, y2  = cvFacesPos[minDistI]                        

                aligned = frame[y1:y2, x1:x2]
                if aligned.size == 0:
                    continue
                aligned = cv2.resize(aligned, (112, 112))

                if angle:
                    M = cv2.getRotationMatrix2D((56, 56), angle, 1)
                    aligned = cv2.warpAffine(aligned, M, (112, 112))
                
                if self.clache_enable:
                    # kaneeun adaptive hist-eq test
                    histAligned     =   aligned.copy()
                    histAligned_1   = self.clache.apply(aligned[:,:,0])
                    histAligned_2   = self.clache.apply(aligned[:,:,1])
                    histAligned_3   = self.clache.apply(aligned[:,:,2])
                    histAligned[:, :, 0]   = histAligned_1
                    histAligned[:, :, 1]   = histAligned_2
                    histAligned[:, :, 2]   = histAligned_3

                    # direction : Face yaw, angle : Face roll
                    faces.append([x1, y1, x2, y2, direction, person, histAligned])
                else:
                    faces.append([x1, y1, x2, y2, direction, person, aligned])

        return faces, people
