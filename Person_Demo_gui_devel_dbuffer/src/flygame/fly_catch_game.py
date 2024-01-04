import numpy as np
import time


# Class FlyCatchGame is responsible game management during frontal image recording.
class FlyCatchGame:

    def __init__(self):
        # Exact fly position
        self.flyPos     = np.zeros(2).astype(np.int16)
        # Fly position boundaries
        # 1st index : 0=left, 1=center, 2=right
        # 2nd index : 0=start, 1=end
        # 3nd index : 0=row, 1=col
        self.flyRange   = np.zeros((3,2,2)).astype(np.int16)
        # Left area range
        self.flyRange[0,0,0]    = 247
        self.flyRange[0,0,1]    = 269
        self.flyRange[0,1,0]    = 494
        self.flyRange[0,1,1]    = 809
        # Center area range
        self.flyRange[1,0,0]    = 742
        self.flyRange[1,0,1]    = 269
        self.flyRange[1,1,0]    = 1237
        self.flyRange[1,1,1]    = 809
        # Right area range
        self.flyRange[2,0,0]    = 1484
        self.flyRange[2,0,1]    = 269
        self.flyRange[2,1,0]    = 1732
        self.flyRange[2,1,1]    = 809
        
        self.landTime           = 0
        # Fly is dead? or alive and flying?
        self.isFlyDead          = False
        # How long took for catching?
        self.catchTime          = 0
        # If person catch fly, it is ready to record person's frontal face.
        # After recording 1 frontal face image, this variable will be set "False"
        self.flyCatched         = False
        # Area candidate Left/Center/Right
        self.areaCandidate      = [0,1,2]
        # If face feature vectors of person are all occupied, 
        # no more game will be executed.
        self.faceFeatComplete   = False
        # Left/Right hand position
        self.leftHandPos        = np.array((0, 0)).astype(np.int16)
        self.rightHandPos       = np.array((0, 0)).astype(np.int16)

        self.initGame()

    def initGame(self):
        self.updateFlyPos()
        self.landTime           = time.time()
        self.isFlyDead          = False
        self.catchTime          = 0
        self.flyCatched         = True
        self.areaCandidate      = [0,1,2]
        self.leftHandPos        = np.array((0, 0)).astype(np.int16)
        self.rightHandPos       = np.array((0, 0)).astype(np.int16)

    def updateFlyPos(self):
        # Left? Center? Right? (0/1/2)
        areaIndex       = np.random.randint(low = 0, high = len(self.areaCandidate))
        areaIndex       = self.areaCandidate[areaIndex]
        self.areaCandidate.remove(areaIndex)
        if len(self.areaCandidate) <= 0:
            self.areaCandidate  = [0,1,2]

        self.flyPos[0]  = np.random.randint(low = self.flyRange[areaIndex, 0, 0], high = self.flyRange[areaIndex, 1, 0])
        self.flyPos[1]  = np.random.randint(low = self.flyRange[areaIndex, 0, 1], high = self.flyRange[areaIndex, 1, 1])
        return

    def update(self, hammerPos):
        
        isHit   = False        

        # If fly is alive
        if self.isFlyDead == False:
            # Fly shall stay same position for 10 sec.
            if (time.time() - self.landTime) > 10:
                self.updateFlyPos()
                self.landTime   = time.time()
            else:
                # Check hammer is covering fly position.
                if ((hammerPos[0] >= (self.flyPos[0] - 60)) and
                    (hammerPos[0] <= (self.flyPos[0] + 60)) and
                    (hammerPos[1] >= (self.flyPos[1] - 60)) and
                    (hammerPos[1] <= (self.flyPos[1] + 60))):
                        self.isFlyDead      = True
                        isHit               = True        
                        # Fly catch time
                        self.catchTime  = time.time() - self.landTime
                        self.landTime   = time.time()
        # If fly is dead
        else:
            # Draw crashed fly 2 sec.
            if (time.time() - self.landTime) > 2:
                self.updateFlyPos()
                self.isFlyDead  = False
                self.landTime   = time.time()
            
        return self.isFlyDead, isHit

    def getFlyDispPos(self):
        # If fly is alive, fly shall vibrate.
        if self.isFlyDead == False:
            dispPos = self.flyPos + np.random.randint(low=-10, high=10, size=2)
        else:
            dispPos = self.flyPos
        return  dispPos

    # If fly is on person display area, move to upper area.
    def avoidFace(self, faceBbox):
        x1, y1, x2, y2  = faceBbox

        if ((self.flyPos[1] >= self.flyRange[0,0,1]) and 
            (self.flyPos[0] >= (x1 - 10)) and 
            (self.flyPos[0] <= (x2 + 10))):
            self.flyPos[1]  = np.random.randint(low = (self.flyRange[0,0,1] - 69), high = (self.flyRange[0,0,1] - 9))
        
        return
        
    
        


