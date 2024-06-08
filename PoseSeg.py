import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT


tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    # device='cuda:0',
    device = 'cpu',
    fp16=True,
    # fp16=False,
)

vid = cv2.VideoCapture(0)

while True:
    ret, im = vid.read()

    keypoints = np.random.rand(2, 17, 3)
    mask = np.random.rand(2, 480, 640)
    # substitute by your object detector, input to tracker has to be N X (x, y, x, y, conf, cls)
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])

    tracks = tracker.update(dets, im) # --> M x (x, y, x, y, id, conf, cls, ind)

    # xyxys = tracks[:, 0:4].astype('int') # float64 to int
    # ids = tracks[:, 4].astype('int') # float64 to int
    # confs = tracks[:, 5]
    # clss = tracks[:, 6].astype('int') # float64 to int
    inds = tracks[:, 7].astype('int') # float64 to int

    # in case you have segmentations or poses alongside with your detections you can use
    # the ind variable in order to identify which track is associated to each seg or pose by:
    # masks = masks[inds]
    # keypoints = keypoints[inds]
    # such that you then can: zip(tracks, masks) or zip(tracks, keypoints)

    # break on pressing q or space
    cv2.imshow('BoxMOT segmentation | pose', im)     
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()