import math

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=270, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=480, type=int, help='Resize input to specific height.')

args = parser.parse_args()
lis = []

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

# POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
POSE_PAIRS = [["LHip", "LKnee"], ["RHip", "RKnee"], ["LKnee", "LAnkle"], ["RKnee", "RAnkle"], ["LAnkle", "RAnkle"]]

inWidth = args.width
inHeight = args.height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# cap = cv2.VideoCapture(args.input if args.input else 0)
cap = cv2.VideoCapture('pp.mp4')
# cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 5.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    hasFrame, frame = cap.read()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        if POSE_PAIRS.index(pair) == 4 and points[idFrom] is not None and points[idTo] is not None:
            pixel_lenth = 0.525
            adis = abs(points[idFrom][0] - points[idTo][0])
            bdis = abs(points[idFrom][1] - points[idTo][1])
            dis = math.sqrt(adis**2 + bdis**2) * pixel_lenth
            cv2.putText(frame, "gait distance is {} cm".format(dis), (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0))



    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('OpenPose using OpenCV', frame)
    output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
