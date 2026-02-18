import cv2 as cv
import cv2.aruco as aruco
import numpy as np

TAG_PATH = "aruco_6x6_250_id23.png"  # <-- change to your actual filename

img = cv.imread(TAG_PATH, cv.IMREAD_GRAYSCALE)
assert img is not None, "Could not read tag image"

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
detector = aruco.ArucoDetector(dictionary, params)

corners, ids, rejected = detector.detectMarkers(img)
print("Detected ids:", ids)

vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
if ids is not None:
    aruco.drawDetectedMarkers(vis, corners, ids)
cv.imshow("tag detection (digital)", vis)
cv.waitKey(0)
cv.destroyAllWindows()
