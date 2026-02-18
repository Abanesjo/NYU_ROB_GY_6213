import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import parameters

cap = cv.VideoCapture(parameters.camera_id)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detector_params = aruco.DetectorParameters()
detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
detector = aruco.ArucoDetector(dictionary, detector_params)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    print(f"Detected {len(corners)} markers")
    vis = frame.copy()
    if ids is not None:
        aruco.drawDetectedMarkers(vis, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, parameters.marker_length, parameters.camera_matrix, parameters.dist_coeffs
        )
        for i in range(len(ids)):
            cv.drawFrameAxes(vis, parameters.camera_matrix, parameters.dist_coeffs,
                             rvecs[i], tvecs[i], 0.05)
            print("id", int(ids[i]), "tvec(m):", tvecs[i].ravel())

    cv.imshow("aruco test", vis)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
