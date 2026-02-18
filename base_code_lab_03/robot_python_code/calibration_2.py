# calibration_step_2.py  (your "calibration_2.py")
import os
import glob
import numpy as np
import cv2 as cv

# -------------------- config --------------------
PATTERN_SIZE = (7, 7)          # inner corners (cols, rows) for an 8x8 squares board
SQUARE_SIZE_CM = 3.254         # your measured square size (in cm). Use meters if you prefer.
USE_SB = True                  # use findChessboardCornersSB when available (more robust)
SHOW_PREVIEW_MS = 150          # show each detection briefly; set 0 to skip
SAVE_UNDISTORTED_EXAMPLE = True

images_glob = "/Users/stavanchristian/Documents/PhD/Sem 4/Rl/NYU_ROB_GY_6213/base_code_lab_03/robot_python_code/calibration_images_webcam/*.jpg"
out_dir = os.path.join(os.path.dirname(images_glob.split("*")[0]), "calibration_output")
os.makedirs(out_dir, exist_ok=True)

# termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# -------------------- prepare object points --------------------
# (0,0,0), (1,0,0), ... in units of SQUARE_SIZE_CM
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_CM

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# -------------------- gather corners --------------------
images = sorted(glob.glob(images_glob))
print(f"Found {len(images)} images for calibration:\n  {images_glob}")

gray_shape = None
used = 0

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"[WARN] Could not read: {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]  # (w, h)

    found, corners = False, None

    if USE_SB and hasattr(cv, "findChessboardCornersSB"):
        found, corners = cv.findChessboardCornersSB(gray, PATTERN_SIZE, None)
    else:
        flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, flags)

    if found:
        # cornerSubPix refinement (SB is already good, but refinement still helps a bit)
        if corners is not None and corners.dtype != np.float32:
            corners = corners.astype(np.float32)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)
        used += 1

        if SHOW_PREVIEW_MS > 0:
            vis = img.copy()
            cv.drawChessboardCorners(vis, PATTERN_SIZE, corners2, found)
            cv.imshow("Corners (press any key to skip preview)", vis)
            cv.waitKey(SHOW_PREVIEW_MS)

    print(f"{os.path.basename(fname)} -> found={found}")

cv.destroyAllWindows()

if used < 5:
    raise RuntimeError(f"Not enough valid detections for calibration (used {used}). Aim for 15-30.")

print(f"\nUsed {used}/{len(images)} images for calibration.")

# -------------------- calibrate --------------------
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None
)

print("\n=== Calibration Results ===")
print(f"RMS reprojection error (OpenCV 'ret'): {ret}")
print("Camera matrix (K):\n", mtx)
print("Distortion coeffs (k1,k2,p1,p2,k3,...):\n", dist.ravel())

# save to disk (npz)
npz_path = os.path.join(out_dir, "camera_calibration.npz")
np.savez(
    npz_path,
    camera_matrix=mtx,
    dist_coeffs=dist,
    rms=ret,
    pattern_size=np.array(PATTERN_SIZE),
    square_size=SQUARE_SIZE_CM,
    units="cm",
)
print(f"\nSaved calibration to:\n  {npz_path}")

# -------------------- reprojection error (mean) --------------------
mean_error = 0.0
per_image_errors = []

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    # L2 norm / number of points gives average pixel error for that image
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    per_image_errors.append(error)
    mean_error += error

mean_error /= len(objpoints)

print("\n=== Reprojection Error (pixels) ===")
print(f"Mean reprojection error: {mean_error:.6f} px")
print(f"Min/Max per-image error: {min(per_image_errors):.6f} / {max(per_image_errors):.6f} px")

# optional: print worst offenders
worst_k = 5
worst_idx = np.argsort(per_image_errors)[-worst_k:][::-1]
print(f"\nWorst {min(worst_k, len(worst_idx))} images by reprojection error:")
for idx in worst_idx:
    print(f"  idx={idx:02d}  err={per_image_errors[idx]:.6f} px")

# -------------------- undistort an example image --------------------
if SAVE_UNDISTORTED_EXAMPLE:
    example_img = cv.imread(images[0])
    h, w = example_img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    undist = cv.undistort(example_img, mtx, dist, None, newcameramtx)

    # crop using ROI (optional)
    x, y, ww, hh = roi
    undist_crop = undist[y:y+hh, x:x+ww]

    out1 = os.path.join(out_dir, "undistorted_full.png")
    out2 = os.path.join(out_dir, "undistorted_cropped.png")
    cv.imwrite(out1, undist)
    cv.imwrite(out2, undist_crop)
    print("\nSaved undistortion examples:")
    print(" ", out1)
    print(" ", out2)
