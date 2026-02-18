import glob
import pickle
import numpy as np
import cv2

def patch_file(path: str, overwrite: bool = False):
    with open(path, "rb") as f:
        data = pickle.load(f)

    cam = data.get("camera_sensor_signal", None)
    if cam is None:
        print(f"[skip] {path}: no camera_sensor_signal")
        return

    cam = np.array(cam, dtype=float)  # shape (T,6) typically
    if cam.ndim != 2 or cam.shape[1] < 6:
        print(f"[skip] {path}: unexpected camera_sensor_signal shape {cam.shape}")
        return

    # Replace column 5 (old rvec_z) with yaw extracted from Rodrigues
    for i in range(cam.shape[0]):
        rvec = cam[i, 3:6].astype(float)
        R, _ = cv2.Rodrigues(rvec)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        cam[i, 5] = yaw

    data["camera_sensor_signal"] = cam.tolist()

    out_path = path if overwrite else path.replace(".pkl", "_yaw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[ok] wrote {out_path}")

def main():
    files = sorted(glob.glob("/Users/stavanchristian/Documents/PhD/Sem 4/Rl/NYU_ROB_GY_6213/base_code_lab_03/robot_python_code/data_aruco/robot_data_*.pkl"))
    print("Found", len(files), "files")
    for p in files:
        patch_file(p, overwrite=False)  # set True if you want in-place overwrite

if __name__ == "__main__":
    main()
