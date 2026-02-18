import glob
import pickle
import numpy as np
import math

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def circular_mean(angles):
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return math.atan2(s, c)

# ---- load all your pose log files (edit pattern if needed) ----
files = sorted(glob.glob("./data_aruco/robot_data_*_yaw.pkl"))
print("Found", len(files), "log files")

Sigmas = []

for f in files:
    with open(f, "rb") as h:
        d = pickle.load(h)

    cam = np.array(d["camera_sensor_signal"])  # shape (T, 6)
    mask = np.linalg.norm(cam[:, :3], axis=1) > 1e-9  # keep non-zero tvec
    cam = cam[mask]

    # measurement vector used by your EKF: [tvec_x, tvec_y, rvec_z]
    x = cam[:, 0]
    y = cam[:, 1]
    th = cam[:, 5]   # NOTE: this is rvec_z, consistent with your current EKF code

    # circular mean for angle + wrapped residuals
    th_mean = circular_mean(th)
    dth = np.array([wrap_to_pi(a - th_mean) for a in th])

    z = np.vstack([x, y, dth]).T  # centered theta residuals, but x,y still uncentered
    z[:, 0] -= np.mean(z[:, 0])
    z[:, 1] -= np.mean(z[:, 1])

    Sigma = np.cov(z.T, bias=False)  # 3x3
    Sigmas.append(Sigma)

    print(f"\n{f}")
    print("Sigma (x,y,theta):\n", Sigma)
    print("std (x,y,theta):", np.sqrt(np.diag(Sigma)))

# ---- aggregate across poses ----
Sigmas = np.array(Sigmas)

Q_avg = np.mean(Sigmas, axis=0)

Q_worst_diag = np.diag(np.max(Sigmas[:, [0,1,2], [0,1,2]], axis=0))

print("\n=== Aggregate ===")
print("Q_avg:\n", Q_avg)
print("Q_worst_diag:\n", Q_worst_diag)
