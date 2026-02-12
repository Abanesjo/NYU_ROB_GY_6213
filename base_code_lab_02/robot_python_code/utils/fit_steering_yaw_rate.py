import csv
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
        

DATA_DIR = Path(__file__).resolve().parents[1] / "data_steer_2_10"
OUTPUT_DIR = Path(__file__).resolve().parent
TRIAL_TIME_S = 4.0
# Set to True to fit w = k * tan(alpha) instead of linear in alpha.
USE_TAN_MODEL = True

FILES_AND_DATA = [
    ###### Steer 5 ######
    ['robot_data__50_5_10_02_26_21_41_57.pkl', 43.5/100, -0.5], # filename, measured distance in meters
    ['robot_data__50_5_10_02_26_21_48_14.pkl', 44/100, -0.75],
    ['robot_data__50_5_10_02_26_21_51_12.pkl', 44.5/100, -0.80],
    ####### Steer 10   ######
    ['robot_data__50_10_10_02_26_21_53_44.pkl', 41.5/100, -2.5],
    ['robot_data__50_10_10_02_26_21_55_31.pkl', 39.5/100, -2.5],
    ['robot_data__50_10_10_02_26_21_57_15.pkl', 41.5/100, -2.6],
    ####### Steer 15 ######
    ['robot_data__50_15_10_02_26_22_00_12.pkl', 37.5/100, -2.9375],
    ['robot_data__50_15_10_02_26_22_05_05.pkl', 38/100, -3.0625],
    ['robot_data__50_15_10_02_26_22_07_37.pkl', 38/100, -3.0],
    # ###### Steer 20 ######
    ['robot_data__50_20_10_02_26_22_09_20.pkl', 34/100, -3.5],
    ['robot_data__50_20_10_02_26_22_10_41.pkl', 33/100, -3.475],
    ['robot_data__50_20_10_02_26_22_12_35.pkl', 33/100, -3.4375],
    ####### Steer -5 #######
    ['robot_data__50_-5_10_02_26_22_15_39.pkl', 43.5/100, 1.75],
    ['robot_data__50_-5_10_02_26_22_21_13.pkl', 44/100, 1.625],
    ['robot_data__50_-5_10_02_26_22_23_06.pkl', 44/100, 1.75],
    ####### Steer -10 #######
    ['robot_data__50_-10_10_02_26_22_25_45.pkl', 38.5/100, 2.625],
    ['robot_data__50_-10_10_02_26_22_30_50.pkl', 41/100, 2.875],
    ['robot_data__50_-10_10_02_26_22_33_17.pkl', 38.5/100, 2.625],
    ####### Steer -15 #######
    ['robot_data__50_-15_10_02_26_22_34_38.pkl', 36.5/100, 3.25],
    ['robot_data__50_-15_10_02_26_22_35_51.pkl', 37/100, 3.5],
    ['robot_data__50_-15_10_02_26_22_37_01.pkl', 37.5/100, 3.625],
    # ###### Steer -20 ######
    ['robot_data__50_-20_10_02_26_22_39_15.pkl', 30.5/100, 2.9375],
    ['robot_data__50_-20_10_02_26_22_41_07.pkl', 30/100, 2.69],
    ['robot_data__50_-20_10_02_26_22_42_49.pkl', 30/100, 2.875]
    ]


@dataclass
class RobotSensorSignal:
    encoder_counts: int
    steering: int
    num_lidar_rays: int
    angles: list
    distances: list


class _RobotUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "robot_python_code" and name == "RobotSensorSignal":
            return RobotSensorSignal
        return super().find_class(module, name)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return _RobotUnpickler(f).load()


def _resolve_path(filename: str) -> Path:
    path = DATA_DIR / filename
    if path.exists():
        return path
    if filename.startswith("robot_data_"):
        alt = "robot_data__" + filename[len("robot_data_") :]
        alt_path = DATA_DIR / alt
        if alt_path.exists():
            return alt_path
    return path


def _parse_alpha_from_filename(filename: str) -> int:
    # Expected pattern: robot_data__50_-10_10_... or robot_data_50_-10_10_...
    # m = re.search(r"robot_data_+50_(-?\\d+)_", filename)
    m = re.search(r"robot_data_+50_(-?\d+)_", filename)
    if not m:
        raise ValueError(f"Cannot parse steering from filename: {filename}")
    return int(m.group(1))


def _distance_components(x_m, y_in):
    y_m = y_in * 0.0254
    return y_m, math.sqrt(x_m * x_m + y_m * y_m)


def _yaw_from_xy(x_m, y_m):
    return math.atan2(y_m, x_m)


def _fit_origin_line(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    denom = float(x @ x)
    if denom == 0:
        return 0.0
    return float((x @ y) / denom)


def main():
    rows = []
    for filename, x_m, y_in in FILES_AND_DATA:
        path = _resolve_path(filename)
        _load_pickle(path)  # load to validate file; not used directly
        alpha = _parse_alpha_from_filename(path.name)
        y_m, _ = _distance_components(x_m, y_in)
        theta = _yaw_from_xy(x_m, y_m)
        w = theta / TRIAL_TIME_S
        rows.append(
            {
                "filename": path.name,
                "alpha": alpha,
                "x_m": x_m,
                "y_in": y_in,
                "y_m": y_m,
                "theta_rad": theta,
                "w_rad_s": w,
            }
        )

    alpha_vals = np.array([r["alpha"] for r in rows], dtype=float)
    w_vals = np.array([r["w_rad_s"] for r in rows], dtype=float)

    if USE_TAN_MODEL:
        def tanh_model(alpha, w_max, c):
            return w_max * np.tanh(c * np.deg2rad(alpha))

        params, _ = curve_fit(
            tanh_model,
            alpha_vals,
            w_vals,
            p0=[max(abs(w_vals)), 1.0]
        )

        w_max, c = params
        w_hat = tanh_model(alpha_vals, w_max, c)

    else:
        k = _fit_origin_line(alpha_vals, w_vals)
        w_hat = k * alpha_vals
    residuals = w_vals - w_hat
    sigma_w2 = residuals ** 2
    rmse_w = np.sqrt(np.mean(residuals ** 2))
    coeffs = np.polyfit(alpha_vals, sigma_w2, 2)
    sigma_hat = np.polyval(coeffs, alpha_vals)
    rmse_sigma = np.sqrt(np.mean((sigma_w2 - sigma_hat) ** 2))

    

    for i, r in enumerate(rows):
        r["w_hat"] = float(w_hat[i])
        r["residual"] = float(residuals[i])
        r["sigma_w2"] = float(sigma_w2[i])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "steering_yaw_rate_fit.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "alpha",
                "tan_alpha",
                "x_m",
                "y_in",
                "y_m",
                "theta_rad",
                "w_rad_s",
                "w_hat",
                "residual",
                "sigma_w2",
            ]
        )
        for r in rows:
            tanh_alpha_val = math.tanh(math.radians(r["alpha"]))
            writer.writerow(
                [
                    r["filename"],
                    r["alpha"],
                    tanh_alpha_val,
                    r["x_m"],
                    r["y_in"],
                    r["y_m"],
                    r["theta_rad"],
                    r["w_rad_s"],
                    r["w_hat"],
                    r["residual"],
                    r["sigma_w2"],
                ]
            )

    plt.figure()
    plt.plot(alpha_vals, w_vals, "ko", label="Measured w")
    alpha_grid = np.linspace(alpha_vals.min(), alpha_vals.max(), 200)
    if USE_TAN_MODEL:
        # w_max, c = params
        w_grid = tanh_model(alpha_grid, w_max, c)
        # w_grid = k * np.tan(np.deg2rad(alpha_grid))
        plt.plot(
            alpha_grid,
            w_grid,
            "r-",
            label=rf"$\omega = {w_max:.3f}\,\tanh({c:.3f}\,\alpha)$"
                        + "\n"
            + rf"RMSE = {rmse_w:.3e}",
        )

    else:
        w_grid = k * alpha_grid
        plt.plot(alpha_grid, w_grid, "r-", label=f"f_w(alpha) = {k:.6f} * alpha")
    plt.xlabel(r"Steering command $\alpha$ (deg)")
    plt.ylabel(r"Yaw rate $\omega$ (rad/s)")
    plt.title("Yaw Rate vs Steering Command")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "steering_yaw_rate_fit.png")

    plt.figure()
    alpha_grid = np.linspace(alpha_vals.min(), alpha_vals.max(), 200)
    sigma_grid = np.polyval(coeffs, alpha_grid)

    plt.plot(alpha_vals, sigma_w2, "ko", label=r"Residual$^2$")
    plt.plot(alpha_grid, sigma_grid, "r-", label=r"$f_{sw}(\alpha)$ quadratic fit"             + "\n"
            + rf"RMSE = {rmse_sigma:.3e}",)
    plt.xlabel(r"Steering command $\alpha$ (deg)")
    plt.ylabel(r"$\sigma_\omega^2$ (rad$^2$/s$^2$)")
    plt.title("Yaw Rate Variance vs Steering Command")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "steering_variance_fit.png")

    plt.show()

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {OUTPUT_DIR / 'steering_yaw_rate_fit.png'}")
    print(f"Wrote plot: {OUTPUT_DIR / 'steering_variance_fit.png'}")
    if USE_TAN_MODEL:
        print(f"f_w params (tanh): w_max={w_max}, c={c}")
    else:
        print(f"f_w slope k = {k}")

    print(f"f_sw coeffs (a,b,c) = {coeffs}")


if __name__ == "__main__":
    main()
