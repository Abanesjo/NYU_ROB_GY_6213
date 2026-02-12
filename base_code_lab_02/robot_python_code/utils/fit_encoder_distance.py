import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data_2_10"
OUTPUT_DIR = Path(__file__).resolve().parent

FILES_AND_DATA = [
    ["robot_data_50_0_10_02_26_16_45_46.pkl", 15 / 100, 0 / 100],
    ["robot_data_50_0_10_02_26_16_49_10.pkl", 17 / 100, 0 / 100],
    ["robot_data_50_0_10_02_26_16_50_12.pkl", 19 / 100, 0 / 100],
    ["robot_data_50_0_10_02_26_16_51_06.pkl", 19.5 / 100, 0 / 100],
    ["robot_data_50_0_10_02_26_16_51_49.pkl", 20 / 100, 0 / 100],
    ["robot_data_50_0_10_02_26_16_52_40.pkl", 17 / 100, 0 / 100],
    ["robot_data_50_0_10_02_26_16_54_53.pkl", 44.5 / 100, -0.75],
    ["robot_data_50_0_10_02_26_16_58_44.pkl", 44 / 100, -0.37],
    ["robot_data_50_0_10_02_26_17_00_17.pkl", 43.5 / 100, -0.45],
    ["robot_data_50_0_10_02_26_17_01_59.pkl", 44 / 100, -0.35],
    ["robot_data_50_0_10_02_26_17_03_15.pkl", 42 / 100, -0.20],
    ["robot_data_50_0_10_02_26_17_04_39.pkl", 77 / 100, -0.75],
    ["robot_data_50_0_10_02_26_17_09_49.pkl", 67.5 / 100, -1.0],
    ["robot_data_50_0_10_02_26_17_11_55.pkl", 69 / 100, -1.0],
    ["robot_data_50_0_10_02_26_17_13_03.pkl", 69.5 / 100, -0.9],
    ["robot_data_50_0_10_02_26_17_22_02.pkl", 67.5 / 100, -0.80],
    ["robot_data_50_0_10_02_26_17_24_52.pkl", 88.5 / 100, -1.25],
    ["robot_data_50_0_10_02_26_17_26_58.pkl", 85 / 100, -1.30],
    ["robot_data_50_0_10_02_26_17_28_10.pkl", 87.5 / 100, -0.95],
    ["robot_data_50_0_10_02_26_17_29_35.pkl", 89.5 / 100, -0.9],
    ["robot_data_50_0_10_02_26_17_30_43.pkl", 90 / 100, -0.80],
    ["robot_data_50_0_10_02_26_17_36_19.pkl", 109 / 100, -1.20],
    ["robot_data_50_0_10_02_26_17_38_23.pkl", 107 / 100, -1.1],
    ["robot_data_50_0_10_02_26_17_40_05.pkl", 107 / 100, -1.1],
    ["robot_data_50_0_10_02_26_17_41_08.pkl", 105 / 100, -1.30],
    ["robot_data_50_0_10_02_26_17_42_10.pkl", 105 / 100, -1.3],
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


def _encoder_delta(data_dict):
    signals = data_dict["robot_sensor_signal"]
    if not signals:
        return 0
    return signals[-1].encoder_counts - signals[0].encoder_counts


def _distance_meters(x_m, y_in):
    y_m = y_in * 0.0254
    return y_m, math.sqrt(x_m * x_m + y_m * y_m)


def _fit_origin_line(e, s):
    e = np.asarray(e)
    s = np.asarray(s)
    denom = float(e @ e)
    if denom == 0:
        return 0.0
    return float((e @ s) / denom)


def main():
    rows = []
    for filename, x_m, y_in in FILES_AND_DATA:
        path = _resolve_path(filename)
        data = _load_pickle(path)
        e = _encoder_delta(data)
        y_m, s = _distance_meters(x_m, y_in)
        rows.append(
            {
                "filename": path.name,
                "encoder_counts": e,
                "x_m": x_m,
                "y_in": y_in,
                "y_m": y_m,
                "s": s,
            }
        )

    e_vals = np.array([r["encoder_counts"] for r in rows], dtype=float)
    s_vals = np.array([r["s"] for r in rows], dtype=float)

    k = _fit_origin_line(e_vals, s_vals)
    s_hat = k * e_vals
    residuals = s_vals - s_hat
    sigma_s2 = residuals ** 2
    # remove single largest variance outlier
    outlier_idx = np.argmax(sigma_s2)
    mask = np.ones_like(sigma_s2, dtype=bool)
    mask[outlier_idx] = False

    rmse_s = np.sqrt(np.mean(residuals ** 2))


    sigma_const = np.mean(sigma_s2[mask])
    sigma_hat = np.full_like(sigma_s2, sigma_const)

    rmse_sigma = np.sqrt(np.mean((sigma_s2[mask] - sigma_const) ** 2))



    for i, r in enumerate(rows):
        r["s_hat"] = float(s_hat[i])
        r["residual"] = float(residuals[i])
        r["sigma_s2"] = float(sigma_s2[i])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "encoder_distance_fit.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "encoder_counts",
                "x_m",
                "y_in",
                "y_m",
                "s",
                "s_hat",
                "residual",
                "sigma_s2",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["filename"],
                    r["encoder_counts"],
                    r["x_m"],
                    r["y_in"],
                    r["y_m"],
                    r["s"],
                    r["s_hat"],
                    r["residual"],
                    r["sigma_s2"],
                ]
            )

    plt.figure()
    plt.plot(e_vals, s_vals, "ko", label=r"Measured $s$")
    e_grid = np.linspace(e_vals.min(), e_vals.max(), 200)
    plt.plot(
        e_grid,
        k * e_grid,
        "r-",
        label=rf"$\hat s = {k:.6f}\,e$" + "\n" + rf"RMSE = {rmse_s:.4f} m",
    )


    # plt.plot(e_vals, s_hat, "r-", label=rf"$\hat s = {k:.6f}\,e$")
    plt.xlabel(r"Encoder counts $e$")
    plt.ylabel(r"Distance traveled $s$ (m)")

    plt.title("Distance vs Encoder Counts")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "encoder_distance_fit.png")

    plt.figure()
    plt.plot(e_vals, sigma_s2, "ko", label=r"Residual$^2$")
    plt.axhline(
        sigma_const,
        color="r",
        linestyle="-",
        label=rf"Const fit ($\bar{{\sigma}}_s^2 = {sigma_const:.3e}$)"
            + "\n"
            + rf"RMSE = {rmse_sigma:.3e}",
    )
    plt.xlabel(r"Encoder counts $e$")
    plt.ylabel(r"$\sigma_s^2$ (m$^2$)")
    plt.title(r"Distance variance $\sigma_s^2$ vs encoder counts $e$")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "encoder_variance_fit.png")


    plt.show()

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {OUTPUT_DIR / 'encoder_distance_fit.png'}")
    print(f"Wrote plot: {OUTPUT_DIR / 'encoder_variance_fit.png'}")
    print(f"f_se slope k = {k}")
    # print(f"f_ss coeffs (a,b,c) = {coeffs}")


if __name__ == "__main__":
    main()
