from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import math

import matplotlib.pyplot as plt

import motion_models


DEFAULT_PICKLE_FILE = Path(__file__).resolve().parent / "data_validation" / "curvy.pkl"


class _RobotSensorSignal:
    pass


class _RobotDataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "robot_python_code" and name == "RobotSensorSignal":
            return _RobotSensorSignal
        return super().find_class(module, name)


def load_data_file(filename: Path) -> dict:
    with filename.open("rb") as file_handle:
        return _RobotDataUnpickler(file_handle).load()


def extract_trial_series(data_dict: dict) -> tuple[list[float], list[int], list[float]]:
    time_list = [float(t) for t in data_dict["time"]]
    control_signal_list = data_dict["control_signal"]
    robot_sensor_signal_list = data_dict["robot_sensor_signal"]

    encoder_count_list: list[int] = [int(row.encoder_counts) for row in robot_sensor_signal_list]
    steering_angle_list: list[float] = [float(row[1]) for row in control_signal_list]

    if not time_list:
        raise ValueError("No time samples found in trial data.")
    if len(encoder_count_list) != len(time_list) or len(steering_angle_list) != len(time_list):
        raise ValueError("time, encoder, and steering series lengths do not match.")

    return time_list, encoder_count_list, steering_angle_list


def plot_angle_over_time(file_path: Path, use_relative_time: bool = True) -> None:
    data_dict = load_data_file(file_path)
    time_list, encoder_count_list, steering_angle_list = extract_trial_series(data_dict)

    model = motion_models.MyMotionModel([0.0, 0.0, 0.0], encoder_count_list[0])
    _, _, theta_list = model.traj_propagation(time_list, encoder_count_list, steering_angle_list)
    theta_deg_list = [math.degrees(theta) for theta in theta_list]

    if use_relative_time:
        t0 = time_list[0]
        time_plot = [t - t0 for t in time_list]
    else:
        time_plot = time_list

    fig, ax = plt.subplots()
    ax.plot(time_plot, theta_deg_list, "b-")
    if time_plot:
        ax.plot([time_plot[-1]], [theta_deg_list[-1]], "bo", markersize=4)
        if len(time_plot) > 1:
            ax.set_xlim(min(time_plot), max(time_plot))
    ax.set_title("Estimated Heading Angle vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.margins(x=0.02, y=0.05)
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot motion-model heading angle from a trial pickle file.")
    parser.add_argument(
        "pkl_file",
        nargs="?",
        default=str(DEFAULT_PICKLE_FILE),
        help="Path to input .pkl trial file",
    )
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="Use absolute logged time instead of relative time starting at zero",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = Path(args.pkl_file).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input pickle file not found: {file_path}")

    plot_angle_over_time(file_path, use_relative_time=not args.absolute_time)


if __name__ == "__main__":
    main()
