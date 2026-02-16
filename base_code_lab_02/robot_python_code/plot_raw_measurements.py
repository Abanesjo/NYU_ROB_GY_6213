from __future__ import annotations

import math
import numbers
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

PICKLE_FILE_PATH = Path(__file__).resolve().parent / "data_curved" / "robot_data_50_10_06_02_26_16_18_04.pkl"
USE_RELATIVE_TIME = True
SHOW_PLOTS = True


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


def _flatten_numeric(value, prefix: str, output: dict[str, float]) -> None:
    if isinstance(value, bool):
        output[prefix] = float(value)
        return

    if isinstance(value, numbers.Number):
        output[prefix] = float(value)
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten_numeric(item, f"{prefix}[{index}]", output)
        return

    if hasattr(value, "__dict__"):
        for field_name, field_value in vars(value).items():
            _flatten_numeric(field_value, f"{prefix}.{field_name}", output)


def extract_measurement_series(data_dict: dict, sample_count: int) -> dict[str, list[float]]:
    series_map: dict[str, list[float]] = {}
    for key, sample_list in data_dict.items():
        if key == "time":
            continue

        for sample_index, sample in enumerate(sample_list):
            flattened: dict[str, float] = {}
            _flatten_numeric(sample, key, flattened)
            for name, value in flattened.items():
                if name not in series_map:
                    series_map[name] = [math.nan] * sample_count
                series_map[name][sample_index] = value

    return series_map


def pretty_name(name: str) -> str:
    replacements = {
        "control_signal[0]": "control_signal.speed_command",
        "control_signal[1]": "control_signal.steering_command",
    }
    return replacements.get(name, name)


def include_series(name: str) -> bool:
    if name.startswith("camera_sensor_signal"):
        return False
    if name.startswith("robot_sensor_signal.angles"):
        return False
    if name.startswith("robot_sensor_signal.distances"):
        return False
    if name == "robot_sensor_signal.num_lidar_rays":
        return False
    return True


def plot_raw_measurements(file_path: Path, use_relative_time: bool = True, show_plots: bool = True) -> None:
    data_dict = load_data_file(file_path)
    time_list = [float(t) for t in data_dict["time"]]
    if not time_list:
        raise ValueError("The input file has no time samples.")

    if use_relative_time:
        start_time = time_list[0]
        time_list = [t - start_time for t in time_list]

    sample_count = len(time_list)
    series_map = extract_measurement_series(data_dict, sample_count)
    series_names = [name for name in sorted(series_map.keys()) if include_series(name)]
    if not series_names:
        raise ValueError("No numeric raw measurements were found to plot.")

    print(f"Loaded file: {file_path}")
    print(f"Time samples: {sample_count}")
    print(f"Found {len(series_names)} raw measurement channels (camera/lidar excluded).")

    figure, axes = plt.subplots(len(series_names), 1, sharex=True, figsize=(10, 2.5 * len(series_names)))
    if len(series_names) == 1:
        axes = [axes]

    for axis, name in zip(axes, series_names):
        y_values = series_map[name]
        label = pretty_name(name)
        axis.plot(time_list, y_values, "b-")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(True)

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("Raw Measurements vs Time")
    figure.tight_layout()

    if show_plots:
        plt.show()


def main() -> None:
    plot_raw_measurements(PICKLE_FILE_PATH, use_relative_time=USE_RELATIVE_TIME, show_plots=SHOW_PLOTS)


if __name__ == "__main__":
    main()
