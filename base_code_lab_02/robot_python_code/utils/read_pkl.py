import argparse
import csv
import pickle
import sys
from pathlib import Path


DEFAULT_PATH = "/Users/stavanchristian/Documents/PhD/Sem 4/Rl/NYU_ROB_GY_6213/base_code_lab_02/robot_python_code/data/robot_data_50_0_06_02_26_16_00_04.pkl"


def _ensure_robot_python_code_on_path():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    # Import needed to resolve classes during unpickling.
    import robot_python_code  # noqa: F401


def _summarize(data):
    if isinstance(data, dict):
        print("keys:", list(data.keys()))
        for key, value in data.items():
            if isinstance(value, list):
                if value:
                    print(f"{key}: list len={len(value)}, first_type={type(value[0]).__name__}")
                else:
                    print(f"{key}: list len=0")
            else:
                print(f"{key}: type={type(value).__name__}")
    else:
        length = len(data) if hasattr(data, "__len__") else "n/a"
        print(f"type={type(data).__name__}, len={length}")


def _write_basic_csv(data, out_path):
    if not isinstance(data, dict):
        raise ValueError("Expected dict data for CSV export.")
    required = ["time", "control_signal", "robot_sensor_signal"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in data.")

    time_list = data["time"]
    control_list = data["control_signal"]
    robot_list = data["robot_sensor_signal"]

    n = min(len(time_list), len(control_list), len(robot_list))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "control_signal_0",
            "control_signal_1",
            "encoder_counts",
            "steering",
            "num_lidar_rays",
        ])
        for i in range(n):
            ctrl = control_list[i]
            robot = robot_list[i]
            writer.writerow([
                time_list[i],
                ctrl[0] if len(ctrl) > 0 else "",
                ctrl[1] if len(ctrl) > 1 else "",
                getattr(robot, "encoder_counts", ""),
                getattr(robot, "steering", ""),
                getattr(robot, "num_lidar_rays", ""),
            ])


def main():
    parser = argparse.ArgumentParser(description="Read a pickle file produced by robot_python_code.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="Path to .pkl file")
    parser.add_argument("--full", action="store_true", help="Print full object instead of summary")
    parser.add_argument("--csv", action="store_true", help="Write basic CSV alongside output")
    args = parser.parse_args()

    _ensure_robot_python_code_on_path()

    with open(args.path, "rb") as f:
        data = pickle.load(f)

    if args.full:
        print(data)
    else:
        _summarize(data)

    if args.csv:
        csv_path = str(Path(args.path).with_suffix(".csv"))
        _write_basic_csv(data, csv_path)
        print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
