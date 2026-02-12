# External Libraries
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np

# Internal Libraries
import parameters
import robot_python_code
import motion_models

# Unit conversion
def inches_to_meters(inches):
    return inches * 0.0254

# Resolve filenames that may have a double underscore after robot_data
def resolve_filename(directory, filename):
    path = Path(directory) / filename
    if path.exists():
        return str(path)
    if filename.startswith("robot_data_"):
        alt = "robot_data__" + filename[len("robot_data_") :]
        alt_path = Path(directory) / alt
        if alt_path.exists():
            return str(alt_path)
    return str(path)

# Open a file and return data in a form ready to plot
def get_file_data(filename):
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    # The dictionary should have keys ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal']
    time_list = data_dict['time']
    control_signal_list = data_dict['control_signal']
    robot_sensor_signal_list = data_dict['robot_sensor_signal']
    encoder_count_list = []
    velocity_list = []
    steering_angle_list = []
    for row in robot_sensor_signal_list:
        encoder_count_list.append(row.encoder_counts)
    for row in control_signal_list:
        velocity_list.append(row[0])
        steering_angle_list.append(row[1])
    
    return time_list, encoder_count_list, velocity_list, steering_angle_list


# For a given trial, plot the encoder counts, velocities, steering angles
def plot_trial_basics(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    
    plt.plot(time_list, encoder_count_list)
    plt.title('Encoder Values')
    plt.show()
    plt.plot(time_list, velocity_list)
    plt.title('Speed')
    plt.show()
    plt.plot(time_list, steering_angle_list)
    plt.title('Steering')
    plt.show()


# Plot a trajectory using the motion model, input data ste from a single trial.
def run_my_model_on_trial(filename, show_plot = True, plot_color = 'ko'):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, y_list, theta_list = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)

    plt.plot(x_list, y_list,plot_color)
    plt.title('Motion Model Predicted XY Traj (m)')
    plt.axis([-0.5, 1.5, -1, 1])
    if show_plot:
        plt.show()

def run_my_model_on_trial_return(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    return motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)

# Plot predicted trajectories and overlay measured endpoints
def plot_trajs_with_measured_endpoints(files_and_data, directory, title, save_path=None):
    plot_color_list = ['r-','k-','g-','c-', 'b-', 'r-','k-','g-','c-', 'b-','r-','k-','g-','c-', 'b.', 'r-','k-','g-','c-', 'b-']
    count = 0
    for row in files_and_data:
        filename = row[0]
        x_m = row[1]
        y_in = row[2]
        filepath = resolve_filename(directory, filename)
        x_list, y_list, _ = run_my_model_on_trial_return(filepath)
        plot_color = plot_color_list[count % len(plot_color_list)]
        plt.plot(x_list, y_list, plot_color, label='Predicted Traj' if count == 0 else None)
        y_m = inches_to_meters(y_in)
        plt.plot([x_m], [y_m], 'rx', label='Measured End' if count == 0 else None)
        count += 1
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='best')
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.show()


# Iterate through many trials and plot them as trajectories with motion model
def plot_many_trial_predictions(directory):
    directory_path = Path(directory)
    plot_color_list = ['r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.','r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.']
    count = 0
    for item in directory_path.iterdir():
        filename = item.name
        plot_color = plot_color_list[count]
        run_my_model_on_trial(directory + filename, False, plot_color)
        count += 1
    plt.show()

# Calculate the predicted distance from single trial for a motion model
def run_my_model_to_predict_distance(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, _, _ = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)
    distance = x_list[-30]
    
    return distance

# Calculate the differences between two lists, and square them.
def get_diff_squared(m_list,p_list):
    diff_squared_list = []
    for i in range(len(m_list)):
        diff_squared = math.pow(m_list[i]-p_list[i],2)
        diff_squared_list.append(diff_squared)

    coefficients = np.polyfit(m_list, diff_squared_list, 2)
    p=np.poly1d(coefficients)

    plt.plot(m_list, diff_squared_list,'ko')
    plt.plot(m_list, p(m_list),'ro')
    plt.title("Error Squared (m^2)")
    plt.xlabel('Measured distance travelled (m)')
    plt.ylabel('(Actual - Predicted)^2 (m^2)')
    plt.show()

    return diff_squared_list


# Open files, plot them to predict with the motion model, and compare with real values
def process_files_and_plot(files_and_data, directory):
    predicted_distance_list = []
    measured_distance_list = []
    for row in files_and_data:
        filename = row[0]
        measured_distance = row[1]
        measured_distance_list.append(measured_distance)
        predicted_distance = run_my_model_to_predict_distance(directory + filename)
        predicted_distance_list.append(predicted_distance)

    # Plot predicted and measured distance travelled.
    plt.plot(measured_distance_list+[0], predicted_distance_list+[0], 'ko')
    plt.plot([0,1.7],[0,1.7])
    plt.title('Distance Trials')
    plt.xlabel('Measured Distance (m)')
    plt.ylabel('Predicted Distance (m)')
    plt.legend(['Measured vs Predicted', 'Slope 1 Line'])
    plt.show()

    # Plot the associated variance
    get_diff_squared(measured_distance_list, predicted_distance_list)




# --- helper: covariance ellipse ---
def _plot_cov_ellipse(mean_xy, cov_xy, n_std=2.0, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 0.0)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width = 2 * n_std * math.sqrt(vals[0])
    height = 2 * n_std * math.sqrt(vals[1])
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))

    t = np.linspace(0, 2 * math.pi, 200)
    ellipse = np.vstack([0.5 * width * np.cos(t),
                          0.5 * height * np.sin(t)])

    R = np.array([[math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
                  [math.sin(math.radians(angle)),  math.cos(math.radians(angle))]])

    xy = R @ ellipse + mean_xy.reshape(2, 1)
    ax.plot(xy[0], xy[1], **kwargs)


# --- main Part 7 plot ---
def sample_model_with_trajectories(
    motion_models,
    num_samples=200,
    duration=10.0,
    steers=(-20, -10, 0, 10, 20),
    n_std=2.0,
    save_path="motion_model_trajectories_uncertainty.png",
):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Starting point (same for all)
    ax.plot(0, 0, "ks", markersize=6, label="Start")

    for steer in steers:
        endpoints = []

        for _ in range(num_samples):
            model = motion_models.MyMotionModel([0, 0, 0], 0)
            x, y, _ = model.generate_simulated_traj(duration, steering_cmd=steer)

            # Plot full trajectory (faint)
            ax.plot(x, y, color="gray", alpha=0.05)

            endpoints.append([x[-1], y[-1]])

        endpoints = np.asarray(endpoints)
        mean_xy = endpoints.mean(axis=0)
        cov_xy = np.cov(endpoints.T)

        # Endpoints
        ax.plot(endpoints[:, 0], endpoints[:, 1], ".", alpha=0.25,
                label=f"Endpoints (steer={steer})")

        # Mean endpoint
        ax.plot(mean_xy[0], mean_xy[1], "o")

        # Covariance ellipse
        _plot_cov_ellipse(mean_xy, cov_xy, n_std=n_std, ax=ax, linewidth=2)

    # Plot styling
    ax.set_title("Sampled Motion Model Trajectories and Endpoint Uncertainty")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_zoomed_uncertainty_for_steer(
    motion_models,
    steer=20,
    num_samples=500,
    duration=10.0,
    n_std=2.0,
    pad_std=3.0,
    save_path="zoomed_endpoint_uncertainty.png",
):
    endpoints = []

    for _ in range(num_samples):
        model = motion_models.MyMotionModel([0, 0, 0], 0)
        x, y, _ = model.generate_simulated_traj(duration, steering_cmd=steer)
        endpoints.append([x[-1], y[-1]])

    endpoints = np.asarray(endpoints, dtype=float)

    mu = endpoints.mean(axis=0)
    cov = np.cov(endpoints.T)

    # eigenvalues determine uncertainty scale
    vals = np.linalg.eigvalsh(cov)
    vals = np.maximum(vals, 1e-12)
    std_max = math.sqrt(vals.max())

    fig, ax = plt.subplots(figsize=(6, 6))

    # endpoints
    ax.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        "k.",
        alpha=0.35,
        label=f"Endpoints (steer={steer})",
    )

    # mean
    ax.plot(mu[0], mu[1], "ro", markersize=6, label="Mean")

    # covariance ellipse
    _plot_cov_ellipse(mu, cov, n_std=n_std, ax=ax, linewidth=2)

    # local zoom based on covariance
    half_width = pad_std * n_std * std_max
    ax.set_xlim(mu[0] - half_width, mu[0] + half_width)
    ax.set_ylim(mu[1] - half_width, mu[1] + half_width)

    ax.set_title(f"Endpoint Uncertainty (steer={steer}), {n_std}σ ellipse")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    # Optional numeric diagnostic
    anisotropy = math.sqrt(vals.max() / vals.min())
    print(f"[steer={steer}] mean={mu}, eig(cov)={vals}, anisotropy≈{anisotropy:.2f}")



if True:
    sample_model_with_trajectories(motion_models, num_samples=400, duration=10.0, steers=(20, 10, 0, -10, -20), n_std=2.0)
    plot_zoomed_uncertainty_for_steer(motion_models, steer=20, num_samples=800, duration=10.0, n_std=2.0)


# Section 4 (straight) and Section 5 (steer) datasets for plotting trajectories
files_and_data_2_10_straight = [
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

files_and_data_2_10_steer = [
    ["robot_data_50_5_10_02_26_21_41_57.pkl", 43.5 / 100, -0.5],
    ["robot_data_50_5_10_02_26_21_48_14.pkl", 44 / 100, -0.75],
    ["robot_data_50_5_10_02_26_21_51_12.pkl", 44.5 / 100, -0.80],
    ["robot_data_50_10_10_02_26_21_53_44.pkl", 41.5 / 100, -2.5],
    ["robot_data_50_10_10_02_26_21_55_31.pkl", 39.5 / 100, -2.5],
    ["robot_data_50_10_10_02_26_21_57_15.pkl", 41.5 / 100, -2.6],
    ["robot_data_50_15_10_02_26_22_00_12.pkl", 37.5 / 100, -2.9375],
    ["robot_data_50_15_10_02_26_22_05_05.pkl", 38 / 100, -3.0625],
    ["robot_data_50_15_10_02_26_22_07_37.pkl", 38 / 100, -3.0],
    ["robot_data_50_20_10_02_26_22_09_20.pkl", 34 / 100, -3.5],
    ["robot_data_50_20_10_02_26_22_10_41.pkl", 33 / 100, -3.475],
    ["robot_data_50_20_10_02_26_22_12_35.pkl", 33 / 100, -3.4375],
    ["robot_data_50_-5_10_02_26_22_15_39.pkl", 43.5 / 100, 1.75],
    ["robot_data_50_-5_10_02_26_22_21_13.pkl", 44 / 100, 1.625],
    ["robot_data_50_-5_10_02_26_22_23_06.pkl", 44 / 100, 1.75],
    ["robot_data_50_-10_10_02_26_22_25_45.pkl", 38.5 / 100, 2.625],
    ["robot_data_50_-10_10_02_26_22_30_50.pkl", 41 / 100, 2.875],
    ["robot_data_50_-10_10_02_26_22_33_17.pkl", 38.5 / 100, 2.625],
    ["robot_data_50_-15_10_02_26_22_34_38.pkl", 36.5 / 100, 3.25],
    ["robot_data_50_-15_10_02_26_22_35_51.pkl", 37 / 100, 3.5],
    ["robot_data_50_-15_10_02_26_22_37_01.pkl", 37.5 / 100, 3.625],
    ["robot_data_50_-20_10_02_26_22_39_15.pkl", 30.5 / 100, 2.9375],
    ["robot_data_50_-20_10_02_26_22_41_07.pkl", 30 / 100, 2.69],
    ["robot_data_50_-20_10_02_26_22_42_49.pkl", 30 / 100, 2.875],
]

# Plot predicted trajectories with measured endpoints for sections 4 and 5
if False:
    plot_trajs_with_measured_endpoints(
        files_and_data_2_10_straight,
        "./data_2_10/",
        "Section 4: Straight Trials (Predicted vs Measured Endpoints)",
        "section4_straight_trajs.png",
    )
    plot_trajs_with_measured_endpoints(
        files_and_data_2_10_steer,
        "./data_steer_2_10/",
        "Section 5: Steering Trials (Predicted vs Measured Endpoints)",
        "section5_steer_trajs.png",
    )
