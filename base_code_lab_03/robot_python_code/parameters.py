# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.0.200" # Put your laptop computer's IP here 199
arduinoIP = "192.168.0.199" # Put your arduino's IP here 200
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
video_device="/dev/video4"
camera_id = 4
marker_length = 0.081756
camera_matrix = np.array(
    [
        [1.03300891e03, 0.00000000e00, 6.04854445e02],
        [0.00000000e00, 1.03249117e03, 3.48291072e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float32,
)
dist_coeffs = np.array(
    [-3.51375985e-01, 2.19285665e-02, 1.32358605e-04, -3.95840024e-05, 8.39202722e-02],
    dtype=np.float32,
)


# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 10000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100