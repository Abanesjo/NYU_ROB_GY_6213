# External libraries
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Local libraries
import parameters
import data_handling

# Motion Model
from motion_models import MyMotionModel


def rpy_to_R(roll, pitch, yaw):
    c_roll = math.cos(roll)
    s_roll = math.sin(roll)
    c_pitch = math.cos(pitch)
    s_pitch = math.sin(pitch)
    c_yaw = math.cos(yaw)
    s_yaw = math.sin(yaw)

    Rz = np.array(
        [
            [c_yaw, -s_yaw, 0.0],
            [s_yaw, c_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    Ry = np.array(
        [
            [c_pitch, 0.0, s_pitch],
            [0.0, 1.0, 0.0],
            [-s_pitch, 0.0, c_pitch],
        ]
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c_roll, -s_roll],
            [0.0, s_roll, c_roll],
        ]
    )
    return Rz @ Ry @ Rx


# Main class
class ExtendedKalmanFilter:
    # Note: We are overriding the input vector. By default, it is encoder count and steering. We are replacing it with linear velocity and steering angle (steering command is not necessarily the same as the actual steering angle)

    def __init__(self, x_0, Sigma_0, encoder_counts_0):
        self.state_mean = x_0
        self.state_covariance = Sigma_0
        self.predicted_state_mean = [0, 0, 0]
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts = encoder_counts_0
        self.motion_model = MyMotionModel(x_0, encoder_counts_0)
        self.drivetrain_length = self.motion_model.drivetrain_length
        self.t_odom_cam = np.array(
            [parameters.tripod_x, parameters.tripod_y, parameters.tripod_z], dtype=float
        )
        R_odom_tripod = rpy_to_R(
            parameters.tripod_roll, parameters.tripod_pitch, parameters.tripod_yaw
        )
        R_tripod_cam = rpy_to_R(
            parameters.camera_roll, parameters.camera_pitch, parameters.camera_yaw
        )
        self.R_odom_cam = R_odom_tripod @ R_tripod_cam
        self.r1 = self.R_odom_cam[0, :]
        self.r2 = self.R_odom_cam[1, :]

    # -------------------------------Prediction--------------------------------------#

    # The nonlinear transition equation that provides new states from past states
    def g_function(self, x_tm1, u_t, delta_t):
        v = u_t[0]
        phi = u_t[1]
        L = self.drivetrain_length

        delta_theta = (v / L) * math.tan(phi) * delta_t
        theta = x_tm1[2] + 0.5 * delta_theta

        delta_x = v * math.cos(theta) * delta_t
        delta_y = v * math.sin(theta) * delta_t

        x_t = np.array(x_tm1) + np.array([delta_x, delta_y, delta_theta])
        return x_t

    # This function returns the R_t matrix which contains transition function covariance terms.
    def get_R(self, delta_t=0.1):
        sigma_v = self.motion_model.get_variance_linear_velocity(delta_t)
        sigma_phi = self.motion_model.get_variance_steering_angle()

        R = np.diag([sigma_v, sigma_phi])

        return R

    # This function returns a matrix with the partial derivatives dg/dx
    # g outputs x_t, y_t, theta_t, and we take derivatives wrt inputs x_tm1, y_tm1, theta_tm1
    def get_G_x(self, x_tm1, u_t, delta_t):
        L = self.drivetrain_length
        v = u_t[0]
        phi = u_t[1]
        theta = x_tm1[2] + 0.5 * (v / L) * math.tan(phi) * delta_t

        G_x = np.array(
            [
                [1, 0, -1 * delta_t * v * math.sin(theta)],
                [0, 1, delta_t * v * math.cos(theta)],
                [0, 0, 1],
            ]
        )

        return G_x

    # This function returns a matrix with the partial derivatives dg/du
    def get_G_u(self, x_tm1, u_t, delta_t):
        L = self.drivetrain_length
        v = u_t[0]
        phi = u_t[1]
        theta = x_tm1[2] + 0.5 * (v / L) * math.tan(phi) * delta_t

        G_u = np.array(
            [
                [delta_t * math.cos(theta), 0],
                [delta_t * math.sin(theta), 0],
                [
                    (delta_t / L) * math.tan(phi),
                    delta_t * (v / L) * (1 / math.cos(phi)) ** 2,
                ],
            ]
        )

        return G_u

        # Set the EKF's predicted state mean and covariance matrix

    def prediction_step(self, u_t, delta_t):
        x_tm1 = self.state_mean
        sigma_tm1 = self.state_covariance
        G_x = self.get_G_x(x_tm1, u_t, delta_t)
        G_u = self.get_G_u(x_tm1, u_t, delta_t)
        R = self.get_R(delta_t)

        x_t = self.g_function(x_tm1, u_t, delta_t)
        sigma_t = G_x @ sigma_tm1 @ G_x.T + G_u @ R @ G_u.T

        self.predicted_state_mean = x_t
        self.predicted_state_covariance = sigma_t
        self.state_mean = x_t
        self.state_covariance = sigma_t
        return x_t, sigma_t

    # -------------------------------Correction--------------------------------------#

    # The nonlinear measurement function
    def get_h_function(self, z_t):
        z_x, z_y, z_z, z_alpha, z_beta, z_theta = z_t
        z_p = np.array([z_x, z_y, z_z], dtype=float)

        x_x = self.t_odom_cam[0] + np.dot(self.r1, z_p)
        x_y = self.t_odom_cam[1] + np.dot(self.r2, z_p)

        c_zbeta = math.cos(z_beta)
        s_zbeta = math.sin(z_beta)
        c_ztheta = math.cos(z_theta)
        s_ztheta = math.sin(z_theta)
        m1 = np.array([c_zbeta * c_ztheta, c_zbeta * s_ztheta, -s_zbeta])

        gamma_1 = np.dot(self.r1, m1)
        gamma_2 = np.dot(self.r2, m1)
        x_theta = math.atan2(gamma_2, gamma_1)

        return np.array([x_x, x_y, x_theta])

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self):
        return parameters.Q6

    # This function returns a matrix with the partial derivatives dh_t/dx_t
    def get_H(self, z_t):
        _, _, _, _, z_beta, z_theta = z_t
        c_zbeta = math.cos(z_beta)
        s_zbeta = math.sin(z_beta)
        c_ztheta = math.cos(z_theta)
        s_ztheta = math.sin(z_theta)

        m1 = np.array([c_zbeta * c_ztheta, c_zbeta * s_ztheta, -s_zbeta])
        m1_zbeta = np.array([-s_zbeta * c_ztheta, -s_zbeta * s_ztheta, -c_zbeta])
        m1_ztheta = np.array([-c_zbeta * s_ztheta, c_zbeta * c_ztheta, 0.0])

        gamma_1 = np.dot(self.r1, m1)
        gamma_2 = np.dot(self.r2, m1)
        gamma_1_zbeta = np.dot(self.r1, m1_zbeta)
        gamma_2_zbeta = np.dot(self.r2, m1_zbeta)
        gamma_1_ztheta = np.dot(self.r1, m1_ztheta)
        gamma_2_ztheta = np.dot(self.r2, m1_ztheta)

        denom = gamma_1 * gamma_1 + gamma_2 * gamma_2
        dtheta_dzbeta = (gamma_1 * gamma_2_zbeta - gamma_2 * gamma_1_zbeta) / denom
        dtheta_dztheta = (gamma_1 * gamma_2_ztheta - gamma_2 * gamma_1_ztheta) / denom

        H = np.array(
            [
                [self.r1[0], self.r1[1], self.r1[2], 0.0, 0.0, 0.0],
                [self.r2[0], self.r2[1], self.r2[2], 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, dtheta_dzbeta, dtheta_dztheta],
            ]
        )
        return H

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        return

    # -------------------------------Update--------------------------------------#

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t):
        u_t = np.asarray(u_t)
        if z_t is not None:
            z_t = np.asarray(z_t)

        # Prediction
        self.prediction_step(u_t, delta_t)

        # Correction (optional)
        if z_t is not None:
            self.correction_step(z_t)

        return self.state_mean, self.state_covariance


class KalmanFilterPlot:
    def __init__(self):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig

    def update(self, state_mean, state_covaraiance):
        plt.clf()

        # Plot covariance ellipse
        lambda_, v = np.linalg.eig(state_covaraiance)
        lambda_ = np.sqrt(lambda_)
        xy = (state_mean[0], state_mean[1])
        angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))
        ell = Ellipse(
            xy,
            alpha=0.5,
            facecolor="red",
            width=lambda_[0],
            height=lambda_[1],
            angle=angle,
        )
        ax = self.fig.gca()
        ax.add_artist(ell)

        # Plot state estimate
        plt.plot(state_mean[0], state_mean[1], "ro")
        plt.plot(
            [state_mean[0], state_mean[0] + self.dir_length * math.cos(state_mean[2])],
            [state_mean[1], state_mean[1] + self.dir_length * math.sin(state_mean[2])],
            "r",
        )
        plt.xlabel("X(m)")
        plt.ylabel("Y(m)")
        plt.axis([-0.25, 2, -1, 1])
        plt.grid()
        plt.draw()
        plt.pause(0.1)


def rvec_to_rpy(rvec):
    rvec = np.array(rvec, dtype=float).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


# Code to run your EKF offline with a data file.
def offline_efk_prediction():
    # Get data to filter
    # filename = "./data_validation/cw.pkl"
    filename = "./data_validation/straight.pkl"
    ekf_data = data_handling.get_file_data_for_prediction(filename)

    # Instantiate PF with no initial guess
    x_0 = [0.0, 0.0, 0.0]
    Sigma_0 = parameters.I3
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    extended_kalman_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    motion_model = MyMotionModel(x_0, encoder_counts_0)

    # Create plotting tool for ekf
    kalman_filter_plot = KalmanFilterPlot()

    last_encoder_count = encoder_counts_0
    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t - 1][0]  # time step size

        # u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        # z_t = np.array([row[3][0],row[3][1],row[3][5]]) # camera_sensor_signal

        v = motion_model.get_linear_velocity(
            row[2].encoder_counts - last_encoder_count, delta_t
        )
        phi = motion_model.get_steering_angle(row[2].steering)

        u_t = np.array([v, phi])
        z_t = None

        last_encoder_count = row[2].encoder_counts

        # Run the EKF for a time step
        extended_kalman_filter.update(u_t, z_t, delta_t)
        kalman_filter_plot.update(
            extended_kalman_filter.state_mean,
            extended_kalman_filter.state_covariance[0:2, 0:2],
        )


# Code to run your EKF offline with a data file (prediction + correction).
def offline_efk():
    # Get data to filter
    filename = "./data/robot_data_68_0_06_02_26_17_12_19.pkl"
    ekf_data = data_handling.get_file_data_for_kf(filename)

    # Instantiate PF with no initial guess
    roll_0, pitch_0, yaw_0 = rvec_to_rpy(ekf_data[0][3][3:6])
    x_0 = [ekf_data[0][3][0] + 0.5, ekf_data[0][3][1], yaw_0]
    Sigma_0 = parameters.I3
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    extended_kalman_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    motion_model = MyMotionModel(x_0, encoder_counts_0)

    # Create plotting tool for ekf
    kalman_filter_plot = KalmanFilterPlot()

    last_encoder_count = encoder_counts_0
    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t - 1][0]  # time step size

        v = motion_model.get_linear_velocity(
            row[2].encoder_counts - last_encoder_count, delta_t
        )
        phi = motion_model.get_steering_angle(row[2].steering)

        u_t = np.array([v, phi])
        roll, pitch, yaw = rvec_to_rpy(row[3][3:6])
        z_t = np.array([row[3][0], row[3][1], row[3][2], roll, pitch, yaw])

        last_encoder_count = row[2].encoder_counts

        # Run the EKF for a time step
        extended_kalman_filter.update(u_t, z_t, delta_t)
        kalman_filter_plot.update(
            extended_kalman_filter.state_mean,
            extended_kalman_filter.state_covariance[0:2, 0:2],
        )


####### MAIN #######
if __name__ == "__main__":
    offline_efk_prediction()
