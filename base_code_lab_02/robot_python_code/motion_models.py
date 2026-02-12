# External Libraries
import math
import random

# Motion Model constants


def variance_distance_travelled_s(distance):
    # NOTE: ensure non-negative variance
    f_ss_coeffs = [-1.59597789e-10, 7.50552794e-07, -4.80881217e-04]
    var_s = f_ss_coeffs[0] * distance**2 + f_ss_coeffs[1] * distance + f_ss_coeffs[2]
    return max(var_s, 0.0)

def distance_travelled_s(delta_encoder_counts):
    f_se_slope = 0.00028189968975046026  # meters / count
    return f_se_slope * delta_encoder_counts

def variance_rotational_velocity_w(steering_angle_command):
    # If you really fitted a poly in steering, use that here.
    # If constant is acceptable, just return a constant.
    f_sw_coeffs = [-3.63538320e-08, -2.04254926e-06, 5.54692150e-05]
    var_w = f_sw_coeffs[0] * steering_angle_command**2 + f_sw_coeffs[1] * steering_angle_command + f_sw_coeffs[2]
    return max(var_w, 0.0)

def rotational_velocity_w(steering_angle_command):
    # f_w_slope = -0.0033860576189776383  # rad/s per steering-unit (at your chosen nominal speed)
    # return f_w_slope * steering_angle_command
    f_w_slope_k_tan_model = -0.18785173321140153
    return f_w_slope_k_tan_model * math.tan(math.radians(steering_angle_command))

def my_motion_model(prev_state, delta_s, w, delta_t):
    x, y, theta = prev_state
    x += delta_s * math.cos(theta)
    y += delta_s * math.sin(theta)
    theta += w * delta_t
    return [x, y, theta]

class MyMotionModel:
    def __init__(self, initial_state, last_encoder_count):
        self.state = initial_state[:]  # copy
        self.last_encoder_count = last_encoder_count

    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        # Compute delta encoder for this timestep
        delta_e = encoder_counts - self.last_encoder_count
        self.last_encoder_count = encoder_counts

        # Convert to delta distance
        delta_s = distance_travelled_s(delta_e)

        # Yaw rate from steering (calibrated at nominal speed)
        w = rotational_velocity_w(steering_angle_command)

        # Unicycle discrete update using delta distance
        self.state = my_motion_model(self.state, delta_s, w, delta_t)

        return self.state

    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]

        self.last_encoder_count = encoder_count_list[0]

        for i in range(1, len(encoder_count_list)):
            delta_t = time_list[i] - time_list[i-1]
            new_state = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

        return x_list, y_list, theta_list


    # Coming soon
    def generate_simulated_traj(self, duration, steering_cmd=10):
        delta_t = 0.1
        t = 0.0
        encoder_counts = 0

        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]

        while t < duration:
            # Nominal motion
            delta_s = 0.05  # meters per step (choose from data)
            w_nominal = rotational_velocity_w(steering_cmd)

            # Noise
            var_s = variance_distance_travelled_s(delta_s)
            var_w = variance_rotational_velocity_w(steering_cmd)

            noisy_s = delta_s + random.gauss(0, math.sqrt(var_s))
            noisy_w = w_nominal + random.gauss(0, math.sqrt(var_w))

            # State update
            self.state[0] += noisy_s * math.cos(self.state[2])
            self.state[1] += noisy_s * math.sin(self.state[2])
            self.state[2] += noisy_w * delta_t

            x_list.append(self.state[0])
            y_list.append(self.state[1])
            theta_list.append(self.state[2])

            t += delta_t

        return x_list, y_list, theta_list

