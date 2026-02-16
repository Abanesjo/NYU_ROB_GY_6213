# External Libraries
import math
import random

# This class is an example structure for implementing your motion model.
class MyMotionModel:

    # Constructor, change as you see fit.
    def __init__(self, initial_state = [0, 0, 0], last_encoder_count = 0):
        #State Structure: 
        #state[0] = x
        #state[1] = y
        #state[2] = theta (heading)
        self.state = initial_state
        self.last_encoder_count = last_encoder_count

        #constants
        self.ticks_per_meter = 3481.84
        self.meters_per_tick = 1 / self.ticks_per_meter
        self.distance_variance = 6.23263e-05
        
        self.drivetrain_length = 0.140

    def get_distance_travelled(self, encoder_counts):
        s = self.meters_per_tick * encoder_counts
        return s

    def get_variance_distance_travelled(self, encoder_counts):
        return self.distance_variance    

    def get_rotational_velocity(self, v, steering_angle):
        d_theta = (1 / self.drivetrain_length) * v * math.tan(steering_angle)
        return d_theta
        
    def get_variance_rotational_velocity(self):
        pass

    # This is the key step of your motion model, which implements x_t = f(x_{t-1}, u_t)

    #Provided information: 
    #state: (x, y, theta)
    #encoder count
    #steering angle command
    #delta_t

    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        # Add student code here
        old_state = self.state.copy()
        delta_encoder = encoder_counts - self.last_encoder_count
        delta_s = self.get_distance_travelled(delta_encoder)
        omega_s = self.get_variance_distance_travelled(delta_encoder)

        v = delta_s / delta_t
        # sigma_v = omega_s * (1/delta_s)**2

        steering_angle = math.radians(steering_angle_command)

        d_theta = self.get_rotational_velocity(v, steering_angle)

        #theta
        theta = old_state[2] + 0.5 * d_theta * delta_t

        new_state = old_state

        #x
        new_state[0] += delta_s * math.cos(theta)
        #y
        new_state[1] += delta_s * math.sin(theta)
        #theta
        new_state[2] = theta

        self.state = new_state

        self.last_encoder_count = encoder_counts

        return self.state
    
    # This is a great tool to take in data from a trial and iterate over the data to create 
    # a robot trajectory in the global frame, using your motion model.
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
    def generate_simulated_traj(self, duration):
        delta_t = 0.1
        t_list = []
        x_list = []
        y_list = []
        theta_list = []
        t = 0
        encoder_counts = 0
        while t < duration:

            t += delta_t 
        return t_list, x_list, y_list, theta_list
            