#!/usr/bin/env python3
"""Headless robot control using a game controller and keyboard logging toggle."""

import contextlib
import os
import select
import sys
import termios
import time
import tty

import pygame

import parameters
import robot_python_code


CONTROL_PERIOD_S = 0.1
BUTTON_SPEED_TOGGLE = 10
STEER_AXIS = 2
STEER_SCALE = 20
SPEED_LOW = 0
SPEED_HIGH = 70
LOG_TOGGLE_KEY = "k"


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class KeyPoller:
    """Non-blocking single-key reader for terminal input."""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard toggle requires a terminal (TTY) input.")
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def poll(self):
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            return sys.stdin.read(1)
        return None

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def clamp(value, low, high):
    return max(low, min(high, value))


def connect_controller():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No controller detected.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick


def connect_robot():
    with suppress_output():
        robot = robot_python_code.Robot()
    udp = robot_python_code.UDPCommunication(
        parameters.arduinoIP,
        parameters.localIP,
        parameters.arduinoPort,
        parameters.localPort,
        parameters.bufferSize,
    )
    with suppress_output():
        robot.setup_udp_connection(udp)
    robot.connected_to_hardware = True
    return robot, udp


def disconnect_robot(robot, udp):
    if robot is not None:
        with suppress_output():
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False
            if hasattr(robot, "camera_sensor") and robot.camera_sensor is not None:
                robot.camera_sensor.close()
    if udp is not None:
        udp.UDPServerSocket.close()


def disconnect_controller(joystick):
    if joystick is not None:
        joystick.quit()
    pygame.joystick.quit()
    pygame.quit()


def main():
    robot = None
    udp = None
    joystick = None

    current_speed = SPEED_LOW
    logging_active = False
    prev_button_10 = False

    try:
        joystick = connect_controller()
        robot, udp = connect_robot()

        with KeyPoller() as key_poller:
            while True:
                loop_start = time.perf_counter()

                pygame.event.pump()

                button_10 = bool(joystick.get_button(BUTTON_SPEED_TOGGLE))
                if button_10 and not prev_button_10:
                    if current_speed == SPEED_LOW:
                        current_speed = SPEED_HIGH
                    else:
                        current_speed = SPEED_LOW
                prev_button_10 = button_10

                steer_axis = clamp(joystick.get_axis(STEER_AXIS), -1.0, 1.0)
                cmd_steering = int(round(steer_axis * STEER_SCALE))

                key = key_poller.poll()
                while key is not None:
                    if key == "\x03":
                        raise KeyboardInterrupt
                    if key.lower() == LOG_TOGGLE_KEY:
                        if not logging_active:
                            robot.start_logging([current_speed, cmd_steering])
                            logging_active = True
                            print("Recording started", flush=True)
                        else:
                            saved_filename = robot.save_log()
                            logging_active = False
                            print(f"Recording stopped: {saved_filename}", flush=True)
                    key = key_poller.poll()

                robot.control_loop(current_speed, cmd_steering, logging_active)

                elapsed = time.perf_counter() - loop_start
                if elapsed < CONTROL_PERIOD_S:
                    time.sleep(CONTROL_PERIOD_S - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        if robot is not None and logging_active:
            saved_filename = robot.save_log()
            print(f"Recording stopped: {saved_filename}", flush=True)
        disconnect_robot(robot, udp)
        disconnect_controller(joystick)


if __name__ == "__main__":
    main()
