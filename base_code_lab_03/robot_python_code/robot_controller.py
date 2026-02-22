#!/usr/bin/env python3
"""GUI robot control with optional game controller input."""

# External libraries
import cv2
import math
import matplotlib
import numpy as np
import pygame
import io
from fastapi.responses import StreamingResponse
from time import time
from typing import cast

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from nicegui import ui, app, run

# Local libraries
from robot import Robot
import robot_python_code
import parameters


CONTROL_PERIOD_S = 0.1
STEER_AXIS = 2
SPEED_AXIS = 1
STEER_SCALE = 20
MAX_SPEED = 100

logging_active = False
stream_video = True
video_capture = None


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


def disconnect_controller(joystick):
    if joystick is not None:
        joystick.quit()
    pygame.joystick.quit()
    pygame.quit()


def get_controller_commands(joystick):
    pygame.event.pump()
    speed_axis = -joystick.get_axis(SPEED_AXIS)
    if speed_axis < 0:
        speed_axis = 0.0
    speed_axis = clamp(speed_axis, 0.0, 1.0)
    cmd_speed = int(round(speed_axis * MAX_SPEED))

    steer_axis = clamp(joystick.get_axis(STEER_AXIS), -1.0, 1.0)
    cmd_steering = int(round(steer_axis * STEER_SCALE))

    return cmd_speed, cmd_steering


# Frame converter for the video stream, from OpenCV to a JPEG image
def convert(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image.
    This is a free function (not in a class or inner-function),
    to allow run.cpu_bound to pickle it and send it to a separate process.
    """
    _, imencode_image = cv2.imencode(".jpg", frame)
    return imencode_image.tobytes()


placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
placeholder_bytes = convert(placeholder_frame)


def jpeg_response(jpeg_bytes: bytes) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(jpeg_bytes),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


# Create the connection with a real camera.
def connect_with_camera(camera_id):
    return cv2.VideoCapture(camera_id, cv2.CAP_V4L2)


def update_video(video_image):
    if stream_video:
        video_image.force_reload()


def get_time_in_ms():
    return int(time() * 1000)


@ui.page("/")
def main():
    global logging_active

    robot = Robot()
    joystick = None
    controller_connected = False

    video_capture = None
    use_robot_camera = False

    max_lidar_range = 12
    lidar_angle_res = 2
    num_angles = int(360 / lidar_angle_res)
    lidar_distance_list = []
    lidar_cos_angle_list = []
    lidar_sin_angle_list = []
    for i in range(num_angles):
        lidar_distance_list.append(max_lidar_range)
        lidar_cos_angle_list.append(math.cos(i * lidar_angle_res / 180 * math.pi))
        lidar_sin_angle_list.append(math.sin(i * lidar_angle_res / 180 * math.pi))

    dark = ui.dark_mode()
    dark.value = True

    if stream_video:
        if (
            robot.camera_sensor is not None
            and robot.camera_sensor.cap is not None
            and robot.camera_sensor.cap.isOpened()
        ):
            use_robot_camera = True
            print("Using robot camera sensor for UI stream")
        else:
            video_device = getattr(parameters, "video_device", parameters.camera_id)
            video_capture = connect_with_camera(video_device)
            if not video_capture.isOpened() and video_device != "/dev/video4":
                video_capture.release()
                video_capture = connect_with_camera("/dev/video4")

            if video_capture.isOpened():
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                print(
                    "GUI resolution:",
                    int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            else:
                print("Camera open failed:", video_device)

    @app.get("/video/frame")
    async def grab_video_frame() -> StreamingResponse:
        frame = None
        if stream_video and use_robot_camera:
            frame = getattr(robot.camera_sensor, "last_frame", None)
            if frame is not None:
                frame = frame.copy()

        if frame is None:
            if (
                not stream_video
                or video_capture is None
                or not video_capture.isOpened()
            ):
                return jpeg_response(placeholder_bytes)
            ret, frame = await run.io_bound(video_capture.read)
            if not ret or frame is None:
                return jpeg_response(placeholder_bytes)
        jpeg = await run.cpu_bound(convert, frame)
        return jpeg_response(jpeg)

    def update_lidar_data():
        for i in range(robot.robot_sensor_signal.num_lidar_rays):
            distance_in_mm = robot.robot_sensor_signal.distances[i]
            angle = 360 - robot.robot_sensor_signal.angles[i]
            if distance_in_mm > 20 and abs(angle) < 360:
                index = max(
                    0,
                    min(
                        int(360 / lidar_angle_res - 1),
                        int((angle - (lidar_angle_res / 2)) / lidar_angle_res),
                    ),
                )
                lidar_distance_list[index] = distance_in_mm / 1000

    def set_recording(active):
        nonlocal record_button
        global logging_active
        logging_active = active
        if logging_active:
            record_button.set_text("Save Recording")
        else:
            record_button.set_text("Record")

    def toggle_recording():
        set_recording(not logging_active)

    def update_commands():
        nonlocal joystick, controller_connected

        if robot.running_trial:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time:
                robot.running_trial = False
                speed_switch.value = False
                steering_switch.value = False
                robot.extra_logging = True
                print("End Trial :", delta_time)

        if robot.extra_logging:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time + parameters.extra_trial_log_time:
                set_recording(False)
                robot.extra_logging = False

        if controller_switch.value and controller_connected and joystick is not None:
            return get_controller_commands(joystick)

        cmd_speed = slider_speed.value if speed_switch.value else 0
        cmd_steering_angle = slider_steering.value if steering_switch.value else 0
        return cmd_speed, cmd_steering_angle

    def update_connection_to_robot():
        if udp_switch.value:
            if not robot.connected_to_hardware:
                udp, udp_success = robot_python_code.create_udp_communication(
                    parameters.arduinoIP,
                    parameters.localIP,
                    parameters.arduinoPort,
                    parameters.localPort,
                    parameters.bufferSize,
                )
                if udp_success:
                    robot.setup_udp_connection(udp)
                    robot.connected_to_hardware = True
                    print("Should be set for UDP!")
                else:
                    udp_switch.value = False
                    robot.connected_to_hardware = False
        else:
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False

    def enable_speed():
        d = 0

    def enable_steering():
        d = 0

    def show_lidar_plot():
        with main_plot:
            fig = main_plot.fig
            fig.patch.set_facecolor("black")
            plt.clf()
            plt.style.use("dark_background")
            plt.tick_params(axis="x", colors="lightgray")
            plt.tick_params(axis="y", colors="lightgray")

            for i in range(num_angles):
                distance = lidar_distance_list[i]
                cos_ang = lidar_cos_angle_list[i]
                sin_ang = lidar_sin_angle_list[i]
                x = [distance * cos_ang, max_lidar_range * cos_ang]
                y = [distance * sin_ang, max_lidar_range * sin_ang]
                plt.plot(x, y, "r")
            plt.grid(True)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)

    def show_localization_plot():
        with main_plot:
            fig = main_plot.fig
            fig.patch.set_facecolor("black")
            plt.clf()
            plt.style.use("dark_background")
            plt.tick_params(axis="x", colors="lightgray")
            plt.tick_params(axis="y", colors="lightgray")

            covar_matrix = (
                parameters.covariance_plot_scale
                * robot.extended_kalman_filter.state_covariance[0:2, 0:2]
            )
            x_est = robot.extended_kalman_filter.state_mean[0]
            y_est = robot.extended_kalman_filter.state_mean[1]
            lambda_, v = np.linalg.eig(covar_matrix)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(
                xy=(x_est, y_est),
                alpha=0.5,
                facecolor="red",
                width=lambda_[0],
                height=lambda_[1],
                angle=np.rad2deg(np.arctan2(*v[:, 0][::-1])),
            )
            ax = cast(Axes, plt.gca())
            ax.add_patch(ell)

            plt.plot(x_est, y_est, "ro")

            plt.grid(True)
            plot_range = 1
            plt.xlim(-plot_range, plot_range)
            plt.ylim(-plot_range, plot_range)

    def run_trial():
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        set_recording(True)
        print("Start time:", robot.trial_start_time)

    def update_controller_status():
        status = "connected" if controller_connected else "disconnected"
        controller_status_label.set_text(f"Controller: {status}")
        controller_button.set_text(
            "Disconnect Controller" if controller_connected else "Connect Controller"
        )

    def toggle_controller_connection():
        nonlocal joystick, controller_connected
        if controller_connected:
            disconnect_controller(joystick)
            joystick = None
            controller_connected = False
            controller_switch.value = False
            update_controller_status()
            return

        try:
            joystick = connect_controller()
            controller_connected = True
            update_controller_status()
            ui.notify("Controller connected")
        except RuntimeError as exc:
            joystick = None
            controller_connected = False
            controller_switch.value = False
            update_controller_status()
            ui.notify(str(exc), color="negative")

    def toggle_controller_use(e):
        if e.value and not controller_connected:
            controller_switch.value = False
            ui.notify("Controller not connected", color="warning")

    with ui.card().classes("w-full  items-center"):
        ui.label("ROB-GY - 6213: Robot Navigation & Localization").style(
            "font-size: 24px;"
        )

    with ui.card().classes("w-full"):
        with ui.grid(columns=3).classes("w-full items-center"):
            with ui.card().classes("w-full items-center h-60"):
                if stream_video:
                    video_image = ui.interactive_image("/video/frame").classes(
                        "w-full h-full"
                    )
                else:
                    ui.image("./a_robot_image.jpg").props("height=2")
                    video_image = None
            with ui.card().classes("w-full items-center h-60"):
                main_plot = ui.pyplot(figsize=(3, 3))
            with ui.card().classes("items-center h-60"):
                ui.label("Encoder:").style("text-align: center;")
                encoder_count_label = ui.label("0")
                record_button = ui.button("Record", on_click=lambda: toggle_recording())
                udp_switch = ui.switch("Robot Connect")
                controller_status_label = ui.label("Controller: disconnected")
                controller_button = ui.button(
                    "Connect Controller",
                    on_click=lambda: toggle_controller_connection(),
                )
                controller_switch = ui.switch(
                    "Use Controller", on_change=lambda e: toggle_controller_use(e)
                )
                run_trial_button = ui.button("Run Trial", on_click=lambda: run_trial())

    with ui.card().classes("w-full"):
        with ui.grid(columns=4).classes("w-full"):
            with ui.card().classes("w-full items-center"):
                ui.label("SPEED:").style("text-align: center;")
            with ui.card().classes("w-full items-center"):
                slider_speed = ui.slider(min=0, max=100, value=0)
            with ui.card().classes("w-full items-center"):
                ui.label().bind_text_from(slider_speed, "value").style(
                    "text-align: center;"
                )
            with ui.card().classes("w-full items-center"):
                speed_switch = ui.switch("Enable", on_change=lambda: enable_speed())

    with ui.card().classes("w-full"):
        with ui.grid(columns=4).classes("w-full"):
            with ui.card().classes("w-full items-center"):
                ui.label("STEER:").style("text-align: center;")
            with ui.card().classes("w-full items-center"):
                slider_steering = ui.slider(min=-20, max=20, value=0)
            with ui.card().classes("w-full items-center"):
                ui.label().bind_text_from(slider_steering, "value").style(
                    "text-align: center;"
                )
            with ui.card().classes("w-full items-center"):
                steering_switch = ui.switch(
                    "Enable", on_change=lambda: enable_steering()
                )

    async def control_loop():
        update_connection_to_robot()
        cmd_speed, cmd_steering_angle = update_commands()
        robot.control_loop(cmd_speed, cmd_steering_angle, logging_active)
        encoder_count_label.set_text(str(robot.robot_sensor_signal.encoder_counts))
        # update_lidar_data()
        # show_lidar_plot()
        show_localization_plot()
        update_video(video_image)

    ui.timer(CONTROL_PERIOD_S, control_loop)


ui.run(host="127.0.0.1", port=8080)
