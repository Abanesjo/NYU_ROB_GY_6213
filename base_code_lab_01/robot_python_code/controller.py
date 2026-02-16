#!/usr/bin/env python3
"""Read controller input and print events to the terminal."""

import sys
import time

try:
    import pygame
except ImportError:
    print("Missing dependency: pygame")
    print("Install with: python3 -m pip install pygame")
    sys.exit(1)


def main() -> None:
    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No controller detected. Connect a controller and run again.")
        pygame.quit()
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Connected controller: {joystick.get_name()}")
    print(
        "axes=%d buttons=%d hats=%d"
        % (
            joystick.get_numaxes(),
            joystick.get_numbuttons(),
            joystick.get_numhats(),
        )
    )
    print("Listening for input events... Press Ctrl+C to exit.")

    try:
        while True:
            for event in pygame.event.get():
                timestamp = time.strftime("%H:%M:%S")
                if event.type == pygame.JOYAXISMOTION:
                    print(
                        f"[{timestamp}] axis {event.axis}: "
                        f"{event.value:+.3f}",
                        flush=True,
                    )
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(
                        f"[{timestamp}] button {event.button}: down",
                        flush=True,
                    )
                elif event.type == pygame.JOYBUTTONUP:
                    print(
                        f"[{timestamp}] button {event.button}: up",
                        flush=True,
                    )
                elif event.type == pygame.JOYHATMOTION:
                    print(
                        f"[{timestamp}] hat {event.hat}: {event.value}",
                        flush=True,
                    )
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
