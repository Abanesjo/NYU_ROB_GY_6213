# generate_aruco.py
import cv2 as cv
import cv2.aruco as aruco
import numpy as np

def main():
    marker_id = 23
    marker_px = 1000   # high resolution for clean printing
    border_bits = 1

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    marker = aruco.generateImageMarker(
        dictionary,
        marker_id,
        marker_px,
        borderBits=border_bits
    )

    # Add white margin (very important for detection)
    margin = 300
    canvas = np.ones((marker_px + 2*margin,
                      marker_px + 2*margin), dtype=np.uint8) * 255
    canvas[margin:margin+marker_px,
           margin:margin+marker_px] = marker

    filename = f"aruco_6x6_250_id{marker_id}.png"
    cv.imwrite(filename, canvas)
    print("Saved:", filename)

if __name__ == "__main__":
    main()
