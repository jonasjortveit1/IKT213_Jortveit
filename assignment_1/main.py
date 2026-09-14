import os
import cv2
import time


def print_image_information(image):
    height, width, channels = image.shape
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", image.size)
    print("data type:", image.dtype)


def open_camera(camera_index=0):
    """Try DirectShow first, then Media Foundation."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cam = cv2.VideoCapture(camera_index, backend)
        if cam.isOpened():
            return cam
        cam.release()
    raise RuntimeError("Could not open the camera.")


def measure_fps(cam, frames=120, warmup=30):
    for _ in range(warmup):          # la eksponering og fokus stabilisere seg
        cam.read()
    start = time.perf_counter()
    counted = 0
    for _ in range(frames):
        ok, _frame = cam.read()
        if not ok:
            break
        counted += 1
    elapsed = time.perf_counter() - start
    return counted / elapsed if elapsed > 0 else 0

def save_camera_information(output_path="solutions/camera_outputs.txt", camera_index=0):
    cam = open_camera(camera_index)

    fps = cam.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:                  # driver did not report it
        fps = measure_fps(cam)

    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam.release()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"fps: {round(fps)}\n")
        f.write(f"height: {int(height)}\n")
        f.write(f"width: {int(width)}\n")

    print(f"Wrote {output_path}: fps {round(fps)}, height {int(height)}, width {int(width)}")

def main():
    image = cv2.imread("iris-1.jpg")
    if image is None:
        raise FileNotFoundError("Image not found. Check the file name and that it is in the project folder.")
    print_image_information(image)
    save_camera_information()


if __name__ == "__main__":
    main()