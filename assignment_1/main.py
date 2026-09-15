"""IKT213 - Lab assignment 1: bildeinformasjon og kameradata."""

import os      # filstier og mapper
import cv2     # bilde og kamera
import time    # tidtaking for fps


# Oppgave IV: skriver ut informasjon om bildet
def print_image_information(image):
    height, width, channels = image.shape   # shape: rader, kolonner, kanaler
    print("height:", height)                # antall rader
    print("width:", width)                  # antall kolonner
    print("channels:", channels)            # 3 = farge, BGR i OpenCV
    print("size:", image.size)              # høyde * bredde * kanaler
    print("data type:", image.dtype)        # uint8: 0-255


# Åpner kameraet med den backenden som svarer
def open_camera(camera_index=0):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):   # noen kameraer svarer bare på én
        cam = cv2.VideoCapture(camera_index, backend)
        if cam.isOpened():                  # kontakt, bruk denne
            return cam
        cam.release()                       # ellers prøv neste
    raise RuntimeError("Could not open the camera.")


# Måler fps selv, brukes når driveren ikke oppgir den
def measure_fps(cam, frames=120, warmup=30):
    for _ in range(warmup):                 # hopp over autoeksponering og fokus
        cam.read()

    start = time.perf_counter()             # presis klokke
    counted = 0
    for _ in range(frames):
        ok, _frame = cam.read()             # ok = False hvis kameraet svikter
        if not ok:
            break
        counted += 1                        # teller bilder som faktisk kom

    elapsed = time.perf_counter() - start   # sekunder brukt
    return counted / elapsed if elapsed > 0 else 0   # bilder per sekund


# Oppgave V: lagrer fps, høyde og bredde til fil
def save_camera_information(output_path="solutions/camera_outputs.txt", camera_index=0):
    cam = open_camera(camera_index)

    fps = cam.get(cv2.CAP_PROP_FPS)         # spør driveren
    if fps is None or fps <= 0:             # driveren oppga ikke fps
        fps = measure_fps(cam)              # mål i stedet

    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)   # oppløsning fra kameraet
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam.release()                           # frigjør kameraet

    os.makedirs(os.path.dirname(output_path), exist_ok=True)   # lager solutions
    with open(output_path, "w") as f:       # "w" overskriver fila
        f.write(f"fps: {round(fps)}\n")     # oppgaven vil ha hele tall
        f.write(f"height: {int(height)}\n")
        f.write(f"width: {int(width)}\n")

    print(f"Wrote {output_path}: fps {round(fps)}, height {int(height)}, width {int(width)}")


# Kjører begge oppgavene
def main():
    image = cv2.imread("iris-1.jpg")        # leter i mappa programmet kjører fra
    if image is None:                       # imread gir None, ikke feilmelding
        raise FileNotFoundError("Image not found. Check the file name and that it is in the project folder.")
    print_image_information(image)          # oppgave IV
    save_camera_information()               # oppgave V


if __name__ == "__main__":                  # kjører bare ved direkte start
    main()
