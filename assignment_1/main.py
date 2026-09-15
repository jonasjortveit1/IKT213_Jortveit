"""IKT213 - Lab assignment 1
Skriver ut bildeinformasjon med OpenCV, og lagrer kameradata til fil.
Jonas Jortveit
"""

import os          # lager mapper og setter sammen filstier
import cv2         # OpenCV: alt som har med bilder og kamera å gjøre
import time        # brukes til å ta tiden når vi måler bildefrekvens


# --- Oppgave IV: skriv ut informasjon om bildet ------------------------------
def print_image_information(image):
    # Et bilde er en matrise. shape gir (rader, kolonner, kanaler),
    # altså høyde først, fordi NumPy teller rader før kolonner.
    height, width, channels = image.shape

    print("height:", height)        # antall pikselrader
    print("width:", width)          # antall pikselkolonner
    print("channels:", channels)    # 3 for farge (OpenCV bruker BGR, ikke RGB)
    print("size:", image.size)      # høyde * bredde * kanaler = alle verdiene
    print("data type:", image.dtype)  # uint8 = ett byte per verdi, 0 til 255


# --- Åpner webkameraet ------------------------------------------------------
def open_camera(camera_index=0):
    """Prøver DirectShow først, så Media Foundation."""
    # Windows har to måter å snakke med kameraet på. Noen kameraer svarer bare
    # på den ene, så vi prøver dem i tur og orden.
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cam = cv2.VideoCapture(camera_index, backend)
        if cam.isOpened():          # fikk vi kontakt, er vi ferdige
            return cam
        cam.release()               # ellers slipper vi kameraet og prøver neste
    raise RuntimeError("Could not open the camera.")


# --- Måler bildefrekvens selv når driveren ikke oppgir den ------------------
def measure_fps(cam, frames=120, warmup=30):
    # Kameraet bruker de første bildene på autoeksponering og autofokus.
    # Tar vi tiden med en gang, måler vi oppstarten og ikke den normale farten.
    for _ in range(warmup):         # la eksponering og fokus stabilisere seg
        cam.read()

    start = time.perf_counter()     # presis klokke, bedre enn time.time() her
    counted = 0
    for _ in range(frames):
        ok, _frame = cam.read()     # ok er False hvis kameraet svikter
        if not ok:
            break
        counted += 1                # teller bildene vi faktisk fikk

    elapsed = time.perf_counter() - start   # sekunder brukt
    # Bilder delt på sekunder gir fps. Testen mot 0 unngår deling på null.
    return counted / elapsed if elapsed > 0 else 0


# --- Oppgave V: lagre kameraets fps, høyde og bredde til fil ----------------
def save_camera_information(output_path="solutions/camera_outputs.txt", camera_index=0):
    cam = open_camera(camera_index)

    fps = cam.get(cv2.CAP_PROP_FPS)              # spør driveren om fps
    if fps is None or fps <= 0:                  # driveren oppga det ikke
        fps = measure_fps(cam)                   # da måler vi selv

    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # oppløsningen kameraet leverer
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam.release()                                # viktig: gi kameraet tilbake

    # Lager solutions-mappa hvis den ikke finnes, slik at open() ikke feiler.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # "w" overskriver fila hver gang, så vi ikke får dobbelt opp med linjer.
    with open(output_path, "w") as f:
        f.write(f"fps: {round(fps)}\n")          # oppgaven vil ha hele tall
        f.write(f"height: {int(height)}\n")
        f.write(f"width: {int(width)}\n")

    print(f"Wrote {output_path}: fps {round(fps)}, height {int(height)}, width {int(width)}")


# --- Kjører begge oppgavene -------------------------------------------------
def main():
    image = cv2.imread("iris-1.jpg")   # leser bildet fra mappa programmet kjører i
    if image is None:
        # imread kaster ingen feil, den returnerer None. Derfor sjekker vi selv.
        raise FileNotFoundError("Image not found. Check the file name and that it is in the project folder.")

    print_image_information(image)
    save_camera_information()


# Kjøres bare når fila startes direkte, ikke når den importeres fra en annen fil.
if __name__ == "__main__":
    main()
