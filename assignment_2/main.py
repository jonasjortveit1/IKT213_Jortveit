"""IKT213 - Lab assignment 2: bildeoperasjoner og lineære filtre."""

import os          # filstier og mapper
import cv2         # bildeoperasjonene
import numpy as np # tomme arrays til copy og hue_shifted

IMAGE_FILE = "iris-1.jpg"     # ligger ved siden av main.py
OUTPUT_DIR = "solutions"    # alle resultater havner her


# Lagrer bildet og skriver ut størrelsen
def save(name, image):
    os.makedirs(OUTPUT_DIR, exist_ok=True)   # lager mappa om den mangler
    path = os.path.join(OUTPUT_DIR, name)    # riktig sti på alle systemer
    cv2.imwrite(path, image)                 # filendelsen bestemmer formatet
    print(f"saved {path}  {image.shape}")


# 1: kant rundt bildet, speilet fra kanten
def padding(image, border_width):
    return cv2.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,  # topp, bunn, venstre, høyre
        cv2.BORDER_REFLECT,                 # speiling, ikke svart kant
    )


# 2: klipper ut et utsnitt
def crop(image, x_0, x_1, y_0, y_1):
    return image[y_0:y_1, x_0:x_1]          # NumPy: rad før kolonne


# 3: skalerer til ny størrelse
def resize(image, width, height):
    return cv2.resize(image, (width, height))   # resize tar (bredde, høyde)


# 4: manuell kopi, piksel for piksel, uten cv2
def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):                 # rad
        for x in range(width):              # piksel
            for c in range(channels):       # B, G, R hver for seg
                emptyPictureArray[y, x, c] = image[y, x, c]
    return emptyPictureArray                # løkkene er trege, det er ventet


# 5: farge til gråtoner
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # én kanal ut


# 6: BGR til HSV
def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # fargetone, metning, lyshet


# 7: legger en konstant til alle fargeverdier
def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                value = int(image[y, x, c]) + hue        # int unngår uint8-overflyt
                emptyPictureArray[y, x, c] = max(0, min(255, value))   # klipp til 0-255
    return emptyPictureArray


# 8: gaussisk uskarphet, lavpassfilter
def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)  # sigma fra ksize


# 9: roterer 90 med klokka eller 180
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError("rotation_angle must be 90 or 180")     # andre vinkler krever warpAffine


# Kjører alle ni oppgavene
def main():
    image = cv2.imread(IMAGE_FILE)          # leter i mappa programmet kjører fra
    if image is None:                       # imread gir None, ikke feilmelding
        raise FileNotFoundError(f"{IMAGE_FILE} not found. Put it next to main.py.")

    height, width, channels = image.shape
    print(f"{IMAGE_FILE}: {width} x {height}, {channels} channels")

    save("padding.png", padding(image, 100))                 # 1: 100 px kant

    # 2: 200 px fra venstre og topp, 130 px fra høyre og bunn
    save("cropped.png", crop(image, 200, width - 130, 200, height - 130))

    save("resized.png", resize(image, 200, 200))             # 3: fast 200x200

    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)   # tomt array
    save("copied.png", copy(image, emptyPictureArray))       # 4

    save("grayscale.png", grayscale(image))                  # 5
    save("hsv.png", hsv(image))                              # 6

    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)   # nytt, ikke gjenbruk
    save("hue_shifted.png", hue_shifted(image, emptyPictureArray, 50))  # 7: +50

    save("smoothed.png", smoothing(image))                   # 8
    save("rotated_90.png", rotation(image, 90))              # 9
    save("rotated_180.png", rotation(image, 180))            # 9


if __name__ == "__main__":                  # kjører bare ved direkte start
    main()
