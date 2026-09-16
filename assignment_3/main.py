"""IKT213 - Lab assignment 3: kantdeteksjon, template matching og pyramider."""

import os          # filstier og mapper
import cv2         # bildeoperasjonene
import numpy as np # terskling av matchekartet

OUTPUT_DIR = "solutions"


# Lagrer bildet og skriver ut størrelsen
def save(name, image):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, image)
    print(f"saved {path}  {image.shape}")


# 1: Sobel, gradient i x og y samtidig
def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # Sobel jobber på én kanal
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)             # glatt først, ellers dominerer støyen
    sobel = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=1, ksize=1)   # derivert i begge retninger
    return cv2.convertScaleAbs(sobel)                       # tilbake til uint8 for lagring


# 2: Canny, tynne kanter med to terskler
def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)             # steg 1 i Canny
    return cv2.Canny(blurred, threshold_1, threshold_2)     # gradient, NMS og hysterese


# 3: finn alle steder malen passer
def template_match(image, template):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # matching krever gråtoner
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape                              # malens høyde og bredde

    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)  # normalisert, tåler lysforskjeller
    marked = image.copy()                                   # tegn på en kopi, ikke originalen

    for pt in zip(*np.where(result >= 0.9)[::-1]):          # alle treff over terskel, som (x, y)
        cv2.rectangle(marked, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)   # rød i BGR

    return marked


# 4: zoom inn eller ut med bildepyramide
def resize(image, scale_factor: int, up_or_down: str):
    steps = int(np.log2(scale_factor))                      # pyrUp/pyrDown gjør faktor 2 per steg
    if 2 ** steps != scale_factor:
        raise ValueError("scale_factor must be a power of two")

    result = image
    for _ in range(steps):
        if up_or_down == "up":
            result = cv2.pyrUp(result)                      # doble størrelsen
        elif up_or_down == "down":
            result = cv2.pyrDown(result)                    # glatt og halver, unngår aliasing
        else:
            raise ValueError('up_or_down must be "up" or "down"')
    return result


def main():
    lambo = cv2.imread("lambo.png")                         # bilde til oppgave 1 og 2
    shapes = cv2.imread("shapes-1.png")                       # bilde til oppgave 3
    template = cv2.imread("shapes_template.jpg")            # malen til oppgave 3

    for name, img in (("lambo.png", lambo), ("shapes.png", shapes), ("shapes_template.jpg", template)):
        if img is None:                                     # imread gir None, ikke feilmelding
            raise FileNotFoundError(f"{name} not found. Put it next to main.py.")

    save("sobel.png", sobel_edge_detection(lambo))                      # 1
    save("canny.png", canny_edge_detection(lambo, 50, 50))              # 2
    save("template_match.png", template_match(shapes, template))        # 3
    save("resized_up.png", resize(lambo, 2, "up"))                      # 4: dobbelt
    save("resized_down.png", resize(lambo, 2, "down"))                  # 4: halvt


if __name__ == "__main__":
    main()
