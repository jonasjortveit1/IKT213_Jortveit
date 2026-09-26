"""IKT213 - Lab assignment 4: Harris hjørnedeteksjon og bildejustering (SIFT + FLANN)."""

import os          # filstier og mapper
import cv2         # bildeoperasjonene
import numpy as np # maskering og homografi
from PIL import Image   # setter utbildene sammen til én PDF

OUTPUT_DIR = "solutions"


# Lagrer bildet og skriver ut størrelsen
def save(name, image):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, image)
    print(f"saved {path}  {image.shape}")


# 1: Harris, marker hjørner i rødt
def harris_corner_detection(reference_image):
    gray = np.float32(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))   # cornerHarris krever float32
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)             # R = det(M) - k*trace(M)^2
    dst = cv2.dilate(dst, None)                                            # gjør markeringene synlige

    marked = reference_image.copy()
    marked[dst > 0.3 * dst.max()] = (0, 0, 255)                           # terskel på R, rødt i BGR
    return marked


# 2: SIFT + FLANN, gir det justerte bildet og matchebildet
def align_images(image_to_align, reference_image, max_features, good_match_percent):
    gray_1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=max_features)                 # 0 = ingen grense
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray_2, None)
    print(f"nøkkelpunkter: {len(keypoints_1)} og {len(keypoints_2)}")

    index_params = dict(algorithm=1, trees=5)                      # KD-tre, passer SIFT
    flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)    # to beste per punkt

    good = [m for m, n in matches if m.distance < good_match_percent * n.distance]   # Lowes ratio-test
    print(f"gode matcher: {len(good)} av {len(matches)}")
    if len(good) < 10:
        raise RuntimeError("For få gode matcher til å regne ut en homografi.")

    good = sorted(good, key=lambda x: x.distance)                  # best først
    match_image = cv2.drawMatches(image_to_align, keypoints_1,
                                  reference_image, keypoints_2, good[:50], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    points_1 = np.float32([keypoints_1[m.queryIdx].pt for m in good])   # i bildet som skal justeres
    points_2 = np.float32([keypoints_2[m.trainIdx].pt for m in good])   # i referansebildet

    h, mask = cv2.findHomography(points_1, points_2, cv2.RANSAC, 5.0)   # RANSAC kaster ut feilmatcher
    print(f"inliers: {int(mask.sum())} av {len(good)}")

    height, width = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, h, (width, height))   # legg bildet over referansen
    return aligned, match_image


# Setter utbildene sammen til PDF-en som skal leveres
def make_pdf(names, pdf_name="assignment_4.pdf"):
    pages = [Image.open(os.path.join(OUTPUT_DIR, n)).convert("RGB") for n in names]
    path = os.path.join(OUTPUT_DIR, pdf_name)
    try:
        pages[0].save(path, save_all=True, append_images=pages[1:])
        print(f"saved {path}  ({len(pages)} sider)")
    except KeyError:                                # Pillow uten JPEG-koder
        print("Pillow mangler JPEG-koder, lag PDF-en manuelt fra PNG-filene.")


def main():
    reference = cv2.imread("reference_img.png")     # referansebildet
    to_align = cv2.imread("align_this.jpg")         # bildet som skal justeres

    for name, img in (("reference_img.png", reference), ("align_this.jpg", to_align)):
        if img is None:                             # imread gir None, ikke feilmelding
            raise FileNotFoundError(f"{name} not found. Put it next to main.py.")

    save("harris.png", harris_corner_detection(reference))          # side 1

    aligned, matches = align_images(to_align, reference, 0, 0.7)    # ingen grense, ratio 0,7
    save("aligned.png", aligned)                                    # side 2
    save("matches.png", matches)                                    # side 3

    make_pdf(["harris.png", "aligned.png", "matches.png"])


if __name__ == "__main__":
    main()
