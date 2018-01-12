import sys
import os
import glob
import joblib
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from imutils.face_utils import FaceAligner


image_dir = sys.argv[1]

output_dir = os.path.join("image_processed/", image_dir)
os.system('mkdir -p ' + output_dir)

IMAGES = glob.glob(os.path.join(image_dir, "*"))
f_shape_predictor = "models/shape_predictor_68_face_landmarks.dat"

if not os.path.exists(output_dir):
    os.system('mkdir -p {}'.format(output_dir))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f_shape_predictor)

def process_image(f_img):
    '''
    Finds the largest face detected and saves it to disk.
    Writes an empty file if no face is found.
    '''

    f_img_out = os.path.join(output_dir, os.path.basename(f_img))
    if os.path.exists(f_img_out):
        return None

    image = cv2.imread(f_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    # If no faces are detected, throw away the sample!
    if not rects:
        with open(f_img_out, 'w'):
            pass
        return None

    # Find and save the largest image
    sizes = [(r.right() - r.left()) * (r.bottom() - r.top()) for r in rects]
    idx = np.argmax(sizes)

    face = FaceAligner(predictor, desiredFaceWidth=160)
    img_out = face.align(image, gray, rects[idx])

    cv2.imwrite(f_img_out, img_out)
    print "Saved", f_img_out
    return f_img_out


if __name__ == "__main__":
    with joblib.Parallel(-1) as MP:
        func = joblib.delayed(process_image)
        for _ in MP(func(x) for x in tqdm(IMAGES)):
            pass
