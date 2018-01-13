import sys
import os
import glob
import joblib
import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from imutils.face_utils import FaceAligner

n_upsample = 1
output_width = 160

image_dir = sys.argv[1]

output_dir = os.path.join("image_processed/", image_dir)
os.system('mkdir -p ' + output_dir)

IMAGES = glob.glob(os.path.join(image_dir, "*"))
f_shape_predictor = "models/shape_predictor_68_face_landmarks.dat"

if not os.path.exists(output_dir):
    os.system('mkdir -p {}'.format(output_dir))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f_shape_predictor)

def process_image(f_img, skip_if_exists=False):
    '''
    Finds the largest face detected and saves it to disk.
    Writes an empty file if no face is found.
    '''
    f_img_out = os.path.join(output_dir, os.path.basename(f_img))
    item = {"f_img_in":f_img, "f_img_out":f_img_out}
    
    if skip_if_exists and os.path.exists(f_img_out):
        return item

    image = cv2.imread(f_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects, scores, robots = detector.run(gray, upsample_num_times=n_upsample)
    item["faces_detected"] = len(rects)

    # If no faces are detected, throw away the sample!
    if not rects:
        with open(f_img_out, 'w'):
            pass
        return item

    # Find and save the largest image
    sizes = [(r.right() - r.left()) * (r.bottom() - r.top()) for r in rects]
    idx = np.argmax(sizes)

    face = FaceAligner(predictor, desiredFaceWidth=output_width)
    img_out = face.align(image, gray, rects[idx])
    cv2.imwrite(f_img_out, img_out)

    item["best_score"] = np.array(scores)[idx]
    item["subdetector_match"] = np.array(robots)[idx]

    #print "Saved {} {:0.4f}".format(f_img_out, item["best_score"])
    return item



if __name__ == "__main__":
    with joblib.Parallel(-1) as MP:
        func = joblib.delayed(process_image)
        data = [x for x in MP(func(x) for x in tqdm(IMAGES))]

    f_csv_dir = os.path.join("results", image_dir)
    os.system('mkdir -p ' + f_csv_dir)
    f_csv = os.path.join(f_csv_dir, "face_detection.csv")
    df = pd.DataFrame(data).sort_values("f_img_in").set_index("f_img_in")
    df.to_csv(f_csv)

    x = df.best_score
    print "Fraction of frames w/o a face: {:0.4f}".format(pd.isnull(x).mean())
    print "Average detection quality    : {:0.4f}".format(x.mean())
