from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import cv2
import dlib
import glob, os
import joblib
import sys
from tqdm import tqdm

#image_dir = "raw_images/"
#image_dir = "jenna/out/"
#image_dir = "videos/guy2girl/out/"
#image_dir = "videos/faking/out/"
#image_dir = "videos/eyelid_surgery/out/"
#image_dir = "videos/drag_queen/out/"
#image_dir = "videos/girl2guy/out/"
image_dir = sys.argv[1]

output_dir = os.path.join("image_processed/", image_dir)
os.system('mkdir -p '+output_dir)

IMAGES = glob.glob(os.path.join(image_dir, "*"))
shape_predictor = "models/shape_predictor_68_face_landmarks.dat"

if not os.path.exists(output_dir):
    os.system('mkdir -p {}'.format(output_dir))


def process_image(f_img):

    f_img_out = os.path.join(output_dir, os.path.basename(f_img))
    if os.path.exists(f_img_out):
        return None
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=160)
    image = cv2.imread(f_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    rect_nums = len(rects)
    XY, aligned_images = [], []
    if rect_nums == 0:
        with open(f_img_out,'w'):
            pass
        return None

    # Find and save the largest image
    sizes = [(r.right()-r.left()) * (r.bottom()-r.top()) for r in rects]
    idx = np.argmax(sizes)
    img_out = fa.align(image, gray, r)
    cv2.imwrite(f_img_out, img_out)

    print "Saved", f_img_out
    

func = joblib.delayed(process_image)
with joblib.Parallel(-1) as MP:
    for res in MP(func(x) for x in tqdm(IMAGES)):
        pass


    
