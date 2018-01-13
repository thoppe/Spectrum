import os
import sys

image_dir = sys.argv[1]
label = "drag_queen"

#os.system('python segment_faces.py {}'.format(image_dir))
#os.system('python predict_images.py {}'.format(image_dir))
os.system('python render_animations.py {} {}'.format(image_dir, label))
