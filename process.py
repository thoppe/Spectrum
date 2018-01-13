import os
import sys

# python process.py videos/drag_queen/out/ drag_queen

image_dir = sys.argv[1]
label = sys.argv[2]

os.system('python segment_faces.py {}'.format(image_dir))
os.system('python predict_images.py {}'.format(image_dir))
os.system('python render_animations.py {} {}'.format(image_dir, label))
