import os
import sys
import glob

# python process.py videos/drag_queen/out/ drag_queen

#image_dir = sys.argv[1]
#label = sys.argv[2]

name = sys.argv[1]
base_image_dir = os.path.join("videos/", name)
assert(os.path.exists(base_image_dir))

image_dir = os.path.join("videos/", name, 'out')
if not os.path.exists(image_dir):
    os.system('mkdir -p {}'.format(image_dir))
    cmd  = 'avconv -r 4 -an -y "{}/%04d.png" -i "{f_video}"'
    f_video = glob.glob(os.path.join(base_image_dir,'*'))
    f_video = [x for x in f_video if '.' in os.path.basename(x)][0]
    cmd = cmd.format(image_dir, f_video=f_video)
    print cmd
    os.system(cmd)
    exit()

os.system('python segment_faces.py {}'.format(image_dir))
os.system('python predict_images.py {}'.format(image_dir))
os.system('python render_animations.py {} {}'.format(image_dir, name))
