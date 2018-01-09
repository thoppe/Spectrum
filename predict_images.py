import cv2
import glob, os, sys
import joblib
import pandas as pd
from tqdm import tqdm


batch_size = 256
#image_dir = "processed_image/videos/guy2girl/out/"
#image_dir = "image_processed/videos/faking/out/"
image_dir = "image_processed/videos/drag_queen/out/"
image_dir = sys.argv[1]


f_model_path = "./models"
save_dest = "results"
os.system('mkdir -p '+save_dest)

f_csv_output = os.path.join(
    save_dest,
    image_dir.replace('/','_').replace('_out','.csv')
    .strip('/').strip('_')
)


F_IMAGES = glob.glob(os.path.join(image_dir, "*"))

def load_image(f_img):
    image = cv2.imread(f_img, cv2.IMREAD_COLOR)
    return f_img, image

func = joblib.delayed(load_image)
with joblib.Parallel(-1) as MP:
    ITR = tqdm(F_IMAGES)
    IMAGES = [(f,img) for f,img in MP(func(x) for x in ITR)]
    IMAGES = dict(IMAGES)

def image_iterator():
    block = []
    for f,img in IMAGES.items():
        if img is not None:
            block.append((f,img))
        if len(block) == batch_size:
            yield block
            block = []
    if block:
        yield block
   
print "Loaded", len(IMAGES)


import tensorflow as tf
import src.inception_resnet_v1

data = []

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Graph().as_default():
    sess = tf.Session(config=config)

    images_in = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
    images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_in) #BGR TO RGB
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = src.inception_resnet_v1.inference(
        images_norm,
        keep_probability=0.8,
        phase_train=train_mode,
        weight_decay=1e-5)
    #gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    gender = tf.nn.softmax(gender_logits)
    
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(f_model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    for block in image_iterator():
        f_names, imgs = zip(*block)
        print len(f_names), len(imgs)
        
        args = {images_in:imgs, train_mode: False}
        age_res, gender_res = sess.run([age, gender], feed_dict=args)

        for f, ax, gx in zip(f_names, age_res, gender_res):
            data.append({"age":ax, "gender":gx[1],"filename":f})

df = pd.DataFrame(data).sort_values('filename').set_index("filename")
df.to_csv(f_csv_output)

print df

