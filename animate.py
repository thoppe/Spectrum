import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
import os, glob
from tqdm import tqdm
import matplotlib.font_manager as fm
import joblib

f_background = "src/plot_background.png"

pal = sns.dark_palette((166/256., 98/256., 129/256.),
                       reverse=True,
                       n_colors=100)

f_font = "src/leaguespartan-bold.ttf"
prop = fm.FontProperties(fname=f_font)
# avconv -y -r 15 -i %05d.jpg -b:v 768k -s 640x360 test5.mp4

# 2:1 aspect ratio
#x_width = 1024
#y_height = 512

# 16:9 aspect ratio
x_width = 1920
y_height = 1080

scale = 2
x_width /= scale
y_height /= scale

save_dest = "animations"
os.system('mkdir -p '+save_dest)

def animate(name):
    print "Starting ", name
    
    f_csv = os.path.join("results/",
                         "image_processed_videos_{}.csv".format(name))

    if not os.path.exists(f_csv):
        print "Missing", f_csv
        return False

    os.system('mkdir -p '+os.path.join(save_dest,name))
        

    df = pd.read_csv(f_csv).sort_values("filename")
    df["timestamp"] = df.filename.apply(lambda x:x.split('/')[-1].split('.')[0])
    df["timestamp"] = df.timestamp.astype(int)
    df = df.set_index("timestamp")

    key = "gender"

    df["gender"] = (df.gender*100).astype(int)

    counter = 0
    df["EMA"] = pd.ewma(df[key], span=30)

    red_line_color = '#FF002A'

            
    for k, (_, row) in tqdm(enumerate(df.iterrows())):

        f_save = os.path.join(save_dest, name, "{:05d}.png".format(k))
        if os.path.exists(f_save): continue
        print f_save

        fig, axes = plt.subplots(1,2,figsize=(x_width/100.,y_height/100.))
        ax = axes[1]

        ax.scatter(df.index, df[key], s=5, lw=2, alpha=0.25)

        ax.plot(df.index, df.EMA, alpha=0.8, lw=2)
        ax.plot(df.index[:k+1], df.EMA[:k+1], lw=4, zorder=2,
                color=red_line_color)
        
        ax.set_ylim(0,100)

        X = np.linspace(df.index.min(), df.index.max())
        ax.plot(X,np.ones(X.shape)*50, '--', color=red_line_color, alpha=0.5)

        ax.set_ylabel(
            r"female $\leftarrow$ Gender spectrum $\rightarrow$ male",
            fontproperties=prop,
            color=pal[-1],
        )
        ax.set_xlabel(
            "Video timestamp (seconds)",
            color=pal[-1],
            fontproperties=prop
        )
        
        sns.despine()       

        f_png = row.filename
        f_png = f_png.replace("image_processed/", "")
        img = plt.imread(f_png)
        
        pimg = axes[0].imshow(img)
        plt.tight_layout()

        val = df.EMA[:k+1].iloc[-1]

        text = "Spectrum: {:d}".format(int(val))
        
        text_color = pal[min(int(val),99)]
        text_x, text_y = 0.04, 0.18
        
        bbox = pimg.get_clip_box().get_points()
        text_x = bbox[0,0]/(x_width)
        text_y = 1-(bbox[1,0]/(y_height))
        
        plt.gcf().text(text_x, text_y, text, fontsize=30,
                       color=pal[-1],
                       zorder=1, fontproperties=prop)

        axes[0].axis('off')

        if val >= 60:
            label = r"Male"
        #elif val > 0.6:            label = "Probably male?"
        elif val > 40:
            label = "Androgynous"
        #elif val > 0.2:            label = "Probably female?"
        elif val >= 0:
            label = "Female"
            
        plt.gcf().text(text_x, text_y - 0.05,
                       label,
                       fontsize=18,
                       color=text_color,
                       zorder=1, fontproperties=prop)
        
        fig.set_size_inches(x_width/100.,y_height/100.)
        plt.savefig(f_save, dpi=200,transparent=True)
        plt.close()

        if f_background is not None:
            add_bg_cmd = "convert {back} {src} -composite {src}"
            cmd = add_bg_cmd.format(src=f_save, back=f_background)
            os.system(cmd)
            
        #plt.show()
        #exit()

NAMES = [
    "jenna", "julien", "drag_queen",
    "girl2guy","girl2guy2","girl2guy3",
    "guy2girl","guy2girl2","guy2girl3","guy2girl4","guy2girl5",    
]

func = joblib.delayed(animate)
with joblib.Parallel(-1) as MP:
    MP(func(x) for x in NAMES)


org_dir = os.getcwd()
os.chdir(save_dest)

def build_mp4(name): 
    f_mp4 = name+'.mp4'

    if not os.path.exists(f_mp4):
        #cmd = "avconv -ac 1 -y -r 15 -i {}/%05d.png -b:v 768k -s 640x360 "
        cmd = "avconv -ac 1 -y -r 15 -i {}/%05d.png -b:v 44000k -s {}x{} "

        cmd = cmd.format(name, x_width, y_height)+f_mp4
        print cmd
        os.system(cmd)

FD = filter(os.path.isdir, os.listdir(os.getcwd()))

import joblib
func = joblib.delayed(build_mp4)

with joblib.Parallel(-1) as MP:
    MP(func(x) for x in FD)


  