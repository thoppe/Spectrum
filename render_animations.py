import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
import os
import glob
import sys
from tqdm import tqdm
import matplotlib.font_manager as fm
import joblib

debug_cutoff = 10**20
f_background = "src/plot_background.png"

pal = sns.dark_palette((166 / 256., 98 / 256., 129 / 256.),
                       reverse=True,
                       n_colors=100)
red_line_color = '#FF002A'

f_font = "src/leaguespartan-bold.ttf"
prop = fm.FontProperties(fname=f_font)

image_dir = sys.argv[1]
label = sys.argv[2]

# 2:1 aspect ratio
# x_width = 1024
# y_height = 512

# 16:9 aspect ratio
x_width = 1920
y_height = 1080

# Hard-code positions of the text
text_x = 0.0486689814815
text_y = 0.857008744856

force_show = False
frame_encoding_speed = 4
frame_per_second = 15*frame_encoding_speed

intro_time = 5
lead_time = 1
post_time = 3

ewm_span = 30 * frame_encoding_speed

scale = 2
x_width /= scale
y_height /= scale

f_intro_png = "src/introduction_slide.png"
f_intro_png2 = "src/introduction_slide2.png"


def read_image(f_png):
    # (1920, 1080, 3) (16:9) is the ideal aspect ratio
    f_png = f_png.replace("image_processed/", "")
    img = plt.imread(f_png)
    return img

def build_intro_slide():
    if os.path.exists(f_intro_png):
        return True

    print "Drawing introduction slide"
    
    fig, axes = plt.subplots(
        1, 1, figsize=(x_width / 100., y_height / 100.),
        #facecolor='#EEEEEE',
    )    
    axes.axis('off')
    g = plt.gcf()
    
    args = {"color":pal[-1], "fontproperties":prop}
    dx = -.10

    g.text(text_x, text_y, "Spectrum:", fontsize=30, **args)

    args["fontproperties"] = fm.FontProperties(
        fname="src/LibreBaskerville-Regular.ttf")
    args["color"] = pal[-3]


    lines = [
        "a measurement of gender expression derived from ",
        "a computational model.",
        "",
        "Trained on celebrity faces, Spectrum made associations",
        "using visual cues (e.g., makeup or facial hair). The ",
        "measurements reflect Western cultural sterotypes.",
    ]
    
    for k,line in enumerate(lines):
        g.text(text_x-dx/2, text_y+(k+0.5)*dx+dx/2, line, fontsize=18, **args)

    plt.savefig(f_intro_png, dpi=200)

    ### Render intro screen #2
        
    fig, axes = plt.subplots(
        1, 1, figsize=(x_width / 100., y_height / 100.),
    )
    axes.axis('off')
    g = plt.gcf()

    lines = [
        "Gender is more complex",
        "than your physical appearance",
        "or the output of a computer model.",
    ]
    for k,line in enumerate(lines):
        g.text(text_x-dx/2, text_y+(k+2.0)*dx+dx/2, line, fontsize=22, **args)
    plt.savefig(f_intro_png2, dpi=200)


def animate(name):
    print("Starting", name)

    f_csv = os.path.join("results", image_dir, "gender_prediction.csv")

    if not os.path.exists(f_csv):
        raise ValueError("Missing " + f_csv)

    output_dir = os.path.join("animations/", image_dir)
    os.system('mkdir -p ' + output_dir)

    df = pd.read_csv(f_csv).sort_values("filename")
    df["timestamp"] = df.filename.apply(
        lambda x: x.split('/')[-1].split('.')[0])
    df["timestamp"] = df.timestamp.astype(float)
    df["seconds"] = df.timestamp / float(frame_encoding_speed)

    # Convert the measurement into a discretized value
    df["gender"] = (df["gender"] * 100).astype(int)

    # Swap direction
    df["gender"] = 100 - df.gender

    # Compute smoothed curve
    df["EMA"] = df.gender.ewm(span=ewm_span).mean()
    df = df[:debug_cutoff]

    fig, axes = plt.subplots(1, 2, figsize=(x_width / 100., y_height / 100.))
    fig.set_size_inches(x_width / 100., y_height / 100.)

    ax = axes[1]
    ax.scatter(df.seconds, df["gender"], s=2, lw=1, alpha=0.15)
    ax.plot(df.seconds, df.EMA, alpha=0.8, lw=2)
    ax.set_ylim(0, 100)

    X = np.linspace(df.seconds.min(), df.seconds.max())
    ax.plot(X, np.ones(X.shape) * 50, '--', color=red_line_color, alpha=0.5)

    ax.set_ylabel(
        r"male $\leftarrow$ Gender expression $\rightarrow$ female",
        fontproperties=prop,
        color=pal[-1],
    )
    ax.set_xlabel(
        "Video timestamp (seconds)",
        color=pal[-1],
        fontproperties=prop
    )

    Pline = ax.plot([], [], lw=4, zorder=2, color=red_line_color)[0]
    axes[0].axis('off')

    axes_color = "#333333"
    axes[1].tick_params(axis='x', colors=axes_color, which='both')
    axes[1].spines['bottom'].set_color(axes_color)
    axes[1].tick_params(axis='y', colors=axes_color, which='both')
    axes[1].spines['left'].set_color(axes_color)

    sns.despine()
    plt.tight_layout()

    pimg = None


    for k, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df)):

        f_save_no_bg = os.path.join(output_dir,
                                    "NO_BG_{:05d}.png".format(k))

        f_save = os.path.join(output_dir, "{:05d}.png".format(k))

        if os.path.exists(f_save) or os.path.exists(f_save_no_bg):
            if not force_show:
                continue

        Pline.set_data([df.seconds[:k + 1], df.EMA[:k + 1]])
        img = read_image(row.filename)

        if pimg is None:
            pimg = axes[0].imshow(img)
            bbox = pimg.get_clip_box().get_points()

            # Something like this
            # new clear axis overlay with 0-1 limits
            # ax2 = pyplot.axes([0,0,1,1], axisbg=(1,1,1,0))

            textbox1 = plt.gcf().text(text_x, text_y, "",
                                      fontsize=30,
                                      color=pal[-1],
                                      zorder=1,
                                      fontproperties=prop)

            textbox2 = plt.gcf().text(text_x, text_y - 0.05, "",
                                      fontsize=18,
                                      zorder=1, fontproperties=prop)
        else:
            pimg.set_data(img)

        val = df.EMA[:k + 1].iloc[-1]

        textbox1.set_text("Spectrum: {:d}".format(int(val)))

        if val >= 60:
            label = "Female"
        elif val > 40:
            label = "Androgynous"
        elif val >= 0:
            label = "Male"

        textbox2.set_text(label)
        textbox2.set_color(pal[min(int(100-val), 99)])

        fig.canvas.draw()
        fig.canvas.flush_events()

        if force_show:
            plt.show()
            exit()

        plt.savefig(f_save_no_bg, dpi=200, transparent=True)

    # Add background images in parallel
    images_need_background = glob.glob(os.path.join(
        output_dir, "NO_BG_*.png".format(k)))

    with joblib.Parallel(-1) as MP:
        MP(joblib.delayed(add_background)(f, f_background)
           for f in images_need_background)


def add_background(f_save, f_background):
    if f_background is not None:
        add_bg_cmd = "convert {back} {src} -composite {final}"
        cmd = add_bg_cmd.format(src=f_save, back=f_background,
                                final=f_save.replace("NO_BG_", ""))
        os.system(cmd)
        os.remove(f_save)


def build_mp4(image_dir, label):

    output_dir = os.path.join("animations/", image_dir)
    f_final = os.path.join('animations', label + '.mp4')

    if not os.path.exists(f_final):
        cross_time = 1
        fps = frame_per_second

        f_first_frame = os.path.join(output_dir, '00000.png')
        f_final_frame = max(glob.glob(os.path.join(output_dir, '*')))
        
        #f_img_collection = os.path.join(output_dir, '000??.png')
        f_img_collection = os.path.join(output_dir, '*.png')

        
        cmd = ' '.join([
            "melt -profile atsc_1080p_60",
            "{} out={}",
            "{} out={} -mix {} -mixer luma",
            "{} out={} -mix {} -mixer luma",
            "-group in=0 out=1 {}",
            "{} out={}",            
            "-consumer avformat:{}",
        ])

        cmd = cmd.format(
            f_intro_png, intro_time*fps,
            f_intro_png2, (lead_time+intro_time)*fps, cross_time*fps,
            f_first_frame, (lead_time+cross_time)*fps, cross_time*fps,
            f_img_collection,
            f_final_frame, post_time*fps,
            f_final,
        )
        print cmd
        os.system(cmd)


if __name__ == "__main__":
    build_intro_slide()    
    animate(image_dir)
    build_mp4(image_dir, label)
