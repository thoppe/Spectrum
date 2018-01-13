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
frame_per_second = 15
lead_time = 1
post_time = 3

scale = 2
x_width /= scale
y_height /= scale


def read_image(f_png):
    # (1920, 1080, 3) (16:9) is the ideal aspect ratio
    f_png = f_png.replace("image_processed/", "")
    img = plt.imread(f_png)
    return img


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
    df["timestamp"] = df.timestamp.astype(int)
    df = df.set_index("timestamp")

    df["gender"] = (df["gender"] * 100).astype(int)
    df["EMA"] = df.gender.ewm(span=30).mean()

    df = df[:debug_cutoff]

    # Preload the images for a small speed boost
    # with joblib.Parallel(-1) as MP:
    #    images = MP(joblib.delayed(read_image)(f) for f in df.filename)

    fig, axes = plt.subplots(1, 2, figsize=(x_width / 100., y_height / 100.))
    fig.set_size_inches(x_width / 100., y_height / 100.)

    ax = axes[1]
    ax.scatter(df.index, df["gender"], s=5, lw=2, alpha=0.25)
    ax.plot(df.index, df.EMA, alpha=0.8, lw=2)
    ax.set_ylim(0, 100)

    X = np.linspace(df.index.min(), df.index.max())
    ax.plot(X, np.ones(X.shape) * 50, '--', color=red_line_color, alpha=0.5)

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
            continue

        Pline.set_data([df.index[:k + 1], df.EMA[:k + 1]])
        img = read_image(row.filename)

        if pimg is None:
            pimg = axes[0].imshow(img)
            bbox = pimg.get_clip_box().get_points()
            text_x = bbox[0, 0] / (x_width)
            text_y = bbox[1, 0] / y_height + 0.025

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
            label = "Male"
        elif val > 40:
            label = "Androgynous"
        elif val >= 0:
            label = "Female"

        textbox2.set_text(label)
        textbox2.set_color(pal[min(int(val), 99)])

        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.show()
        # exit()

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

    # org_dir = os.getcwd()
    # os.chdir(output_dir)

    f_avi = os.path.join('animations', label + '.avi')

    args = " -r {} -b:v 44000k -s {}x{} -y "
    args = args.format(frame_per_second, x_width * scale, y_height * scale)

    if not os.path.exists(f_avi) or True:
        f_main = os.path.join('animations/', label + '_main.avi')
        f_header = os.path.join('animations/', label + '_header.avi')
        f_footer = os.path.join('animations/', label + '_footer.avi')

        f_png = os.path.join(output_dir, '00000.png')
        cmd = "avconv -loop 1 -i {} -t 00:00:{} -shortest "
        cmd = cmd.format(f_png, lead_time,) + args + f_header
        os.system(cmd)

        img_f_final = max(glob.glob(os.path.join(output_dir, '*')))
        cmd = "avconv -loop 1 -i {} -t 00:00:{} -shortest "
        cmd = cmd.format(img_f_final, post_time,) + args + f_footer
        os.system(cmd)

        cmd = "avconv -ac 1 -i {} "
        cmd = cmd.format(os.path.join(output_dir, "%05d.png")) + args
        cmd += f_main
        os.system(cmd)

        # Merge the files
        names = ("animations/{f}_header.avi|"
                 "animations/{f}_main.avi|"
                 "animations/{f}_footer.avi"
                 )
        cmd = 'ffmpeg -y -i "concat:' + names + '" -c copy {final}'
        cmd = cmd.format(f=label, final=f_avi)

        os.system(cmd.format(f=label))
        os.remove(f_main)
        os.remove(f_header)
        os.remove(f_footer)

if __name__ == "__main__":
    animate(image_dir)
    build_mp4(image_dir, label)
