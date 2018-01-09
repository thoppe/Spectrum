import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
import os, glob

save_dest = "figures"
os.system('mkdir -p '+save_dest)

def plot(name):
    print "Starting ", name
    
    f_csv = os.path.join("results/",
                         "image_processed_videos_{}.csv".format(name))

    if not os.path.exists(f_csv):
        print "Missing", f_csv
        return False
        

    df = pd.read_csv(f_csv).sort_values("filename")
    df["timestamp"] = df.filename.apply(lambda x:x.split('/')[-1].split('.')[0])
    df["timestamp"] = df.timestamp.astype(int)
    df = df.set_index("timestamp")

    for key in ["gender", "age"]:
        f_png = os.path.join(save_dest, name + "_" + key + ".png")
        #if os.path.exists(f_png):  continue
        
        plt.figure(figsize=(8,4))
        plt.scatter(df.index, df[key], s=5, lw=2, alpha=0.25)
        #plt.plot([],[])
        #plt.plot([],[])
        plt.plot(pd.ewma(df[key], span=30), alpha=0.8, lw=2)

        if key=="gender":
            plt.ylim(0,1)
            X = np.linspace(df.index.min(), df.index.max())
            plt.plot(X,np.ones(X.shape)*0.5, 'r--', alpha=0.5)
            plt.ylabel(r"male $\leftarrow$  Gender prediction  $\rightarrow$ female")
        if key=="age":
            plt.xlabel("Age prediction")
            plt.ylim(10, 55)

        sns.despine()
        plt.xlabel("Video timestamp (seconds)")
        plt.tight_layout()

        plt.savefig(f_png)

names = glob.glob("videos/*")
for name in names:
    name = name.split('/')[-1]
    plot(name)

#plt.show()
