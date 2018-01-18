## Install notes

Install missing deps from fresh install

	sudo apt-get install build-essential cmake
	sudo apt-get install libgtk-3-dev
	sudo apt-get install libboost-all-dev
	sudo apt-get install libopenblas-dev liblapack-dev
	sudo apt install webp melt
	sudo pip install -r requirements.txt

#### Fresh install of Ubuntu 16.04, CUDA and TF

Fresh 16.04, install updates. Install chome, emacs for ease of use.

      google-chrome-stable_current_amd64.deb
      sudo apt install git emacs24

Install DEB

	cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb

Install CUDA, then restart.

	sudo apt update
	sudo apt install cuda libcupti-dev

Install CUDNN

	./libcudnn6_6.0.21-1+cuda8.0_amd64.deb

Add the following lines to .bashrc

        export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
    	export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

Install python deps

	sudo apt install python-pip python-dev python-tk
	sudo pip install --upgrade pip
	sudo pip install --upgrade numpy scipy matplotlib scikit-learn h5py seaborn pandas
	sudo pip install --upgrade tensorflow-gpu


#### Other notes

Convert images back to video

    avconv -y -r 15 -i {}/%05d.png -b:v 768k -s 640x360

Sample musical starting tracks for youtube audio
+ Music track (Lisa spector)
+ Gymnopedie no1
+ Prelude Op. 28 No. 15 D flat Major (Raindrop)
+ Apres un Reve