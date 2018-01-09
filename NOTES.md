
## Install notes

Getting the repo working:

	https://github.com/BoyuanJiang/Age-Gender-Estimate-TF

Install missing deps from fresh install

	sudo pip install imutils opencv-python
	sudo apt-get install build-essential cmake
	sudo apt-get install libgtk-3-dev
	sudo apt-get install libboost-all-dev
	sudo apt-get install libopenblas-dev liblapack-dev
	sudo pip install dlib

       sudo -H pip install --upgrade youtube-dl
       sudo apt install webp


Convert images back to video
avconv -y -r 15 -i {}/%05d.png -b:v 768k -s 640x360

Music track (Lisa spector)
Gymnopedie no1
Prelude Op. 28 No. 15 D flat Major (Raindrop)
Apres un Reve