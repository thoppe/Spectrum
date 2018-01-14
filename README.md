# Spectrum
_exploring the gender continuum with deep learning_

Videos are posted to the [eXs](https://www.youtube.com/channel/UCAJIi3CN0WhFw-egGapQ7ug) youtube channel.

![](src/sample_image.jpg)

### Notes for a new video

+ Download the video from the url `youtube-dl URL`
+ Convert the images, one per second `mkdir out; avconv -r 4 -an -y "out/%04d.png" -i VIDEO.webm`
+ Manually remove frames at the start and end that are not content
+ Identify the faces from the video `python process_images.py videos/drag_queen/out/`. This takes a long time.
+ Use the CNN model to label each image `python predict_images.py image_processed/videos/drag_queen/out/`. Data is saved in the [results](results/) folder.
+ Generate the animation `python render_animations.py drag_queen`
+ Update the video sources in this [README](https://github.com/thoppe/Spectrum/edit/master/README.md).
+ Upload to youtube and fill out the [sources template](template_youtube.md).

### Attribution

Tools and assests for the Spectrum project. Model was trained off images from [IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). Code was adapted from [BoyuanJiang](https://github.com/BoyuanJiang/Age-Gender-Estimate-TF) from the the paper, [Deep Expectation of Real and Apparent Age from a Single Image Without Facial Landmarks](https://link.springer.com/article/10.1007/s11263-016-0940-3).

### Video sources

#### Drag Queens
+ Martin Catalogne, [Drag Queen Make Up Tutorial](https://www.youtube.com/watch?v=khGXJxF_LjI), (label _drag\_queen_)
+ dope2111, [Guy to Girl Makeup Transformation](https://www.youtube.com/watch?v=_4FoWD6zpKU), (label _guy2girl_)
+ Miss Fame, [SuperNatural Blonde Makeup Tutorial](https://www.youtube.com/watch?v=lu1zSZui8Gc), (label _guy2girl2_)
+ Martin Catalogne, [Increíble Transformación - Maquillaje Drag Queen en Tonos Cálidos](https://www.youtube.com/watch?v=aJAMcE9cP0E), (label _guy2girl4_)
+ Darrian Glover, [Black Drag Queen Makeup Tutorial || Orlando Pride](https://www.youtube.com/watch?v=9PC428YyCas), (label _guy2girl5_)

#### Drag Kings

+ Beauty by Jannelle, [Gender Makeup Transformation](https://www.youtube.com/watch?v=Bw8M-wfHC9A), (label _girl2guy_)
+ Claire Dim, [Woman To Man Makeup Transformation](https://www.youtube.com/watch?v=7MwfKiRlRA4&list=RDBw8M-wfHC9A), (label _girl2guy2_)
+ Sailor Cruz, [Female to male makeup tutorial](https://www.youtube.com/watch?v=GQ1tDCOr_ko), (label _girl2guy3_)


#### Control
+ Jenna, [How To Avoid Talking To People You Don't Want To Talk To](https://www.youtube.com/watch?v=8wRXa971Xw0), (label _jenna_)
+ Julien Solomita, [reading comments about my mustache](https://www.youtube.com/watch?v=0kYykClLSqo), (label _julien_)

### [Install notes](NOTES.md)
