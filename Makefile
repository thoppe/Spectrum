all:
	echo "pass"

render:
	python render_animations.py videos/drag_queen/out drag_queen
	python render_animations.py videos/guy2girl/out/ guy2girl
	python render_animations.py videos/guy2girl2/out/ guy2girl2
	python render_animations.py videos/guy2girl4/out/ guy2girl4
	python render_animations.py videos/guy2girl5/out/ guy2girl5
	python render_animations.py videos/jenna_avocado/out jenna
	python render_animations.py videos/julien/out/ julien
	python render_animations.py videos/girl2guy2/out/ girl2guy2
	python render_animations.py videos/girl2guy3/out/ girl2guy3
	python render_animations.py videos/girl2guy/out/ girl2guy
