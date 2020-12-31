IMAGES = hoffimg/0_0.png hoffimg/0_1.png hoffimg/1_0.png hoffimg/1_1.png hoffimg/1_2.png hoffimg/1_3.png hoffimg/2_0.png hoffimg/2_1.png hoffimg/3_0.png hoffimg/3_1.png hoffimg/4_0.png hoffimg/4_1.png hoffimg/4_2.png hoffimg/5_0.png

all: hoffstuff.html $(IMAGES)

hoffstuff.html: hoffcode/header.html hoffcode/hoffstuff.md
	cp hoffcode/header.html hoffstuff.html && cmark --unsafe --width 80 hoffcode/hoffstuff.md >> hoffstuff.html

hoffimg/0_0.png hoffimg/0_1.png: hoffcode/part0.py
	python3 hoffcode/part0.py

hoffimg/1_0.png hoffimg/1_1.png hoffimg/1_2.png hoffimg/1_3.png: hoffcode/part1.py
	python3 hoffcode/part1.py

hoffimg/2_0.png hoffimg/2_1.png: hoffcode/part2.py
	python3 hoffcode/part2.py

hoffimg/3_0.png: hoffcode/part3_0.py
	python3 hoffcode/part3_0.py

hoffimg/3_1.png: hoffcode/part3_1.py
	python3 hoffcode/part3_1.py

hoffimg/4_0.png hoffimg/4_1.png hoffimg/4_2.png: hoffcode/part4.py
	python3 hoffcode/part4.py

hoffimg/5_0.png: hoffcode/part5.py
	python3 hoffcode/part5.py

clean:
	rm hoffstuff.html $(IMAGES)
