IMAGES = hoffimg/0_0.png hoffimg/0_1.png

all: hoffstuff.html $(IMAGES)

hoffstuff.html: header.html hoffstuff.md
	cp header.html hoffstuff.html && cmark --unsafe --width 80 hoffstuff.md >> hoffstuff.html

hoffimg/0_0.png hoffimg/0_1.png: hoffcode/part0.py
	python3 hoffcode/part0.py

clean:
	rm hoffstuff.html hoffimg/$(IMAGES).png
