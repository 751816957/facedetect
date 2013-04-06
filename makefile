all:facedetect
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

facedetect : facedetect.cpp
	g++ facedetect.cpp -fpermissive $(CFLAGS) $(LIBS) -o facedetect

clean:
	rm facedetect
