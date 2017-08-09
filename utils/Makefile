%.o: %.cpp
	$(CXX) -c -O3 -std=c++11 -o $@ $< $(CFLAGS)

quadrant_masks: quadrant_masks.o
	$(CXX) -O3 -std=c++11 -o $@ $^ $(CFLAGS)

all: quadrant_masks

clean:
	rm -f quadrant_masks quadrant_masks.o
