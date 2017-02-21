
SRC = affinity_check.cpp \
      mpi_local_ranks.cpp \
      driver.cpp

OBJ:= $(SRC:.cpp=.o)

CXXFLAGS=-fopenmp -std=c++11 -O3 -I. -Wall -pedantic

%.o: %.cpp $(SRC)
	mpicxx $(CXXFLAGS) -c -o $@ $<

driver: $(OBJ)
	mpicxx $(CXXFLAGS) -o $@ $^


clean:
	rm -f $(OBJ) driver
