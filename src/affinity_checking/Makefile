
SRC = affinity_check.cpp \
      mpi_local_ranks.cpp \
      driver.cpp

OBJ:= $(SRC:.cpp=.o)

%.o: %.cpp $(SRC)
	mpicxx -fopenmp -std=c++11 -O3 -I. -c -o $@ $<

driver: $(OBJ)
	mpicxx -fopenmp -std=c++11 -o $@ $^


clean:
	rm -f $(OBJ) driver
