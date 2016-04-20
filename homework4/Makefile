COMPILE_FLAGS = -Wall -O2

all: mpi_nearest_neighbor

mpi_nearest_neighbor: directory_scanner.o Timing.o mpi_nearest_neighbor.o
	mpic++ $(COMPILE_FLAGS) -g -o mpi_nearest_neighbor directory_scanner.o Timing.o mpi_nearest_neighbor.o  -lboost_system -lboost_filesystem -lboost_thread-mt

mpi_nearest_neighbor.o: mpi_nearest_neighbor.cpp directory_scanner.cpp
	mpic++ $(COMPILE_FLAGS) -c mpi_nearest_neighbor.cpp

directory_scanner.o: directory_scanner.cpp
	mpic++ $(COMPILE_FLAGS) -c directory_scanner.cpp

timing.o: Timing.cpp
	mpic++ $(COMPILE_FLAGS) -c Timing.cpp

clean:
	rm -f mpi_nearest_neighbor *.o








