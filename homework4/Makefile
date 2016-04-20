COMPILE_FLAGS = {-Wall}

all: mpi_nearest_neighbor

mpi_nearest_neighbor: directory_scanner.o timing.o mpi_nearest_neighbor.o
	mpic++ -Wall -g -o mpi_nearest_neighbor directory_scanner.o timing.o mpi_nearest_neighbor.o  -lboost_system -lboost_filesystem -lboost_thread-mt

mpi_nearest_neighbor.o: mpi_nearest_neighbor.cpp directory_scanner.cpp
	mpic++ -Wall -c mpi_nearest_neighbor.cpp

directory_scanner.o: directory_scanner.cpp
	mpic++ -Wall -c directory_scanner.cpp

timing.o: Timing.cpp
	mpic++ -Wall -c Timing.cpp

clean:
	rm -f mpi_nearest_neighbor *.o








