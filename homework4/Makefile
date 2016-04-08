all: mpi_nearest_neighbor

mpi_nearest_neighbor: directory_scanner.o mpi_nearest_neighbor.o
	mpic++ -g -o mpi_nearest_neighbor directory_scanner.o mpi_nearest_neighbor.o  -lboost_system -lboost_filesystem 

mpi_nearest_neighbor.o: mpi_nearest_neighbor.cpp directory_scanner.cpp
	mpic++ -c mpi_nearest_neighbor.cpp

directory_scanner.o: directory_scanner.cpp
	mpic++ -c directory_scanner.cpp

clean:
	rm -f mpi_nearest_neighbor *.o