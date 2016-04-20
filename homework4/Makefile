# $COMPILE_FLAGS = {-Wall}

# all: mpi_nearest_neighbor

# mpi_nearest_neighbor: directory_scanner.o mosnickThread.o hw3.o mpi_nearest_neighbor.o
# 	mpic++ -Wall -g -o mpi_nearest_neighbor directory_scanner.o mosnickThread.o hw3.o mpi_nearest_neighbor.o  -lboost_system -lboost_filesystem -lboost_thread-mt

# mpi_nearest_neighbor.o: mpi_nearest_neighbor.cpp directory_scanner.cpp ./homework3_copy/hw3.cpp
# 	mpic++ -Wall -c mpi_nearest_neighbor.cpp

# directory_scanner.o: directory_scanner.cpp
# 	mpic++ -Wall -c directory_scanner.cpp

# hw3.o: mosnickThread.o ./homework3_copy/hw3.cpp 
# 	mpic++ -Wall -c ./homework3_copy/hw3.cpp

# mosnickThread.o:
# 	mpic++ -Wall -c ./homework3_copy/mosnickThread.cpp

# clean:
# 	rm -f mpi_nearest_neighbor *.o

COMPILE_FLAGS = {-Wall}

all: mpi_nearest_neighbor

mpi_nearest_neighbor: directory_scanner.o mpi_nearest_neighbor.o
	mpic++ -Wall -g -o mpi_nearest_neighbor directory_scanner.o mpi_nearest_neighbor.o  -lboost_system -lboost_filesystem -lboost_thread-mt

mpi_nearest_neighbor.o: mpi_nearest_neighbor.cpp directory_scanner.cpp
	mpic++ -Wall -c mpi_nearest_neighbor.cpp

directory_scanner.o: directory_scanner.cpp
	mpic++ -Wall -c directory_scanner.cpp

# hw3.o: mosnickThread.o ./homework3_copy/hw3.cpp 
# 	mpic++ -Wall -c ./homework3_copy/hw3.cpp

# mosnickThread.o:
# 	mpic++ -Wall -c ./homework3_copy/mosnickThread.cpp

clean:
	rm -f mpi_nearest_neighbor *.o








