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


run_tests: all run_even_split


run_even_split:
	mpirun --hostfile hostfiles/8_even mpi_nearest_neighbor sample_search_vector.csv /cluster/ 5	&& \
	mpirun --hostfile hostfiles/16_even mpi_nearest_neighbor sample_search_vector.csv /cluster/ 5  && \
	mpirun --hostfile hostfiles/24_even mpi_nearest_neighbor sample_search_vector.csv /cluster/ 5 && \
	mpirun --hostfile hostfiles/32_even mpi_nearest_neighbor sample_search_vector.csv /cluster/ 5 && \
	mpirun --hostfile hostfiles/40_even mpi_nearest_neighbor sample_search_vector.csv /cluster/ 5 && \
	mpirun --hostfile hostfiles/48_even mpi_nearest_neighbor sample_search_vector.csv /cluster/ 5



