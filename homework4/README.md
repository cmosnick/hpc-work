To run:

* Add a data directory containing some data files

    > mkdir ../data
    > cp <path-to-datafile> ../data/<datafile>

* Use sample search vector file provided, or create own by copying 1 line from any data file into a new csv

* make:
    
    > make

    > mpirun -n 4 mpi_nearest_neighbor sample_search_vector.csv ../data/ 5 

    > mpirun -n <num threads> mpi_nearest_neighbor <search-vector-file> <data directory> <num results> 

To run tests:

*  make:
    > make run_tests NUMRUNS=<number of runs>

