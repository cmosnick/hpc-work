To run:

* Add a data directory containing some data files

    > mkdir ../data
    > cp <path-to-datafile> ../data/<datafile>

* Use sample search vector file provided, or create own by copying 1 line from any data file into a new csv

* make:
    
    > make

    > mpirun -n 4 mpi_nearest_neighbor filename ../data/ 5 sample_search_vector.csv

    >  
