all: child_shm_fib

child_shm_fib: shm_example.cpp
	g++ shm_example.cpp -o child_shm_fib

clean:
	rm child_shm_fib
