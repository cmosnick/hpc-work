all: child_semaphore child_semaphore_spin

child_semaphore: semaphore_example.cpp
	g++ -o child_semaphore semaphore_example.cpp

child_semaphore_spin: semaphore_example_spin.cpp
	g++ -o child_semaphore_spin semaphore_example_spin.cpp

clean:
	rm child_semaphore child_semaphore_spin
