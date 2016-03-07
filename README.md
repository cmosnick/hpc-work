To run:

		mkdir build
		cd build
		cmake ..
		make
		./homework3 agricultural/agricultural00_rot_000.tif ../4200_HPC.csv 100 1

* Note: segfault is occurring for higher number of threads, >5 on my machine.  I can't figure out why, I must have done something because I had it working before.