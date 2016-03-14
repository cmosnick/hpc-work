# set this to what I want
OTHER_FLAGS=-std=c++0x -O2 -Werror -Wall -v
# OTHER_FLAGS=-std=c++0x -O2

# Standard all target
all: hw3_analyze

# HW 1 Analysis Code
hw3_analyze: hw3_analyze.o hw3.o mosnickThread.o  
	g++ -L/usr/local/lib -lboost_system -lboost_thread-mt  mosnickThread.o hw3.o hw3_analyze.o -o hw3_analyze

hw3_analyze.o:	hw3_analyze.cpp 
	g++ -I/usr/local/include/boost/thread ${OTHER_FLAGS} -c hw3_analyze.cpp -I/usr/local/include/

hw3.o: ../src/hw3.cpp
	g++ ${OTHER_FLAGS} -c ../src/hw3.cpp -I/usr/local/include/

mosnickThread.o: ../src/mosnickThread.cpp
	g++ -I/usr/local/include/ ${OTHER_FLAGS} -c ../src/mosnickThread.cpp  

clean:
	rm -f *.o hw3_analyze p1.time p1.out


### Analysis Commands and Dependencies

analysis: data
	@echo "Generating Plots:" ; \
	(gnuplot < plot_hw3.threads.gnu && \
	gnuplot < plot_hw3.threads.logxy.gnu && \
	gnuplot < plot_hw3.threads.logx.gnu && \
	gnuplot < plot_hw3.threads.logy.gnu && \
	gnuplot < plot_hw3.files.gnu && \
	gnuplot < plot_hw3.files.logxy.gnu && \
	gnuplot < plot_hw3.files.logx.gnu && \
	gnuplot < plot_hw3.files.logy.gnu ) 2> /dev/null ; \
	echo "Program Execution Times" ; \
	cat p1.f.data > processesVsTime.csv; \
	cat p1.time;

data: p1.out
	@echo "Scrapping Timing Statistics:"; ./analyze_3.sh

test_1: hw3_analyze
	@echo "Running hw2_analyze:"; (time ./hw3_analyze > p1.out) 2> p1.time

# this will always run the test_1 because the p1.out 
# always has a time stamp newer that the dependency of test_1
p1.out: test_1

