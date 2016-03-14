set terminal png nocrop enhanced
# log xy
set output 'hw3.threads.logxy.png'
set log xy
set ylabel 'Seconds'
set xlabel 'Number of Threads'
plot 	"p1.interleave.threads.data" using 2:1 title 'Row Interleave' with linespoints lw 2, \
		"p1.block.threads.data" using 2:1 title 'Block Partition' with linespoints lw 2, \
		"p1.m2.data" using 2:1 title 'm2' with linespoints lw 2, \
		"p1.m3.data" using 2:1 title 'm3' with linespoints lw 2, \
		"p1.m4.data" using 2:1 title 'boost' with linespoints lw 2

