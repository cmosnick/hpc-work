set terminal png nocrop enhanced
set output 'hw3.threads.png'
set ylabel 'Seconds'
set xlabel 'Number of Threads'
plot 	"p1.interleave.threads.data" using 2:1 title 'row interleave' with linespoints lw 2, \
     	"p1.block.threads.data" using 2:1 title 'block partition' with linespoints lw 2, \
		"p1.m2.data" using 2:1 title 'm2' with linespoints lw 2, \
		"p1.m3.data" using 2:1 title 'm3' with linespoints lw 2, \
		"p1.m4.data" using 2:1 title 'boost' with linespoints lw 2

