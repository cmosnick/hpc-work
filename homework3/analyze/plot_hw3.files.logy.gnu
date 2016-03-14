set terminal png nocrop enhanced
# log y
set output 'hw3.files.logy.png'
set log y
set ylabel 'Seconds'
set xlabel 'File Size'
plot 	"p1.interleave.files.data" using 2:1 title 'Row Interleave' with linespoints lw 2, \
        "p1.block.files.data" using 2:1 title 'Block Partition' with linespoints lw 2, \
		"p1.m2.data" using 2:1 title 'm2' with linespoints lw 2, \
		"p1.m3.data" using 2:1 title 'm3' with linespoints lw 2, \
		"p1.m4.data" using 2:1 title 'boost' with linespoints lw 2

