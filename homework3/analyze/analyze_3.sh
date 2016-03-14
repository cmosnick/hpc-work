#!/bin/bash

grep "Point:interleavethreads" p1.out | awk -F':' '{printf("%s,\t%s,\t%s\n",$3,$4,$5);}' > p1.interleave.threads.data && \
grep "Point:blockthreads" p1.out | awk -F':' '{printf("%s,\t%s,\t%s\n",$3,$4,$5);}' > p1.block.threads.data && \
grep "Point:interleavefiles" p1.out | awk -F':' '{printf("%s,\t%s,\t%s\n",$3,$4,$5);}' > p1.interleave.files.data && \
grep "Point:blockfiles" p1.out | awk -F':' '{printf("%s,\t%s,\t%s\n",$3,$4,$5);}' > p1.block.files.data
