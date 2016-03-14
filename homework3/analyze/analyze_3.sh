#!/bin/bash

grep "Point:interleave" p1.out | awk -F':' '{printf("%s,\t%s,\t%s\n",$3,$4,$5);}' > p1.interleave.threads.data && \
grep "Point:block" p1.out | awk -F':' '{printf("%s,\t%s,\t%s\n",$3,$4,$5);}' > p1.block.threads.data



