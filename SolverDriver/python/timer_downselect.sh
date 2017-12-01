#!/bin/bash

csv="$1"
timer_list=${HOME}/src/TrilinosDrivers/SolverDriver/python/timer_downselect.txt
output=${csv}.reduced

grep -E -f ${timer_list} ${csv} > ${output}

head -n1 ${csv} | cat - ${output} > temp && mv temp ${output}

