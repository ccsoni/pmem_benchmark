#!/bin/bash

nshift=(10 12 14 16 18 20 22 24 26 28)
nomp=(32)

for nomp in "${nomp[@]}" ; do
	for nshift in "${nshift[@]}" ; do
		echo ${nshift} ${nomp}
		./pmem_io_single ${nshift} ${nomp} > pmem_io_single_${nshift}_${nomp}.dat
	done
done
