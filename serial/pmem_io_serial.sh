#!/bin/bash

nshift=(10 12 14 16 18 20 22 24 26 28)
nomp=(16 32)

for nomp_ in "${nomp[@]}" ; do
	for nshift_ in "${nshift[@]}" ; do
		echo ${nshift_} ${nomp_}
		./pmem_io_serial ${nshift_} ${nomp_} > pmem_io_serial_${nshift_}_${nomp_}.dat
	done
done
