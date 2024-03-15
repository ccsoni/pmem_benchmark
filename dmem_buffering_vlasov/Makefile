CC=icx
CFLAGS=-O3 -mavx2

ALL = dmem_buffering_normal dmem_buffering_pmem_loop dmem_buffering_pmem_memcpy

all: $(ALL)

dmem_buffering_normal: dmem_buffering.c
	$(CC) $(CFLAGS) -o $@ $^ -lpmem -lm -fiopenmp

dmem_buffering_pmem_loop: dmem_buffering.c
	$(CC) $(CFLAGS) -D__PMEM_LOOP__ -o $@ $^ -lpmem -lm -fiopenmp

dmem_buffering_pmem_memcpy: dmem_buffering.c
	$(CC) $(CFLAGS) -D__PMEM_MEMCPY__ -o $@ $^ -lpmem -lm -fiopenmp

clean:
	\rm -rf *.o
	\rm -rf $(ALL)

distclean:
	\rm -rf $(ALL)
	\rm -rf *~
