CC=icx
CFLAGS=-O3
CFLAGS+=-qopenmp

PMEM_IO_SINGLE_OBJ = pmem_io_single.o
PMEM_IO_SINGLE_DEP := $(PMEM_IO_SINGLE_OBJ)

pmem_io_single: $(PMEM_IO_SINGLE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lpmem -lm


clean:
	-rm -f *.o

distclean: clean
	-rm -f *~
	-rm -f pmem_io_single
