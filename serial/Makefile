CC=icx
CFLAGS=-O3
CFLAGS+=-qopenmp

PMEM_IO_SERIAL_OBJ = pmem_io_serial.o
PMEM_IO_SERIAL_DEP := $(PMEM_IO_SERIAL_OBJ)

pmem_io_serial: $(PMEM_IO_SERIAL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lpmem -lm


clean:
	-rm -f *.o

distclean: clean
	-rm -f *~
	-rm -f pmem_io_serial
