CC=mpicc
CFLAGS=-O3 -fiopenmp

SEND_RECV_OBJ = send_recv.o
SEND_RECV_DEP := $(SEND_RECV_OBJ)

BCAST_OBJ = bcast.o
BCAST_DEP := $(BCAST_OBJ)

ALLGATHER_OBJ = allgather.o
ALLGATHER_DEP := $(ALLGATHER_OBJ)

ALL = send_recv bcast allgather

all: $(ALL)

send_recv: $(SEND_RECV_DEP)
	$(CC) $(CFLAGS) -o $@ $^ -lpmem

bcast: $(BCAST_DEP)
	$(CC) $(CFLAGS) -o $@ $^ -lpmem

allgather: $(ALLGATHER_DEP)
	$(CC) $(CFLAGS) -o $@ $(ALLGATHER_OBJ) -lpmem

clean:
	-rm -f *.o

distclean: clean
	-rm -f *~
	-rm -f $(ALL)

