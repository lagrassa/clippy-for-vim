CFLAGS = -O3
#CFLAGS = -g
LDLIBS = -lm
CC = gcc

gjkdemo:	gjkdemo.o gjk.o
	$(CC) $(CFLAGS) gjkdemo.o gjk.o -o gjkdemo -lm 

gjk:	gjkHPN.o

clean:
	rm -f *.o core

.c.o:	gjk.h
	$(CC) -c $(CFLAGS) $*.c
