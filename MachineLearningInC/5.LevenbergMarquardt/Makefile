CC=gcc
OPT=-O3
CFLAGS=-Wall -Wextra -Wshadow $(OPT)
CPPFILES=main.c
OBJECTS=main.o

output: $(OBJECTS)
	$(CC) $(CFAGS) $(CPPFILES) -o output.out -lm

 main.o: main.c
	$(CC) $(CFLAGS) -c main.c

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o output.*

# target: dependencies
# 	action
