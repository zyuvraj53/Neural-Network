CC=gcc
OPT=-O3
CFLAGS=-Wall -Wextra -Wshadow $(OPT)
CPPFILES=main.c nn.h
OBJECTS=main.o nn.o

output: $(OBJECTS)
	$(CC) $(CFAGS) $(CPPFILES) -o output.exe -lm

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

%.o: %.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o output.* *.gch *.exe

# target: dependencies
# 	action