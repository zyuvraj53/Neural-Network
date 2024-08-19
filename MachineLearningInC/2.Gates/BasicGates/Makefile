CC=g++
OPT=-O3
CFLAGS=-Wall -Wextra -Wshadow $(OPT)
CLIBS=-lm
CPPFILES=main.c++
OBJECTS=main.o

output: $(OBJECTS)
	$(CC) $(CFAGS) $(CPPFILES) $(CLIBS) -o output

main.o: main.c++
	$(CC) $(CFLAGS) -c main.c++

%.o: %.c++ %.hpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o output.*

# target: dependencies
# 	action