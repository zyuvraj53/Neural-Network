CC=g++
OPT=-O3
CFLAGS=-Wall -Wextra -Wshadow $(OPT)
CPPFILES=twice.c++
OBJECTS=twice.o

output: $(OBJECTS)
	$(CC) $(CFAGS) $(CPPFILES) -o output

twice.o: twice.c++
	$(CC) $(CFLAGS) -c twice.c++

%.o: %.c++ %.hpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o output.*

# target: dependencies
# 	action