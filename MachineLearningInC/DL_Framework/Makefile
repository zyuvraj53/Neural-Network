CC=g++
OPT=-O3
CFLAGS=-Wall -Wextra -Wshadow $(OPT)
CPPFILES=main.c++ nn.hpp
OBJECTS=main.o nn.o

output: $(OBJECTS)
	$(CC) $(CFAGS) $(CPPFILES) -o output.exe

main.o: main.c++
	$(CC) $(CFLAGS) -c -lm main.c++

%.o: %.hpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o output.* *.gch

# target: dependencies
# 	action