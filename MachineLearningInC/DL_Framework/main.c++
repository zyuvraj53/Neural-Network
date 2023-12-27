#define NN_IMPLEMENTATION
#include <time.h>
#include "nn.hpp"

//~   x?-->?--w?-?b?(a?)
//~         \  /
//~          \w?
//~         / \
//~        /   w?
//~  x?-->?--w?--?b?(a?)

//~   a? = x?w? + x?w? + b?
//~   a? = x?w? + x?w? + b?


int main(void){

  srand(time(NULL));

  Mat a = mat_alloc(1, 2);
  mat_rand(a, 5, 10);
  mat_print(a);

  printf("----------------\n");

  Mat b = mat_alloc(2, 2);
  mat_fill(b, 1);
  mat_print(b);

  printf("----------------\n");

  Mat dst = mat_alloc(1, 2);
  mat_dot(dst, a, b);
  mat_print(dst);

  return 0;
}