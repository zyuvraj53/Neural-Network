#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0};

int main(void) {

  srand(time(NULL));

  size_t arch[] = {2, 2, 1};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN_PRINT(nn);

  return 0;
}