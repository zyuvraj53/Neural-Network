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

  /// Making a XOR gate with the framework

  size_t stride = 3;
  size_t n = sizeof(td) / sizeof(td[0]) / 3;
  Mat ti = {
      .rows = n,
      .cols = 2,
      .stride = stride,
      .es = td};

  Mat to = {
      .rows = n,
      .cols = 1,
      .stride = stride,
      .es = td + 2};

  // MAT_PRINT(ti);
  // MAT_PRINT(to);

  XOR m = xor_alloc();
  XOR g = xor_alloc();

  mat_rand(m.w1, 0, 1);
  mat_rand(m.b1, 0, 1);
  mat_rand(m.w2, 0, 1);
  mat_rand(m.b2, 0, 1);

  float eps = 1e-1;
  float rate = 1e-1;

  printf("%f\n", cost(m, ti, to));

  for (size_t i = 0; i < 100 * 1000; i++) {
    finite_diff(m, g, eps, ti, to);
    xor_learn(m, g, rate);
  }
  printf("%f\n", cost(m, ti, to));

  printf("-------------\n");

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {

      MAT_AT(m.a0, 0, 0) = i;
      MAT_AT(m.a0, 0, 1) = j;

      forward_xor(m);

      float y = *(m.a2).es;

      printf("%d ^ %d = %f\n", i, j, y);
    }
  }

  return 0;
}