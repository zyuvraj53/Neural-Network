#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//~ Sometimes, NN's fail to come up with a model for XOR gate, for instance, once my output looked like this:
//~         0 ^ 0 = 0.015308
//~         0 ^ 1 = 0.982008
//~         1 ^ 0 = 0.499481
//~         1 ^ 1 = 0.500578
//~ This is not uncommon, it depends on the initialization of the weights.
//~ Perhaps, we need to introduce more neurons, idk...

typedef struct {
  float or_w1;
  float or_w2;
  float or_b;

  float nand_w1;
  float nand_w2;
  float nand_b;

  float and_w1;
  float and_w2;
  float and_b;
} XOR;

typedef float sample[3]; //. Nice trick

sample xor_train[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}};

float rand_float(void) {
  return ((float)rand() / (float)RAND_MAX); // returns a number between 0 and 1
}

float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

float forward(XOR m, float x1, float x2) { //% m is a model
  float a = sigmoidf(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b);
  float b = sigmoidf(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b);

  return sigmoidf(a * m.and_w1 + b * m.and_w2 + m.and_b);
}

sample *train = xor_train;

#define train_count 4

float cost(XOR m) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = forward(m, x1, x2);

    float d = y - train[i][2];

    result += d * d;
  }
  result /= train_count;

  return result;
}

XOR rand_xor() {
  XOR m;

  m.or_w1 = rand_float();
  m.or_w2 = rand_float();
  m.or_b = rand_float();

  m.nand_w1 = rand_float();
  m.nand_w2 = rand_float();
  m.nand_b = rand_float();

  m.and_w1 = rand_float();
  m.and_w2 = rand_float();
  m.and_b = rand_float();

  return m;
}

void print_xor(XOR m) {

  printf("or_w1 = %f\n", m.or_w1);
  printf("or_w2 = %f\n", m.or_w2);
  printf("or_b = %f\n", m.or_b);

  printf("nand_w1 = %f\n", m.nand_w1);
  printf("nand_w2 = %f\n", m.nand_w2);
  printf("nand_b = %f\n", m.nand_b);

  printf("and_w1 = %f\n", m.and_w1);
  printf("and_w2 = %f\n", m.and_w2);
  printf("and_b = %f\n", m.and_b);
}

XOR finite_diff(XOR m, float eps) {
  XOR g;

  float c = cost(m);
  float saved;

  saved = m.or_w1;
  m.or_w1 += eps;
  g.or_w1 = (cost(m) - c) / eps;
  m.or_w1 = saved;

  saved = m.or_w2;
  m.or_w2 += eps;
  g.or_w2 = (cost(m) - c) / eps;
  m.or_w2 = saved;

  saved = m.or_b;
  m.or_b += eps;
  g.or_b = (cost(m) - c) / eps;
  m.or_b = saved;

  //------------------

  saved = m.nand_w1;
  m.nand_w1 += eps;
  g.nand_w1 = (cost(m) - c) / eps;
  m.nand_w1 = saved;

  saved = m.nand_w2;
  m.nand_w2 += eps;
  g.nand_w2 = (cost(m) - c) / eps;
  m.nand_w2 = saved;

  saved = m.nand_b;
  m.nand_b += eps;
  g.nand_b = (cost(m) - c) / eps;
  m.nand_b = saved;

  //-------------------

  saved = m.and_w1;
  m.and_w1 += eps;
  g.and_w1 = (cost(m) - c) / eps;
  m.and_w1 = saved;

  saved = m.and_w2;
  m.and_w2 += eps;
  g.and_w2 = (cost(m) - c) / eps;
  m.and_w2 = saved;

  saved = m.and_b;
  m.and_b += eps;
  g.and_b = (cost(m) - c) / eps;
  m.and_b = saved;

  return g;
}

XOR learn(XOR m, XOR g, float rate) {
  m.or_w1 -= rate * g.or_w1;
  m.or_w2 -= rate * g.or_w2;
  m.or_b -= rate * g.or_b;

  m.nand_w1 -= rate * g.nand_w1;
  m.nand_w2 -= rate * g.nand_w2;
  m.nand_b -= rate * g.nand_b;

  m.and_w1 -= rate * g.and_w1;
  m.and_w2 -= rate * g.and_w2;
  m.and_b -= rate * g.and_b;

  return m;
}

int main(void) {

  srand(time(NULL));

  XOR m = rand_xor();

  float eps = 1e-3;
  float rate = 1e-1;

  for (int i = 0; i < 100000; i++) {
    XOR g = finite_diff(m, eps); // the values that we need to subtract
    m = learn(m, g, rate);
    // printf("%f\n", cost(m));
  }

  printf("------------\n");

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      printf("%u ^ %u = %f\n", i, j, forward(m, i, j));

  return 0;
}