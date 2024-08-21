#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
    {0.0, 0.0},
    {0.1, 0.2},
    {0.2, 0.4},
    {0.3, 0.6},
    {0.4, 0.8},
}; 

#define train_count (sizeof(train) / sizeof(train[0]))

float rand_float(void) {
  return ((float)rand() / (float)RAND_MAX); // returns random float between 0 and 1
}

float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

float ReLUf(float x) {
  if (x > 0)
    return x;
  else
    return 0;
}

float tanhf(float x) {
  return 2 * sigmoidf(2 * x) - 1;
}

float gcost_w(float w, float b) {
  float result = 0.0f;
  size_t n = train_count;
  for (size_t i = 0; i < n; i++) {
    float x = train[i][0];
    float y = train[i][1];

    float a = sigmoidf(x * w + b);

    result += 2 * (a - y) * a * (1 - a) * x;
  }
  result /= n;

  return result;
}

float gcost_b(float w, float b) {
  float result = 0.0f;
  size_t n = train_count;
  for (size_t i = 0; i < n; i++) {
    float x = train[i][0];
    float y = train[i][1];

    float a = sigmoidf(x * w + b);

    result += 2 * (a - y) * a * (1 - a);
  }
  result /= n;

  return result;
}

int main() {

  srand(time(0));

  float w = rand_float(); // function which returns a random number between 0 and 1.
  float b = rand_float(); // function which returns a random number between 0 and 1.

  //~ The below given function computes the limit as h tends to zero for a cost function

  float alpha = 2;

  alpha = 2;
  printf("Starting weight: %f, and bias: %f\n", w, b);

  for (size_t i = 0; i < 50; i++) {
    float dw = gcost_w(w, b);
    float db = gcost_b(w, b);
    w -= alpha * dw;
    b -= alpha * db;
    printf("gcost_w = %f, gcost_b = %f, w = %f, b = %f\n", gcost_w(w, b), gcost_b(w, b), w, b);
  }
  printf("6*w + b = %f\n", 6 * w + b);

  return 0;
}
