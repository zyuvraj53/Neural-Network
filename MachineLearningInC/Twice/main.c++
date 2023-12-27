#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//~ We will be making a Mathematical Model which will predict the next number in a sequence.

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
}; // # The first number will be the input of the model, and the second number will be what we expect from the model.

#define train_count (sizeof(train) / sizeof(train[0]))

//% y = x*w , the ai only knows this, with an unknown 'w'.
//% we can also have more parameters (w's), 2 to billions

float rand_float(void) {
  return ((float)rand() / (float)RAND_MAX); // returns random float between 0 and 1
}

float cost(float w, float b) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; i++) {
    float x = train[i][0];
    float y = x * w + b;
    //~ We want to find the difference between the randomly generated y and the actual y
    float d = y - train[i][1];
    result += d * d;
  }
  result /= train_count;

  return result;
}

int main() {

  srand(time(0));

  float w = rand_float() * 10.0f; // function which returns a random number between 0 and 1.
  float b = rand_float() * 5.0f; // function which returns a random number between 0 and 1.

  float eps = 1e-3;
  float alpha = 1e-3; // since the derivative will be a very large number we'll mult by alpha or learning rate.
  printf("%f\n", cost(w, b));

  for (size_t i = 0; i < 500; i++) {
    float dw = (cost(w + eps, b) - cost(w, b)) / eps; // this is the general defn of a derivative.
    float db = (cost(w, b + eps) - cost(w, b)) / eps; // this is the general defn of a derivative.
    w -= alpha * dw;
    b -= alpha * db;
    printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
  }
  return 0;
}