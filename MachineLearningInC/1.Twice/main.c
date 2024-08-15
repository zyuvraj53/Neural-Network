#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//~ We will be making a Mathematical Model which will predict the next number in a sequence.

float train[][2] = {
    {0.0, 0.0},
    {0.1, 0.2},
    {0.2, 0.4},
    {0.3, 0.6},
    {0.4, 0.8},
}; // # The first number will be the input of the model, and the second number will be what we expect from the model.

#define train_count (sizeof(train) / sizeof(train[0]))

//% y = x*w , the ai only knows this, with an unknown 'w'.
//% we can also have more parameters (w's), 2 to billions

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

float cost(float w, float b) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; i++) {
    float x = train[i][0];
    float y = x * w + b;
    //~ We want to find the difference between the randomly generated y and the actual y
    float d = y - train[i][1];
    result += d * d;
  }
  result /= (size_t)train_count;

  return result;
}

float dcost(float w) {
  float result = 0.0f;
  size_t n = train_count;
  for (size_t i = 0; i < n; i++) {
    float x = train[i][0];
    float y = train[i][1];

    result += 2 * (x * w - y) * x;
  }
  result /= n;

  return result;
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

  float eps = 1e-3;
  float alpha = 1e-3; // since the derivative will be a very large number we'll mult by alpha or learning rate.
  printf("starting cost: %f\n", cost(w, b));

  for (size_t i = 0; i < 50; i++) {
    float dw = (cost(w + eps, b) - cost(w, b)) / eps; // this is the general defn of a derivative.
    float db = (cost(w, b + eps) - cost(w, b)) / eps; // this is the general defn of a derivative.
    w -= alpha * dw;
    b -= alpha * db;
    printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
  }
  printf("6*w + b = %f\n", 6 * w + b);

  //~ The next segment actually computes the derivative but only has one weight, and no activation function

  //%  NOTE: say  C(w)  = 1/n * (sum_from_i=1_to_i=n (x_i*w - y_i)^2)
  //%  NOTE: Then C'(w) = (1/n * (sum_from_i=1_to_i=n (x_i*w - y_i)^2))'
  //%  NOTE:            = 1/n * ((sum_from_i=1_to_i=n (x_i*w - y_i)^2))'
  //%  NOTE:            = 1/n * (sum_from_i=1_to_i=n (2 * (x_i*w - y_i) * x_i))

  printf("\n\n Differentiation starts: -----\n\n");

  w = rand_float(); // function which returns a random number between 0 and 1.

  // @ We have still not applied an activation function and this only has one weight

  alpha = 3e-1;
  printf("Starting weight: %f\n", w);

  for (size_t i = 0; i < 50; i++) {
    float dw = dcost(w);
    w -= alpha * dw;
    printf("dcost = %f, w = %f\n", dcost(w), w);
  }

  printf("6 * w = %f\n", 6 * w);

  //~ The function given below will backpropagate using gradient descent

  printf("\n\n GD starts: -----\n\n");

  w = rand_float(); // function which returns a random number between 0 and 1.
  b = rand_float(); // function which returns a random number between 0 and 1.

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
