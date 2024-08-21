#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
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

float lm_cost_w(float w, float b, float lambda_w) {
  float result = 0.0f;
  size_t n = train_count;
  for (size_t i = 0; i < n; i++) {
    float x = train[i][0];
    float y = train[i][1];

    float y_cap = sigmoidf(x * w + b);

    float c = (y_cap - 1)*(y_cap * y_cap + (y_cap - y) * y_cap);
    float d = (y_cap - y)*y_cap;

    result += (4 * c * d) * (1 / ((x + lambda_w) * (1 + lambda_w) - x * x)) * (x * (1 + lambda_w) - x);
  }

  result /= n;

  return result;
}

float lm_cost_b(float w, float b, float lambda_b) {
  float result = 0.0f;
  size_t n = train_count;
  for (size_t i = 0; i < n; i++) {
    float x = train[i][0];
    float y = train[i][1];

    float y_cap = sigmoidf(x * w + b);

    float c = (y_cap - 1)*(y_cap * y_cap + (y_cap - y) * y_cap);
    float d = (y_cap - y)*y_cap;

    result += (4 * c * d) * (1 / ((x + lambda_b) * (1 + lambda_b) - x * x)) * (-x * x + x + lambda_b);
  }

  result /= n;

  return result;
}

int main() {

  srand(time(0));

  float w = rand_float(); // function which returns a random number between 0 and 1.
  float b = rand_float(); // function which returns a random number between 0 and 1.

  //~ The below given function computes the limit as h tends to zero for a cost function

  float lambda_w = 100;
  float lambda_b = 100;
  printf("Starting weight: %f, and bias: %f\n", w, b);

  float *d = (float*) malloc(2 * sizeof(float));

  float dw = 0.0f, db = 0.0f;

  for (size_t i = 0; i < 1000; i++) {

    float prev_dw = dw;
    float prev_db = db;

    dw = lm_cost_w(w, b, lambda_w);
    db = lm_cost_b(w, b, lambda_b);

    w -= dw;
    b -= db;
    printf("dw = %f, db = %f, w = %f, b = %f, lambda_w = %f,lambda_b = %f\n", dw, db, w, b, lambda_w, lambda_b);

    //if(prev_db > db){
      lambda_b /= 2;
    //}else{
      //lambda_b *= 2;
    //}

    //if(prev_dw > dw){
      lambda_w /= 2;
    //}else{
      //lambda_w *= 2;
    //}
  }
  printf("6*w + b = %f\n", 6 * w + b);
  free(d);

  return 0;
}
