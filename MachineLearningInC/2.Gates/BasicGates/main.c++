#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//~ Mathematical model for the Basic Gates

typedef float sample[3]; //. Nice trick

sample or_train[] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 1}
};

sample and_train[] = {
  {0, 0, 0},
  {0, 1, 0},
  {1, 0, 0},
  {1, 1, 1}
};
  
sample nand_train[] = {
  {0, 0, 1},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};

sample *train = and_train;

#define train_count 4

float rand_float(void) {
  return ((float)rand() / (float)RAND_MAX); // returns a number between 0 and 1
}

float sigmoidf(float x){
  return 1.f/(1.f + expf(-x));
}

float cost(float w1, float w2, float b) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = sigmoidf(x1 * w1 + x2 * w2 + b); // this squeezes the value of the linear combination between 0 and 1

    float d = y - train[i][2];

    result += d * d;
  }
  result /= train_count;

  return result;
}

int main() {

  srand(time(0));

  float w1 = rand_float();
  float w2 = rand_float();

  //% We had to add a b, as if x1 = 0, and x2 = 0, it would make the whole arg 0, and sigf(0) is 1/2...

  float b  = rand_float();

  printf("w1 = %f, w2 = %f, b = %f, c = %f\n", w1, w2, b, cost(w1, w2, b));

  float eps = 1e-4;
  float rate = 1e-1;
  for (int i = 0; i < 10000; i++) {
    float c = cost(w1, w2, b);

    float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
    float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
    float db  = (cost(w1, w2, b + eps) - c) / eps;

    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b -= rate * db;
  }

  printf("w1 = %f, w2 = %f, b = %f, c = %f\n", w1, w2, b, cost(w1, w2, b));

  printf("-------------------------\n");

  for(int i = 0; i < 2; i++){
    for(int j = 0; j < 2; j++){
      printf("%d | %d = %f\n", i , j, sigmoidf(w1*i + w2*j + b));
    }
  }

  return 0;
}