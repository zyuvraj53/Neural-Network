#ifndef NN_H_
#define NN_H_

//% float d[] = {0, 1, 0, 1}; // d for data
//%     Mat m = {.rows = 2, .cols = 2, .es = d};
//% or: Mat m = {.rows = 4, .cols = 1, .es = d};

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif //! NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif //! NN_ASSERT

typedef struct Mat {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

float rand_float(void);
float sigmoidf(float);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float x);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_print(Mat m, char *, size_t);
#define MAT_PRINT(m) mat_print(m, #m);


#endif //! NN_H_

// #define NN_IMPLEMENTATION // remove later or code crashes
#ifdef NN_IMPLEMENTATION

//~   x1-->●--w1-●b1(a1)
//~         \  /
//~          \w3
//~         / \
//~        /   w2
//~  x?-->●--w4--●b2(a2)

//~   a1 = x1w1 + x2w3 + b1
//~   a2 = x1w2 + x2w4 + b2

typedef struct {

  size_t count; // the amount of inner layers, w/o the input layer
  Mat *ws;      // weights
  Mat *bs;      // biases
  Mat *as;      // activations, the amount of activations is count + 1

  // There is one more activation layer in the NN, which is the input layer
  // First (input) Layer
  //% Mat a0;
  // Second Layer
  //% Mat w1, b1, a1;
  // Third (output) Layer
  //% Mat w2, b2, a2;
} NN;

void nn_print(NN m, char *);
#define NN_PRINT(nn) nn_print(nn, #nn) // doing #, stringifies it
void forward_xor(NN);

NN nn_alloc(size_t *, size_t);

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

// size_t arch[] = {2, 2, 1};
// NN nn = nn_alloc(arch, ARRAY_LEN(arch));

NN nn_alloc(size_t *architecture, size_t arch_count) {

  NN_ASSERT(arch_count > 0); //~ This is because there must be some number of inputs.

  NN nn;

  nn.count = arch_count - 1; // count doesn't include the input layer, but arch_count does include the input layer

  nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
  NN_ASSERT(nn.ws != NULL);
  nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
  NN_ASSERT(nn.bs != NULL);
  nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
  NN_ASSERT(nn.as != NULL);

  nn.as[0] = mat_alloc(1, architecture[0]); //~ The first(0-th) activation layer will be equal to the number of inputs

  for (size_t i = 1; i < arch_count; i++) {
    //~ the first weight matrix will have the same number of rows as the number of cols in the 0th activation or input matrix, and the number of colums will be mentioned in the architecture
    nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, architecture[i]);
    nn.bs[i - 1] = mat_alloc(1, architecture[i]);
    nn.as[i] = mat_alloc(1, architecture[i]);
  }

  return nn;
}

float cost(NN m, Mat ti, Mat to) { // ti == training input, to == training output
  assert(ti.rows == to.rows);
  assert(to.cols == m.as[2].cols);
  size_t n = ti.rows;

  float c = 0;
  for (size_t i = 0; i < n; i++) {
    Mat x = mat_row(ti, i);
    Mat y = mat_row(to, i);

    mat_copy(m.as[0], mat_row(ti, i));
    forward_xor(m);

    size_t q = to.cols;
    for (size_t j = 0; j < q; j++) {
      float d = MAT_AT(m.as[2], 0, j) - MAT_AT(y, 0, j);
      c += d * d;
    }
  }

  return c / n;
}

void forward_xor(NN m) {
  mat_dot(m.as[1], m.as[0], m.ws[1]);
  mat_sum(m.as[1], m.bs[1]);
  mat_sig(m.as[1]);

  mat_dot(m.as[2], m.as[1], m.ws[2]);
  mat_sum(m.as[2], m.bs[2]);
  mat_sig(m.as[2]);
}

void finite_diff(NN m, NN g, float eps, Mat ti, Mat to) { // m is the model, g is the gradient for the model
  float saved;

  float c = cost(m, ti, to);

  for (size_t i = 0; i < m.ws[1].rows; i++) {
    for (size_t j = 0; j < m.ws[1].cols; j++) {
      saved = MAT_AT(m.ws[1], i, j);
      MAT_AT(m.ws[1], i, j) += eps;
      MAT_AT(g.ws[1], i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.ws[1], i, j) = saved;
    }
  }

  for (size_t i = 0; i < m.bs[2].rows; i++) {
    for (size_t j = 0; j < m.bs[2].cols; j++) {
      saved = MAT_AT(m.bs[2], i, j);
      MAT_AT(m.bs[2], i, j) += eps;
      MAT_AT(g.bs[2], i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.bs[2], i, j) = saved;
    }
  }

  for (size_t i = 0; i < m.ws[2].rows; i++) {
    for (size_t j = 0; j < m.ws[2].cols; j++) {
      saved = MAT_AT(m.ws[2], i, j);
      MAT_AT(m.ws[2], i, j) += eps;
      MAT_AT(g.ws[2], i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.ws[2], i, j) = saved;
    }
  }

  for (size_t i = 0; i < m.bs[2].rows; i++) {
    for (size_t j = 0; j < m.bs[2].cols; j++) {
      saved = MAT_AT(m.bs[2], i, j);
      MAT_AT(m.bs[2], i, j) += eps;
      MAT_AT(g.bs[2], i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.bs[2], i, j) = saved;
    }
  }
}

void xor_learn(NN m, NN g, float rate) {

  for (size_t i = 0; i < m.ws[1].rows; i++) {
    for (size_t j = 0; j < m.ws[1].cols; j++) {
      MAT_AT(m.ws[1], i, j) -= rate * MAT_AT(g.ws[1], i, j);
    }
  }

  for (size_t i = 0; i < m.bs[1].rows; i++) {
    for (size_t j = 0; j < m.bs[1].cols; j++) {
      MAT_AT(m.bs[1], i, j) -= rate * MAT_AT(g.bs[1], i, j);
    }
  }

  for (size_t i = 0; i < m.ws[2].rows; i++) {
    for (size_t j = 0; j < m.ws[2].cols; j++) {
      MAT_AT(m.ws[2], i, j) -= rate * MAT_AT(g.ws[2], i, j);
    }
  }

  for (size_t i = 0; i < m.bs[2].rows; i++) {
    for (size_t j = 0; j < m.bs[2].cols; j++) {
      MAT_AT(m.bs[2], i, j) -= rate * MAT_AT(g.bs[2], i, j);
    }
  }
}

float rand_float(void) {
  return ((float)rand() / RAND_MAX);
}

float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols) {
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.es = (float *)NN_MALLOC(sizeof(*m.es) * rows * cols);
  NN_ASSERT(m.es != NULL);
  return m;
}

void mat_fill(Mat m, float x) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = x;
    }
  }
}

void mat_dot(Mat dst, Mat a, Mat b) {
  //%   1x(2   2)x3
  //% the innards should be equal, and the resultant matrix comes out to be 1x3

  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols * b.cols;
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == b.cols);

  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; k++) { //% We're iterating k times, the number of addn's we've to perform
        //%   i  (k  k)  j
        MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }
}

Mat mat_row(Mat m, size_t row) {
  return (Mat){
      .rows = 1,
      .cols = m.cols,
      .stride = m.stride,
      .es = &MAT_AT(m, row, 0)};
}

void mat_copy(Mat dst, Mat src) {
  NN_ASSERT(dst.rows == src.rows);
  NN_ASSERT(dst.cols == src.cols);

  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_sum(Mat dst, Mat a) {
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == a.cols);

  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_sig(Mat m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
    }
  }
}

void mat_print(Mat m, char *name, size_t padding) {
  printf("%*s%s = [\n",(int) padding,"", name);
  for (size_t i = 0; i < m.rows; i++) {
    printf("%*s      ",(int) padding, "");
    for (size_t j = 0; j < m.cols; j++) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s     ]\n", (int) padding, "");
}

void mat_rand(Mat m, float low, float high) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = rand_float() * (high - low) + low;
    }
  }
}

void nn_print(NN nn, char *name) {

  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.count; i++) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    mat_print(nn.ws[i], buf, 4);
    snprintf(buf, sizeof(buf), "ws%zu", i);
    mat_print(nn.bs[i], buf, 4);
  }

  printf("]\n");
}

#endif // NN_IMPLEMENTATION