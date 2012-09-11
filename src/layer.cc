#include "layer.h"

#include <cstring>

Layer::Layer(int size) {
  m_size = size;
  m_biases = new double[size];
  memset(m_biases, 0, size * sizeof(double));
}

Layer::~Layer() {
  delete[] m_biases;
}

int Layer::size() {
  return m_size;
}

double Layer::get_bias(int i) {
  return m_biases[i];
}

void Layer::update_biases(double *delta) {
  for(int i = 0; i < m_size; ++i) {
    m_biases[i] += delta[i];
  }
}
