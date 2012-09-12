#include "layer.h"

#include <cstring>

Layer::Layer(int size) {
  m_size = size;
  m_biases = new double[size];
  m_deltas = new double[size];
  memset(m_biases, 0, size * sizeof(double));
}

Layer::~Layer() {
  delete[] m_biases;
  delete[] m_deltas;
}

int Layer::size() {
  return m_size;
}

double Layer::get_bias(int i) {
  return m_biases[i];
}

void Layer::update_biases(int i, double delta) {
  m_deltas[i] += delta;
}

void Layer::reset_deltas() {
  memset(m_deltas, 0, m_size * sizeof(double));
}

void Layer::commit_deltas() {
  for(int i = 0; i < m_size; ++i) {
    m_biases[i] += m_deltas[i];
  }
}
