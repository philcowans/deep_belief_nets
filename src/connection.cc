#include "connection.h"

#include <cstring>

Connection::Connection(Layer *below, Layer *above) {
  m_num_above = above->size();
  m_num_below = below->size();
  m_weights = new double[m_num_above * m_num_below];
  memset(m_weights, 0, m_num_above * m_num_below * sizeof(double));
}

Connection::~Connection() {
  delete[] m_weights;
}

double Connection::get_weight(int i, int j) {
  return m_weights[i + j * m_num_above];
}

void Connection::update_weights(double *delta) {
  for(int i = 0; i < m_num_above * m_num_below; ++i) {
    m_weights[i] += delta[i];
  }
}
