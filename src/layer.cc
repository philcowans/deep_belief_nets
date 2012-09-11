#include "layer.h"

Layer::Layer(int size) {
  m_size = size;
}

int Layer::size() {
  return m_size;
}

double Layer::get_bias(int i) {
  return 0.0;
}

void Layer::update_biases(double *delta) {
}
