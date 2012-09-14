#include "layer.h"

#include <cmath>
#include <cstring>
#include <gsl/gsl_blas.h>

Layer::Layer(int size) {
  m_size = size;
  m_biases = gsl_vector_calloc(m_size);
  m_deltas = gsl_vector_alloc(m_size);
  m_state = gsl_vector_alloc(m_size);
  m_p = gsl_vector_alloc(m_size);
}

Layer::~Layer() {
  gsl_vector_free(m_biases);
  gsl_vector_free(m_deltas);
  gsl_vector_free(m_state);
  gsl_vector_free(m_p);
}

int Layer::size() {
  return m_size;
}

double Layer::get_bias(int i) {
  return gsl_vector_get(m_biases, i);
}

void Layer::reset_deltas() {
  // TODO: 'double buffering'
  gsl_vector_set_zero(m_deltas);
}

void Layer::commit_deltas() {
  gsl_vector_add(m_biases, m_deltas);
}

void Layer::sample(gsl_rng *rng) {
  for(int i = 0; i < m_size; ++i) {
    gsl_vector_set(m_state, i, gsl_rng_uniform(rng) < gsl_vector_get(m_p, i));
  }
}

void Layer::set_state(const gsl_vector *state) {
  gsl_vector_memcpy(m_state, state);
}

void Layer::activate_from_bias() {
  for(int i = 0; i < m_size; ++i) {
    gsl_vector_set(m_p, i, 1.0 / (1.0 + exp(-gsl_vector_get(m_biases, i))));
  }
}
  
  
