#include "connection.h"

#include <cmath>
#include <cstring>
#include <gsl/gsl_blas.h>

Connection::Connection(Layer *below, Layer *above) {
  m_above = above;
  m_below = below;
  m_num_above = above->size();
  m_num_below = below->size();
  m_weights = gsl_matrix_alloc(m_num_above, m_num_below);
  m_deltas = gsl_matrix_alloc(m_num_above, m_num_below);
  m_activation_above = gsl_vector_alloc(m_num_above);
  m_activation_below = gsl_vector_alloc(m_num_below);
}

Connection::~Connection() {
  gsl_matrix_free(m_weights);
  gsl_matrix_free(m_deltas);
  gsl_vector_free(m_activation_below);
  gsl_vector_free(m_activation_above);
}

double Connection::get_weight(int i, int j) {
  return gsl_matrix_get(m_weights, i, j);
}

void Connection::update_weights(int i, int j, double delta) {
  gsl_matrix_set(m_deltas, i, j, gsl_matrix_get(m_deltas, i, j) + delta);
}

void Connection::reset_deltas() {
  gsl_matrix_set_zero(m_deltas);
}

void Connection::commit_deltas() {
  gsl_matrix_add(m_weights, m_deltas);
}

void Connection::find_probs_upwards() {
  gsl_vector_memcpy(m_activation_above, m_above->m_biases);
  gsl_blas_dgemv(CblasNoTrans, 1.0, m_weights, m_below->m_state, 1.0, m_activation_above);
  for(int i = 0; i < m_num_above; ++i) {
    gsl_vector_set(m_above->m_p, i, 1.0 / (1.0 + exp(-gsl_vector_get(m_activation_above, i))));
  }
}

void Connection::find_probs_downwards() {
  gsl_vector_memcpy(m_activation_below, m_below->m_biases);
  gsl_blas_dgemv(CblasTrans, 1.0, m_weights, m_above->m_state, 1.0, m_activation_below);
  for(int i = 0; i < m_num_below; ++i) {
    gsl_vector_set(m_below->m_p, i, 1.0 / (1.0 + exp(-gsl_vector_get(m_activation_below, i))));
  }
}

void Connection::propagate_observation(gsl_rng *rng) {
  find_probs_upwards();
  m_above->sample(rng);
}

void Connection::propagate_hidden(gsl_rng *rng) {
  find_probs_downwards();
  m_below->sample(rng);
}

void Connection::perform_update_step(gsl_rng *rng) {
  // Assume that we already have suitable input data in place at this stage

  double epsilon = 0.001;

  reset_deltas();
  m_below->reset_deltas();
  m_above->reset_deltas();
  
  find_probs_upwards();
  m_above->sample(rng);
    
  gsl_blas_dger(epsilon, m_above->m_state, m_below->m_state, m_deltas);
  
  m_below->update_biases(epsilon, true, true);
  m_above->update_biases(epsilon, true, true);
  
  find_probs_downwards();
  m_below->sample(rng);
  find_probs_upwards();
  
  gsl_blas_dger(-epsilon, m_above->m_p, m_below->m_state, m_deltas); 

  m_below->update_biases(epsilon, false, true);
  m_above->update_biases(epsilon, false, false);
  
  commit_deltas();
  m_below->commit_deltas();
  m_above->commit_deltas();
}

void Connection::sample_layer(gsl_rng *rng, int num_iterations) {
  // Assume that sensible values are already loaded into layer above's activity
  for(int i = 0; i < num_iterations; ++i) {
    find_probs_downwards();
    m_below->sample(rng);
    find_probs_upwards();
    m_above->sample(rng);
  }
}
