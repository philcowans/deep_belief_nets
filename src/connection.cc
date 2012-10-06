#include "connection.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <gsl/gsl_blas.h>

Connection::Connection(Layer *below, Layer *above, Layer *below_input) {
  m_above = above;
  m_below = below;
  m_below_input = below_input;
  m_num_above = above->size(false);
  m_num_below = below->size(true);
  if(m_below_input)
    m_num_below_input = m_below_input->size(true);
  m_weights = gsl_matrix_alloc(m_num_above, m_num_below);
  if(m_below_input)
    m_weights_input = gsl_matrix_alloc(m_num_above, m_num_below_input);
  else
    m_weights_input = NULL;
  m_deltas = gsl_matrix_alloc(m_num_above, m_num_below);
  m_weights_down = gsl_matrix_alloc(m_num_above, m_num_below);
}

Connection::~Connection() {
  gsl_matrix_free(m_weights);
  gsl_matrix_free(m_deltas);
  gsl_matrix_free(m_weights_down);
  if(m_weights_input)
    gsl_matrix_free(m_weights_input);
}

double Connection::get_weight(int i, int j) {
  return gsl_matrix_get(m_weights, i, j);
}

void Connection::set_weight(int i, int j, double v) {
  gsl_matrix_set(m_weights, i, j, v);
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
  gsl_vector_memcpy(m_above->activation(false), m_above->biases(false));
  gsl_blas_dgemv(CblasNoTrans, 1.0, m_weights, m_below->state(true), 1.0, m_above->activation(false));
  for(int i = 0; i < m_num_above; ++i) {
    gsl_vector_set(m_above->p(false), i, 1.0 / (1.0 + exp(-gsl_vector_get(m_above->activation(false), i))));
  }
}

void Connection::find_probs_downwards() {
  gsl_vector_memcpy(m_below->activation(true), m_below->biases(true));
  gsl_blas_dgemv(CblasTrans, 1.0, m_weights, m_above->state(false), 1.0, m_below->activation(true));
  for(int i = 0; i < m_num_below; ++i) {
    gsl_vector_set(m_below->p(true), i, 1.0 / (1.0 + exp(-gsl_vector_get(m_below->activation(true), i))));
  }
  if(m_below_input) {
    gsl_vector_memcpy(m_below_input->activation(true), m_below_input->biases(true));
    gsl_blas_dgemv(CblasTrans, 1.0, m_weights_input, m_above->state(false), 1.0, m_below_input->activation(true));
    for(int i = 0; i < m_num_below_input; ++i) {
      gsl_vector_set(m_below_input->p(true), i, 1.0 / (1.0 + exp(-gsl_vector_get(m_below_input->activation(true), i))));
    }
  }
}

void Connection::propagate_observation(gsl_rng *rng, bool mean_field) {
  find_probs_upwards();
 if(mean_field) {
    m_above->transfer();
 }
  else {
    m_above->sample(rng);
  }
}

void Connection::propagate_hidden(gsl_rng *rng, bool ext) {
  find_probs_downwards();
  m_below->sample(rng, ext);
  if(m_below_input)
    m_below_input->sample(rng, ext);
}

void Connection::perform_update_step(gsl_rng *rng) {
//   std::cout << "t: ";
//   for(int i = 0; i < m_num_below; ++i) {
//     std::cout << gsl_vector_get(m_below->state(true), i) << " ";
//   }
//   std::cout << std::endl;
  
  // Assume that we already have suitable input data in place at this stage
  double epsilon = 1.0;

  reset_deltas();
  m_below->reset_deltas();
  m_above->reset_deltas();

  // Positive phase
  
  propagate_observation(rng);

  gsl_blas_dger (epsilon, m_above->state(false), m_below->state(true), m_deltas);
  gsl_blas_daxpy(epsilon, m_below->state(true),                        m_below->deltas(true));
  gsl_blas_daxpy(epsilon, m_above->state(false),                       m_above->deltas(false));

  // Negative phase

  propagate_hidden(rng);
  find_probs_upwards();
  
  gsl_blas_dger (-epsilon, m_above->p(false), m_below->state(true), m_deltas); 
  gsl_blas_daxpy(-epsilon, m_below->state(true),                    m_below->deltas(true));
  gsl_blas_daxpy(-epsilon, m_above->p(false),                       m_above->deltas(false));

  
  // Then commit the update

  commit_deltas(); // TODO: Handle weights layer here too
  m_below->commit_deltas();
  m_above->commit_deltas();
}

void Connection::sample_layer(gsl_rng *rng, int num_iterations, int label) {
  // Assume that sensible values are already loaded into layer above's activity
  //  gsl_vector_set_zero(m_below->state(true));
  m_below->activate_from_bias();
  m_below->sample(rng, false);
  m_below->set_label(label);
  propagate_observation(rng);
  for(int i = 0; i < num_iterations; ++i) {
    propagate_hidden(rng, false);
    propagate_observation(rng);
    //    for(int j = 0; j < m_num_below; ++j) {
    //  std::cout << gsl_vector_get(m_below->state(true), j) << " ";
    // }
    //std::cout << std::endl;
  }
}

int Connection::find_label() {
  m_below->set_label(-1); // Hack, but this should work
  find_probs_upwards();
  m_above->transfer();
  find_probs_downwards();
  return m_below->most_probable_label();
}
