#include "connection.h"

#include <cmath>
#include <cstring>

Connection::Connection(Layer *below, Layer *above) {
  m_above = above;
  m_below = below;
  m_num_above = above->size();
  m_num_below = below->size();
  m_weights = new double[m_num_above * m_num_below];
  m_deltas = new double[m_num_above * m_num_below];
  memset(m_weights, 0, m_num_above * m_num_below * sizeof(double));
}

Connection::~Connection() {
  delete[] m_weights;
}

double Connection::get_weight(int i, int j) {
  return m_weights[i + j * m_num_above];
}

void Connection::update_weights(int i, int j, double delta) {
  m_deltas[i + j * m_num_above] += delta;
}

void Connection::reset_deltas() {
  memset(m_deltas, 0, m_num_above * m_num_below * sizeof(double));
}

void Connection::commit_deltas() {
  for(int i = 0; i < m_num_above * m_num_below; ++i) {
    m_weights[i] += m_deltas[i];
  }
}

void Connection::find_probs_upwards() {
  for(int i = 0; i < m_num_above; ++i) {
    double activation = 0.0;
    for(int j = 0; j < m_num_below; ++j) {
      if(m_below->m_state[j]) {
	activation += get_weight(i,j);
      }
    }
    m_above->m_p[i] = 1.0 / (1.0 + exp(-m_above->get_bias(i) - activation));
  }
}

void Connection::find_probs_downwards() {
  for(int i = 0; i < m_num_below; ++i) {
    double activation = 0.0;
    for(int j = 0; j < m_num_above; ++j) {
      if(m_above->m_state[j]) {
	activation += get_weight(j,i);
      }
    }
    m_below->m_p[i] = 1.0 / (1.0 + exp(-m_below->get_bias(i) - activation));
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
    
  for(int i = 0; i < m_num_above; ++i) {
    for(int j = 0; j < m_num_below; ++j) {
      if(m_below->m_state[j] && m_above->m_state[i]) {
	update_weights(i, j, epsilon);
      }
    }
  }
  
  m_below->update_biases(epsilon, true, true);
  m_above->update_biases(epsilon, true, true);
  
  find_probs_downwards();
  m_below->sample(rng);
  find_probs_upwards();
  
  for(int i = 0; i < m_num_above; ++i) {
    for(int j = 0; j < m_num_below; ++j) {
      if(m_below->m_state[j]) {
	update_weights(i, j, -epsilon * m_above->m_p[i]);
      }
    }
  }

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
