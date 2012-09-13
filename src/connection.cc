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

  // Scratch space for training
  m_observed = new bool[m_num_below];
  m_hidden = new bool[m_num_above];
  m_p_observed = new double[m_num_below];
  m_p_hidden = new double[m_num_above];
}

Connection::~Connection() {
  delete[] m_weights;
  delete[] m_observed;
  delete[] m_hidden;
  delete[] m_p_observed;
  delete[] m_p_hidden;
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

void Connection::find_probs_upwards(double *p_above, int n_above, bool *below, int n_below, Connection *connection, Layer *layer_above) {
  for(int i = 0; i < n_above; ++i) {
    double activation = 0.0;
    for(int j = 0; j < n_below; ++j) {
      if(below[j]) {
	activation += connection->get_weight(i,j);
      }
    }
    p_above[i] = 1.0 / (1.0 + exp(-layer_above->get_bias(i) - activation));
  }
}

void Connection::find_probs_downwards(double *p_below, int n_below, bool *above, int n_above, Connection *connection, Layer *layer_below) {
  for(int i = 0; i < n_below; ++i) {
    double activation = 0.0;
    for(int j = 0; j < n_above; ++j) {
      if(above[j]) {
	activation += connection->get_weight(j,i);
      }
    }
    p_below[i] = 1.0 / (1.0 + exp(-layer_below->get_bias(i) - activation));
  }
}

void Connection::sample(gsl_rng *rng, bool *target, double *p, int size) {
  for(int i = 0; i < size; ++i) {
    target[i] = (gsl_rng_uniform(rng) < p[i]);
  }
}

void Connection::perform_update_step(gsl_rng *rng, bool *input_data) {

  for(int i = 0; i < m_num_below; ++i) {
    m_observed[i] = input_data[i];
  }

  double epsilon = 0.001;

  reset_deltas();
  m_below->reset_deltas();
  m_above->reset_deltas();
  
  find_probs_upwards(m_p_hidden, m_num_above, m_observed, m_num_below, this, m_above); // Passing this here seems bad
  sample(rng, m_hidden, m_p_hidden, m_num_above);
    
  for(int i = 0; i < m_num_above; ++i) {
    for(int j = 0; j < m_num_below; ++j) {
      if(m_observed[j] && m_hidden[i]) {
	update_weights(i, j, epsilon);
      }
    }
  }
  
  for(int i = 0; i < m_num_below; ++i) {
    if(m_observed[i]) {
      m_below->update_biases(i, epsilon);
    }
  }
  
  for(int i = 0; i < m_num_above; ++i) {
    if(m_hidden[i]) {
      m_above->update_biases(i, epsilon);
    }
  }
  
  find_probs_downwards(m_p_observed, m_num_below, m_hidden, m_num_above, this, m_below);
  sample(rng, m_observed, m_p_observed, m_num_below);
  
  find_probs_upwards(m_p_hidden, m_num_above, m_observed, m_num_below, this, m_above);
  
  for(int i = 0; i < m_num_above; ++i) {
    for(int j = 0; j < m_num_below; ++j) {
      if(m_observed[j]) {
	update_weights(i, j, -epsilon * m_p_hidden[i]);
      }
    }
  }
  
  for(int i = 0; i < m_num_below; ++i) {
    if(m_observed[i]) {
      m_below->update_biases(i, -epsilon);
    }
  }
  
  for(int i = 0; i < m_num_above; ++i) {
    m_above->update_biases(i, -epsilon * m_p_hidden[i]);
  }
  
  commit_deltas();
  m_below->commit_deltas();
  m_above->commit_deltas();
}
