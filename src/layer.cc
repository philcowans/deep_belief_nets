#include "layer.h"

#include <cmath>
#include <cstring>

Layer::Layer(int size) {
  m_size = size;
  m_biases = new double[size];
  m_deltas = new double[size];
  m_state = new bool[size];
  m_p = new double[size];
  memset(m_biases, 0, size * sizeof(double));
}

Layer::~Layer() {
  delete[] m_biases;
  delete[] m_deltas;
  delete[] m_state;
  delete[] m_p;
}

int Layer::size() {
  return m_size;
}

double Layer::get_bias(int i) {
  return m_biases[i];
}

void Layer::update_biases(double epsilon, bool positive, bool stochastic) {
  for(int i = 0; i < m_size; ++i) {
    if(positive) {
      if(stochastic) {
	if(m_state[i]) {
	  m_deltas[i] += epsilon;
	}
      } 
      else {
	// YAGNI - this is never used.
      }
    }
    else {
      if(stochastic) {
	if(m_state[i]) {
	  m_deltas[i] -= epsilon;
	}
      }
      else {
	m_deltas[i] -= epsilon * m_p[i];
      }
    }
  }
}

void Layer::reset_deltas() {
  // TODO: 'double buffering'
  memset(m_deltas, 0, m_size * sizeof(double));
}

void Layer::commit_deltas() {
  for(int i = 0; i < m_size; ++i) {
    m_biases[i] += m_deltas[i];
  }
}

void Layer::sample(gsl_rng *rng) {
  for(int i = 0; i < m_size; ++i) {
    m_state[i] = (gsl_rng_uniform(rng) < m_p[i]);
  }
}

void Layer::set_state(bool *state) {
  memcpy(m_state, state, m_size * sizeof(bool));
}

void Layer::activate_from_bias() {
  for(int i = 0; i < m_size; ++i) {
    m_p[i] = 1.0 / (1.0 + exp(-m_biases[i]));
  }
}
  
  
