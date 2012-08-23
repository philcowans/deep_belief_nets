#include "network.h"

Network::Network() {
  m_num_layers = 4;
  m_layer_sizes = new int[m_num_layers];
  m_layers = new Layer*[m_num_layers];
  m_connections = new Connection*[m_num_layers - 1];

  for(int i = 0; i < m_num_layers; ++i) {
    m_layers[i] = new Layer(m_layer_sizes[i]);
  }

  for(int i = 0; i < m_num_layers - 1; ++i) {
    m_connections[i] = new Connection(m_layers[i], m_layers[i + 1]);
  }
}

Network::~Network() {
  delete[] m_layer_sizes;

  for(int i = 0; i < m_num_layers; ++i) {
    delete m_layers[i];
  }
  delete[] m_layers;
    
  for(int i = 0; i < m_num_layers - 1; ++i) {
    delete m_connections[i];
  }
  delete[] m_connections;
}

void Network::train(Dataset *training_data) {
  for(int i = 0; i < m_num_layers - 2; ++i) {
    greedily_train_layer(training_data, i);
  }
  optimize_weights(training_data);
}

void Network::greedily_train_layer(gsl_rng *rng, Dataset *training_data, int n) {
  int size_below = m_layers[n]->size();
  int size_above = m_layers[n + 1]->size();
  bool *observed = new bool[size_below];
  bool *hidden = new bool[size_above];
  double *p_observed = new double[size_below];
  double *p_hidden = new double[size_above];

  // Rough outline
  // 1. Clamp visible units at observed values

  transform_dataset_for_layer(training_data, n, observed);

  // 2. Compute expectation over visible / hidden product

  for(int k = 0; k < 1; ++k) {
    find_probs_upwards(p_hidden, size_above, observed, size_below, m_connections[n]);
    
    // 3. Run Gibbs sampling for the appropriate number of steps
    
    sample(rng, hidden, p_hidden, size_above);
    find_probs_downwards(p_observed, size_below, hidden, size_above, m_connections[n]);
    sample(rng, observed, p_observed, size_below);
  }

  // 4. Update weights

  delete[] observed;
  delete[] hidden;
}

void Network::optimize_weights(Dataset *training_data) {
}

void Network::transform_dataset_for_layer(Dataset *training_data, int n, bool *observed) {
  int size = m_layers[n]->size();
  for(int i = 0; i < size; ++i) {
    observed[i] = training_data->get_value(i);
  }
}

void Network::sample(gsl_rng *rng, bool *target, double *p, int size) {
  for(int i = 0; i < size; ++i) {
    target[i] = (gsl_rng_uniform(rng) < p[i]);
  }
}

void Network::find_probs_upwards(double *p_above, int n_above, bool *below, int n_below, Connection *connection) {
  
}

void Network::find_probs_downwards(double *p_below, int n_below, bool *above, int n_above, Connection *connection) {
  for(int i = 0; i < n_below, ++i) {
    double activation = 0.0;
    for(int j = 0; j < n_above; ++j) {
      if(above[j]) {
	activation += connection->get_weight(i,j);
      }
      else {
	activation -= connection->get_weight(i,j);
      }
    }
    p_below[i] = 1.0 / (1.0 + exp(-connection->get_bias(i) - activation));
  }
}
