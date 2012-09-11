#include "network.h"

#include <cmath>
#include <cstring>

Network::Network(Monitor *monitor) {
  m_monitor = monitor;
  m_num_layers = 4;
  m_layer_sizes = new int[m_num_layers];

  for(int i = 0; i < m_num_layers; ++i) {
    m_layer_sizes[i] = 784;
  }

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

void Network::train(gsl_rng *rng, Dataset *training_data) {
  m_monitor->log_event("Starting network training");
  for(int i = 0; i < m_num_layers - 2; ++i) {
    greedily_train_layer(rng, training_data, i);
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

  double epsilon = 0.1;
  double delta_w[size_above * size_below];
  double delta_b[size_below];
  double delta_c[size_above];

  int num_iterations = 100;

  for(int k = 0; k < num_iterations; ++k) {
    memset(delta_w, 0, size_above * size_below * sizeof(double));
    memset(delta_b, 0, size_below * sizeof(double));
    memset(delta_c, 0, size_above * sizeof(double));
    
    training_data->get_sample(rng, observed);
    transform_dataset_for_layer(observed, n);
    
    find_probs_upwards(p_hidden, size_above, observed, size_below, m_connections[n], m_layers[n + 1]);
    sample(rng, hidden, p_hidden, size_above);

    // TODO: Check these are the right way round
    for(int i = 0; i < size_above; ++i) {
      for(int j = 0; j < size_below; ++j) {
	if(observed[i] && hidden[i]) {
	  delta_w[i + size_above * j] += epsilon;
	}
      }
    }

    for(int i = 0; i < size_below; ++i) {
      if(observed[i]) {
	delta_b[i] += epsilon;
      }
    }

    for(int i = 0; i < size_above; ++i) {
      if(hidden[i]) {
	delta_c[i] += epsilon;
      }
    }
    
    find_probs_downwards(p_observed, size_below, hidden, size_above, m_connections[n], m_layers[n]);
    sample(rng, observed, p_observed, size_below);
    find_probs_upwards(p_hidden, size_above, observed, size_below, m_connections[n], m_layers[n + 1]);

    for(int i = 0; i < size_above; ++i) {
      for(int j = 0; j < size_below; ++j) {
        if(observed[i]) {
          delta_w[i + size_above * j] -= epsilon * p_hidden[j];
        }
      }
    }

    for(int i = 0; i < size_below; ++i) {
      if(observed[i]) {
        delta_b[i] -= epsilon;
      }
    }

    for(int i = 0; i < size_above; ++i) {
      delta_c[i] -= epsilon * p_hidden[i];
    }
      
    m_connections[n]->update_weights(delta_w);
    m_layers[n]->update_biases(delta_b);
    m_layers[n]->update_biases(delta_c);
  }
  delete[] observed;
  delete[] hidden;
}

void Network::optimize_weights(Dataset *training_data) {
}

void Network::transform_dataset_for_layer(bool *sample, int n) {
  // no-op until we start working at deeper layers
}

void Network::sample(gsl_rng *rng, bool *target, double *p, int size) {
  for(int i = 0; i < size; ++i) {
    target[i] = (gsl_rng_uniform(rng) < p[i]);
  }
}

void Network::find_probs_upwards(double *p_above, int n_above, bool *below, int n_below, Connection *connection, Layer *layer_above) {
  for(int i = 0; i < n_above; ++i) {
    double activation = 0.0;
    for(int j = 0; j < n_below; ++j) {
      if(below[j]) {
	activation += connection->get_weight(j,i);
      }
      else {
	activation -= connection->get_weight(j,i);
      }
    }
    p_above[i] = 1.0 / (1.0 + exp(-layer_above->get_bias(i) - activation));
  }
}

void Network::find_probs_downwards(double *p_below, int n_below, bool *above, int n_above, Connection *connection, Layer *layer_below) {
  for(int i = 0; i < n_below; ++i) {
    double activation = 0.0;
    for(int j = 0; j < n_above; ++j) {
      if(above[j]) {
	activation += connection->get_weight(i,j);
      }
      else {
	activation -= connection->get_weight(i,j);
      }
    }
    p_below[i] = 1.0 / (1.0 + exp(-layer_below->get_bias(i) - activation));
  }
}
