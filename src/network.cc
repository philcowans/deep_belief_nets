#include "network.h"

#include <cmath>
#include <cstring>
#include <sstream>

#include <iostream>

Network::Network(Monitor *monitor) {
  m_monitor = monitor;
  m_num_layers = 4;
  m_layer_sizes = new int[m_num_layers];

  m_layer_sizes[0] = 784;
  m_layer_sizes[1] = 500;
  m_layer_sizes[2] = 500;
  m_layer_sizes[3] = 2000;

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

void Network::train(gsl_rng *rng, Dataset *training_data, Schedule *schedule) {
  m_monitor->log_event("Starting network training");
  schedule->reset();
  while(schedule->step()) {
    greedily_train_layer(rng, training_data, schedule->target_layer(), schedule);
  }
}

void Network::sample_input(gsl_rng *rng, bool *outputs) {
  int size_below = m_layers[m_num_layers - 2]->size();
  int size_above = m_layers[m_num_layers - 1]->size();
  double *p_above = new double[size_above];
  double *p = new double[size_below];
  bool *s = new bool[size_above];
  bool *o = new bool[size_below];
  for(int i = 0; i < size_above; ++i) {
    p_above[i] = 1.0 / (1.0 + exp(-m_layers[m_num_layers - 1]->get_bias(i)));
  }
  sample(rng, s, p, size_above);
  int num_top_iterations = 1000;
  for(int i = 0; i < num_top_iterations; ++i) {
    find_probs_downwards(p, size_below, s, size_above, m_connections[m_num_layers-2], m_layers[m_num_layers-2]);
    sample(rng, o, p, size_below);
    find_probs_upwards(p_above, size_above, o, size_below, m_connections[m_num_layers-2], m_layers[m_num_layers-1]);
    sample(rng, s, p_above, size_above);    
  }
  delete[] p_above;
  delete[] s;

  for(int i = m_num_layers - 3; i >= 0; --i) {
    size_above = size_below;
    size_below = m_layers[i]->size();
    std::cout << size_below << std::endl;
    delete[] p;
    p = new double[size_below];
    find_probs_downwards(p, size_below, o, size_above, m_connections[i], m_layers[i]);    
    for(int i = 0 ; i < size_below / 100; ++i) {
      std::cout << "Downwards prob " << i * 100 << " = " << p[i*100] << std::endl;
    }
    delete[] o;
    o = new bool[size_below];
    sample(rng, o, p, size_below);
  }
  for(int i = 0; i < size_below; ++i) {
    outputs[i] = o[i];
  }
  delete[] p;
  delete[] o;
}

void Network::greedily_train_layer(gsl_rng *rng, Dataset *training_data, int n, Schedule *schedule) {
  int input_size = m_layers[0]->size();
  bool *input_observations = new bool[input_size];

  int size_below = m_layers[n]->size();
  int size_above = m_layers[n + 1]->size();

  bool *observed = new bool[size_below];
  bool *hidden = new bool[size_above];
  double *p_observed = new double[size_below];
  double *p_hidden = new double[size_above];

  double epsilon = 0.001;
  double delta_w[size_above * size_below];
  double delta_b[size_below];
  double delta_c[size_above];

  memset(delta_w, 0, size_above * size_below * sizeof(double));
  memset(delta_b, 0, size_below * sizeof(double));
  memset(delta_c, 0, size_above * sizeof(double));
  
  training_data->get_sample(rng, input_observations, schedule->active_image());
  transform_dataset_for_layer(rng, input_observations, observed, n);
  
  find_probs_upwards(p_hidden, size_above, observed, size_below, m_connections[n], m_layers[n + 1]);
  sample(rng, hidden, p_hidden, size_above);
    
    // TODO: Check these are the right way round
    for(int i = 0; i < size_above; ++i) {
      for(int j = 0; j < size_below; ++j) {
	if(observed[j] && hidden[i]) {
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
        if(observed[j]) {
          delta_w[i + size_above * j] -= epsilon * p_hidden[i];
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
    m_layers[n + 1]->update_biases(delta_c);


  delete[] input_observations;
  delete[] observed;
  delete[] hidden;
}

void Network::transform_dataset_for_layer(gsl_rng *rng, bool *input, bool *s, int n) {
  int size_below = m_layers[0]->size();
  bool *d = new bool[size_below];
  for(int i = 0; i < size_below; ++i) {
    d[i] = input[i];
  }

  for(int i = 0; i < n; ++i) {
    int size_below = m_layers[i]->size();
    int size_above = m_layers[i + 1]->size();
    double *p_hidden = new double[size_above];

    find_probs_upwards(p_hidden, size_above, d, size_below, m_connections[i], m_layers[i + 1]);

    delete[] d;
    d = new bool[size_above];

    sample(rng, d, p_hidden, size_above);

    delete[] p_hidden;
  }
  
  for(int i = 0; i < m_layers[n]->size(); ++i) {
    s[i] = d[i];
  }

  delete[] d;
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
	activation += connection->get_weight(i,j);
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
	activation += connection->get_weight(j,i);
      }
    }
    p_below[i] = 1.0 / (1.0 + exp(-layer_below->get_bias(i) - activation));
  }
}
