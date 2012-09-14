#include "network.h"

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

void Network::sample_input(gsl_rng *rng) {
  m_layers[m_num_layers - 1]->activate_from_bias();
  m_connections[m_num_layers - 2]->sample_layer(rng, 1000);
  for(int i = m_num_layers - 3; i >= 0; --i) {
    m_connections[i]->propagate_hidden(rng);
  }

  // Something something display output
}

void Network::greedily_train_layer(gsl_rng *rng, Dataset *training_data, int n, Schedule *schedule) {
  int input_size = m_layers[0]->size();
  bool *input_observations = new bool[input_size]; // TODO: Should be okay for dataset to own this rather than copying
  training_data->get_sample(rng, input_observations, schedule->active_image());
  m_layers[0]->set_state(input_observations);
  for(int i = 0; i < n; ++i) {
    m_connections[i]->propagate_observation(rng);
  }
  m_connections[n]->perform_update_step(rng);
  delete[] input_observations;
}
