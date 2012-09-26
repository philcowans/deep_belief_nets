#include "network.h"

#include <cstring>
#include <sstream>

#include <iostream>
#include <fstream>

Network::Network(Monitor *monitor) {
  m_mean_field = true;
  m_monitor = monitor;
  m_num_layers = 4;
  m_layer_sizes = new int[m_num_layers];

  m_layer_sizes[0] = 784;
  m_layer_sizes[1] = 500;
  m_layer_sizes[2] = 510; // Special case
  m_layer_sizes[3] = 2000;

  m_layers = new Layer*[m_num_layers];
  m_connections = new Connection*[m_num_layers - 1];

  for(int i = 0; i < m_num_layers; ++i) {
    bool labels = (i == 2);
    m_layers[i] = new Layer(m_layer_sizes[i], labels);
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
    //    std::cout << "layer " << schedule->target_layer() << std::endl;
    greedily_train_layer(rng, training_data, schedule->target_layer(), schedule);
  }
}

void Network::sample_input(gsl_rng *rng, int label) {
  m_layers[m_num_layers - 1]->activate_from_bias();
  m_connections[m_num_layers - 2]->sample_layer(rng, 1000, label);
  for(int i = m_num_layers - 3; i >= 0; --i) {
    m_connections[i]->propagate_hidden(rng);
  }

  // Something something display output
}

int Network::classify(gsl_vector *observations) {
  m_layers[0]->set_state(observations);
  for(int i = 0; i < 2; ++i) {
    m_connections[i]->propagate_observation(NULL, m_mean_field);
  }
  return m_connections[m_num_layers - 2]->find_label();
}

gsl_vector *Network::extract_input_states() {
  return m_layers[0]->state(true);
}

void Network::dump_states(const char *filename) {
  std::ofstream f(filename);
  for(int i = 0; i < m_num_layers; ++i) {
    for(int j = 0; j < m_layer_sizes[i]; ++j) {
      f << "l\t" << i << "\t" << j << "\t" << m_layers[i]->get_bias(j) << std::endl;
    }
  }
  for(int i = 0; i < m_num_layers - 1; ++i) {
    for(int j = 0; j < m_layer_sizes[i+1]; ++j) {
      for(int k = 0; k < m_layer_sizes[i]; ++k) {
	if((i != 1) || (j < 500)) {
	  f << "c\t" << i << "\t" << j << "\t" << k << "\t" << m_connections[i]->get_weight(j, k) << std::endl;
	}
      }
    }
  }
  f.close();
}

void Network::load_states(const char *filename) {
  std::ifstream f(filename);
  for(int i = 0; i < m_num_layers; ++i) {
    for(int j = 0; j < m_layer_sizes[i]; ++j) {
      std::string t;
      int n_i;
      int n_j;
      double bias;
      f >> t >> n_i >> n_j >> bias;
      //      std::cout << t << " " << n_i << " " << n_j << " " << bias << std::endl;
      m_layers[i]->set_bias(j, bias);
    }
  }
  for(int i = 0; i < m_num_layers - 1; ++i) {
    for(int j = 0; j < m_layer_sizes[i+1]; ++j) {
      for(int k = 0; k < m_layer_sizes[i]; ++k) {
        if((i != 1) || (j < 500)) {
	  std::string t;
	  int n_i;
	  int n_j;
	  int n_k;
	  double weight;
	  f >> t >> n_i >> n_j >> n_k >> weight;
	  //  std::cout << t << " " << n_i << " " << n_j << " " << n_k << " " << weight << std::endl;
	  m_connections[i]->set_weight(j, k, weight);
	}
      }
    }
  }
  f.close();
}

void Network::greedily_train_layer(gsl_rng *rng, Dataset *training_data, int n, Schedule *schedule) {
  int input_size = m_layers[0]->size(false);
  gsl_vector *input_observations = gsl_vector_alloc(input_size); // TODO: Should be okay for dataset to own this rather than copying
  if(m_mean_field) {
    training_data->get_state(input_observations, schedule->active_image());
  } 
  else {
    training_data->get_sample(rng, input_observations, schedule->active_image());
  }
  m_layers[0]->set_state(input_observations);
  for(int i = 0; i < n; ++i) {
    m_connections[i]->propagate_observation(rng, m_mean_field);
  }
  if(n == 2) {
    m_layers[2]->set_label(training_data->get_label(schedule->active_image()));
  }
  m_connections[n]->perform_update_step(rng);
  gsl_vector_free(input_observations);
}
