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

void Network::greedily_train_layer(Dataset *training_data, int n) {
}

void Network::optimize_weights(Dataset *training_data) {
}
