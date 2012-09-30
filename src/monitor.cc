#include "monitor.h"

#include <iostream>

void Monitor::set_network(Network *network) {
  m_network = network;
}

void Monitor::log_event(const char *description) {
  std::cout << description << std::endl;
}

int Monitor::read_int(const char *key) {
  return m_network->get_label();
}
