#include "monitor.h"

#include <iostream>

void Monitor::log_event(const char *description) {
  std::cout << description << std::endl;
}

int Monitor::read_int(const char *key) {
  return 0;
}
