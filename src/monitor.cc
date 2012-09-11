#include "monitor.h"

#include <iostream>

void Monitor::log_event(const char *description) {
  std::cout << description << std::endl;
}
