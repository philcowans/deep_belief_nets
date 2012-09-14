#include "schedule.h"

#include <ctime>
#include <iostream>

void Schedule::reset() {
  m_step_index = -1; // Urgh
}

bool Schedule::step() {
  ++m_step_index;
  if(m_step_index % 10000 == 0) {
    std::cout << "Completed " << m_step_index << " steps (time is " << time(NULL) << ")" << std::endl;
  }
  if(m_debug) {
    return (m_step_index < 1000 * 3);
  }
  else {
    return (m_step_index < 60000 * 30 * 3);
  }
}

int Schedule::target_layer() {
  if(m_debug) {
    return m_step_index / 1000;
  }
  else {
    return m_step_index / 60000 * 30;
  }
}

int Schedule::active_image() {
  if(m_debug) {
    return 0;
  }
  else {
    return m_step_index % 60000;
  }
}
