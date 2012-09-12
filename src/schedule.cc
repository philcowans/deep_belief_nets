#include "schedule.h"

void Schedule::reset() {
  m_step_index = -1; // Urgh
}

bool Schedule::step() {
  ++m_step_index;
  return (m_step_index < 3000);
}

int Schedule::target_layer() {
  return m_step_index / 1000;
}
