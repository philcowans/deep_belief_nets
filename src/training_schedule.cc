#include "training_schedule.h"

#include <ctime>
#include <iostream>

void TrainingSchedule::reset() {
  m_step_index = -1; // Urgh
}

bool TrainingSchedule::step() {
  ++m_step_index;
  if(m_step_index % 10000 == 0) {
    std::cout << "Completed " << m_step_index << " steps (time is " << time(NULL) << ")" << std::endl;
  }
  return (m_step_index < 1000 * 300 * 3);
}

int TrainingSchedule::target_layer() {
  return m_step_index / (1000 * 300);
}

int TrainingSchedule::active_image() {
  return m_step_index % 1000; //60000;
}

int TrainingSchedule::step_type() {
  return 0;
}
