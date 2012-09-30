#include "test_schedule.h"

TestSchedule::TestSchedule(int example_id) {
  m_example_id = example_id;
  m_complete = false;
}

void TestSchedule::reset() {}

bool TestSchedule::step() {
  if(m_complete) {
    return false;
  }
  else {
    m_complete = true;
    return true;
  }
}

int TestSchedule::target_layer() {
  return 0; // Not really needed here - refactor out
}

int TestSchedule::active_image() {
  return m_example_id;
}

int TestSchedule::step_type() {
  return 1;
}
