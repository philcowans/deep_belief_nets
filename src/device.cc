#include "device.h"
#include "world.h"

Device::Device(World *w, Monitor *m) :
  m_network(w, m) {
}

void Device::set_schedule(Schedule *schedule) {
  m_schedule = schedule;
}

void Device::run() {
  m_schedule->reset();
  while(m_schedule->step()) {
    m_network.run_step(m_schedule);
  }
}

void Device::save_state(const char *filename) {
  m_network.dump_states(filename);
}

void Device::load_state(const char *filename) {
  m_network.load_states(filename);
}
