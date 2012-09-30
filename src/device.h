#ifndef __DEVICE_H__
#define __DEVICE_H__

#include "monitor.h"
#include "network.h"
#include "schedule.h"
#include "world.h"

class Device {
public:
  Device(World *w, Monitor *m);

  void set_schedule(Schedule *schedule);
  void run();

  void save_state(const char* filename);
  void load_state(const char* filename);

private:
  Network m_network;
  Schedule *m_schedule;
};

#endif
