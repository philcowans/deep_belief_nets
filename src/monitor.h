#ifndef __MONITOR_H__
#define __MONITOR_H__

#include "network.h"

class Network;

class Monitor {
public:
  void set_network(Network *network);

  void log_event(const char *message);
  int read_int(const char *key);

private:
  Network *m_network;
};

#endif
