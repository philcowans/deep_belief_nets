#ifndef __MONITOR_H__
#define __MONITOR_H__

class Monitor {
public:
  void log_event(const char *message);
  int read_int(const char *key);
};

#endif
