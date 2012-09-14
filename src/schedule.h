#ifndef __SCHEDULE_H__
#define __SCHEDULE_H__

class Schedule {
public:
  void reset();
  bool step();
  int target_layer();
  int active_image();

  bool m_debug;

private:
  int m_step_index;
};

#endif
