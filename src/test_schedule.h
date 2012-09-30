#ifndef __TEST_SCHEDULE_H__
#define __TEST_SCHEDULE_H__

#include "schedule.h"

class TestSchedule : public Schedule {
public:
  TestSchedule(int example_id);

  virtual void reset();
  virtual bool step();
  virtual int target_layer();
  virtual int active_image();
  virtual int step_type();

private:
  bool m_complete;
  int m_example_id;
};

#endif
