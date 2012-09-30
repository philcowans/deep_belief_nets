#ifndef __TRAINING_SCHEDULE_H__
#define __TRAINING_SCHEDULE_H__

#include "schedule.h"

class TrainingSchedule : public Schedule {
public:
  virtual void reset();
  virtual bool step();
  virtual int target_layer();
  virtual int active_image();
  virtual int step_type();

private:
  int m_step_index;
};

#endif
