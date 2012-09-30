#ifndef __SCHEDULE_H__
#define __SCHEDULE_H__

class Schedule {
public:
  virtual void reset() = 0;
  virtual bool step() = 0;
  virtual int target_layer() = 0;
  virtual int active_image() = 0;
  virtual int step_type() = 0;
};

#endif
