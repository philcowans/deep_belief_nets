#ifndef __WORLD_H__
#define __WORLD_H__

#include "dataset.h"

class World {
public:
  virtual Dataset *training_data() = 0;
  virtual Dataset *test_data() = 0;
};

#endif
