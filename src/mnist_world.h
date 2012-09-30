#ifndef __MNIST_WORLD_H__
#define __MNIST_WORLD_H__

#include "mnist_dataset.h"
#include "world.h"

class MnistWorld : public World {
public:
  MnistWorld();

  virtual Dataset *training_data();
  virtual Dataset *test_data();

private:
  MnistDataset m_training_data;
  MnistDataset m_test_data;
};

#endif
