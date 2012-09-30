#include "mnist_dataset.h"
#include "mnist_world.h"

MnistWorld::MnistWorld() :
  m_training_data("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte"), 
  m_test_data("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte") {
}

Dataset *MnistWorld::training_data() {
  return &m_training_data;
}

Dataset *MnistWorld::test_data() {
  return &m_test_data;
}
