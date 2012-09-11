#ifndef __connection_h__
#define __connection_h__

#include "layer.h"

class Connection {
public:
  Connection(Layer *below, Layer *above);
  double get_weight(int i, int j);
  void update_weights(double *delta);
};

#endif
