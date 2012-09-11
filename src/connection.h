#ifndef __connection_h__
#define __connection_h__

#include "layer.h"

class Connection {
public:
  Connection(Layer *below, Layer *above);
  ~Connection();
  double get_weight(int i, int j);
  void update_weights(double *delta);

private:
  int m_num_above;
  int m_num_below;
  double *m_weights;
};

#endif
