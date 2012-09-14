#ifndef __connection_h__
#define __connection_h__

#include "layer.h"

#include <gsl/gsl_rng.h>

class Connection {
public:
  Connection(Layer *below, Layer *above);
  ~Connection();
  inline double get_weight(int i, int j);
  inline void update_weights(int i, int j, double delta);
  void reset_deltas();
  void commit_deltas();
  void perform_update_step(gsl_rng *rng);
  void propagate_observation(gsl_rng *rng);
  void propagate_hidden(gsl_rng *rng);
  void sample_layer(gsl_rng *rng, int num_iterations);

private:
  int m_num_above;
  int m_num_below;
  double *m_weights;
  double *m_deltas;
  Layer *m_above;
  Layer *m_below;

  void find_probs_upwards();
  void find_probs_downwards();
};

#endif
