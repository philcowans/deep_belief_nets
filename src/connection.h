#ifndef __connection_h__
#define __connection_h__

#include "layer.h"

#include <gsl/gsl_rng.h>

class Connection {
public:
  Connection(Layer *below, Layer *above);
  ~Connection();
  double get_weight(int i, int j);
  void update_weights(int i, int j, double delta);
  void reset_deltas();
  void commit_deltas();
  void perform_update_step(gsl_rng *rng, bool *input_data);

private:
  int m_num_above;
  int m_num_below;
  double *m_weights;
  double *m_deltas;
  Layer *m_above;
  Layer *m_below;

  void find_probs_upwards(double *p_above, int n_above, bool *below, int n_below, Connection *connection, Layer *layer_above);
  void find_probs_downwards(double *p_below, int n_below, bool *above, int n_above, Connection *connection, Layer *layer_below);
  void sample(gsl_rng *rng, bool *target, double *p, int size);

};

#endif
