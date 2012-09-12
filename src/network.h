#ifndef __network_h__
#define __network_h__

#include "connection.h"
#include "dataset.h"
#include "layer.h"
#include "monitor.h"
#include "schedule.h"

#include "gsl/gsl_rng.h"

class Network {
public:
  Network(Monitor *monitor);
  ~Network();
  void train(gsl_rng *rng, Dataset *training_data, Schedule *schedule);
  void sample_input(gsl_rng *rng, bool *outputs);
  
private:
  Monitor *m_monitor;
  int m_num_layers;
  int *m_layer_sizes;
  Layer **m_layers;
  Connection **m_connections;

  void greedily_train_layer(gsl_rng *rng, Dataset *training_data, int n);
  void transform_dataset_for_layer(gsl_rng *rng, bool *input, bool *s, int n);
  void sample(gsl_rng *rng, bool *target, double *p, int size);
  void find_probs_upwards(double *p_above, int n_above, bool *below, int n_below, Connection *connection, Layer *layer_above);
  void find_probs_downwards(double *p_below, int n_below, bool *above, int n_above, Connection *connection, Layer *layer_below);
};

#endif
