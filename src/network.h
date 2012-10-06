#ifndef __network_h__
#define __network_h__

#include "connection.h"
#include "dataset.h"
#include "layer.h"
#include "monitor.h"
#include "schedule.h"
#include "world.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

class Monitor;

class Network {
public:
  Network(World *world, Monitor *monitor);
  ~Network();

  void run_step(Schedule *schedule);

  void sample_input(gsl_rng *rng, int label);
  gsl_vector *extract_input_states();
  void dump_states(const char *filename);
  void load_states(const char *filename);
  
  int get_label();
  
private:
  Monitor *m_monitor;
  World *m_world;
  gsl_rng *m_rng;

  int m_num_layers;
  int *m_layer_sizes;
  Layer **m_layers;
  Layer *m_input_label_layer;
  Connection **m_connections;
  bool m_mean_field;

  void greedily_train_layer(gsl_rng *rng, Dataset *training_data, int n, Schedule *schedule);
  void transform_dataset_for_layer(gsl_rng *rng, bool *input, bool *s, int n);
  void sample(gsl_rng *rng, bool *target, double *p, int size);
  void find_probs_upwards(double *p_above, int n_above, bool *below, int n_below, Connection *connection, Layer *layer_above);
  void find_probs_downwards(double *p_below, int n_below, bool *above, int n_above, Connection *connection, Layer *layer_below);

  int classify(gsl_vector *observations);
  void train(gsl_rng *rng, Dataset *training_data, Schedule *schedule);
  void fine_tune(gsl_vector *observations, int label);
};

#endif
