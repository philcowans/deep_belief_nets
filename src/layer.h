#ifndef __layer_h__
#define __layer_h__

#include <gsl/gsl_rng.h>

class Layer {
public:
  Layer(int size);
  ~Layer();
  int size();
  double get_bias(int i);
  
  void reset_deltas();
  void update_biases(double epsilon, bool positive, bool stochastic);
  void commit_deltas();
  void sample(gsl_rng *rng);
  void set_state(bool *state);
  void activate_from_bias();

  // Privatise these at some point
  bool *m_state;
  double *m_p;

private:
  int m_size;
  double *m_biases;
  double *m_deltas;
};

#endif
