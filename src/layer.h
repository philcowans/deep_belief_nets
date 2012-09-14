#ifndef __layer_h__
#define __layer_h__

#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

class Layer {
public:
  Layer(int size);
  ~Layer();
  int size();
  double get_bias(int i);
  
  void reset_deltas();
  void commit_deltas();
  void sample(gsl_rng *rng);
  void set_state(const gsl_vector *state);
  void activate_from_bias();

  // Privatise these at some point
  gsl_vector *m_state;
  gsl_vector *m_p;
  gsl_vector *m_biases;
  gsl_vector *m_deltas;

private:
  int m_size;
};

#endif
