#ifndef __layer_h__
#define __layer_h__

#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

class Layer {
public:
  Layer(int size, bool labels);
  ~Layer();
  int size(bool ext);
  double get_bias(int i);
  void set_bias(int i, double v);
  
  void reset_deltas();
  void commit_deltas();
  void sample(gsl_rng *rng, bool ext = true);
  void transfer();
  void set_state(const gsl_vector *state);
  void activate_from_bias();
  void set_label(int label);
  int most_probable_label();

  // Privatise these at some point
  gsl_vector *state(bool ext);
  gsl_vector *p(bool ext);
  gsl_vector *activation(bool ext);
  gsl_vector *biases(bool ext);
  gsl_vector *deltas(bool ext);

  bool m_labels;


private:
  int m_size;
  gsl_vector *m_state;
  gsl_vector *m_p;
  gsl_vector *m_activation;
  gsl_vector *m_biases;
  gsl_vector *m_deltas;
  gsl_vector *m_biases_down; // Not 100% sure this is needed
  gsl_vector_view m_state_view;
  gsl_vector_view m_p_view;
  gsl_vector_view m_activation_view;
  gsl_vector_view m_biases_view;
  gsl_vector_view m_deltas_view;
};

#endif
