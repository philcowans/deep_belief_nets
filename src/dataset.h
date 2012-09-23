#ifndef __dataset_h__
#define __dataset_h__

#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

class Dataset {
public:
  virtual bool get_value(int i) = 0;
  virtual void get_sample(gsl_rng *r, gsl_vector *sample, int example_id) = 0;
  virtual void get_state(gsl_vector *sample, int example_id) = 0;
  virtual int get_label(int example_id) = 0;
};

#endif
