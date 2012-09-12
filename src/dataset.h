#ifndef __dataset_h__
#define __dataset_h__

#include <gsl/gsl_rng.h>

class Dataset {
public:
  virtual bool get_value(int i) = 0;
  virtual void get_sample(gsl_rng *r, bool *sample, int example_id) = 0;
};

#endif
