#include "mnist_dataset.h"
#include "monitor.h"
#include "network.h"
#include <iostream>
#include <gsl/gsl_vector.h>

int main(int argc, char **argv) {
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);

  // params
  bool fixed_image = false;
  // ---

  Monitor m;
  Network n(&m);

  MnistDataset dataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", fixed_image);
  Schedule s;
  s.m_debug = false;
  n.train(rng, &dataset, &s);

  for(int i = 0; i < 10; ++i) {
    std::cout << "Label: " << dataset.get_label(i) << std::endl;
    n.sample_input(rng, dataset.get_label(i));
    gsl_vector *sample = n.extract_input_states();
    
    for(int i = 0; i < 28; ++i ) {
      for(int j = 0; j < 28; ++j ) {
	if(gsl_vector_get(sample, i*28 + j) == 1.0)
	  std::cout << "*";
	else
	  std::cout << ".";
      }
      std::cout << std::endl;
    }
  }

  n.dump_states("final_state.tsv");

  gsl_rng_free(rng);
}
