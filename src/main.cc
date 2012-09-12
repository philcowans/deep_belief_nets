#include "mnist_dataset.h"
#include "monitor.h"
#include "network.h"
#include <iostream>

int main(int argc, char **argv) {
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);

  // params
  bool fixed_image = false;
  // ---

  Monitor m;
  Network n(&m);

  MnistDataset dataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", fixed_image);
  Schedule s;
  n.train(rng, &dataset, &s);

  bool sample[28*28];
  n.sample_input(rng, sample);
  for(int i = 0; i < 28; ++i ) {
    for(int j = 0; j < 28; ++j ) {
      if(sample[i*28 + j])
	std::cout << "*";
      else
	std::cout << ".";
    }
    std::cout << std::endl;
  }

  gsl_rng_free(rng);
}
