#include "mnist_dataset.h"
#include "monitor.h"
#include "network.h"

int main(int argc, char **argv) {
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);

  Monitor m;
  Network n(&m);

  MnistDataset dataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
  n.train(rng, &dataset);

  gsl_rng_free(rng);
}
