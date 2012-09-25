#include "mnist_dataset.h"
#include "monitor.h"
#include "network.h"
#include <iostream>
#include <gsl/gsl_vector.h>
#include <gd.h>

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
  // n.train(rng, &dataset, &s);
  n.load_states("final_state_0.1.tsv");


  gdImagePtr img;
  int black;
  int white;
  img = gdImageCreate(280*2, 280*2);
  black = gdImageColorAllocate(img, 0, 0, 0);  
  white = gdImageColorAllocate(img, 255, 255, 255);  

  for(int ni = 0; ni < 10; ++ni) {
    for(int nj = 0; nj < 10; ++nj) {
      std::cout << "Label: " << ni << std::endl;
      n.sample_input(rng, ni);
      gsl_vector *sample = n.extract_input_states();
      
      for(int i = 0; i < 28; ++i ) {
	for(int j = 0; j < 28; ++j ) {
	  int y = 2 * (ni * 28 + i);
	  int x = 2 * (nj * 28 + j);
	  if(gsl_vector_get(sample, i*28 + j) == 1.0)
	    gdImageRectangle(img, x, y, x+1, y+1, black);
	  else
	    gdImageRectangle(img, x, y, x+1, y+1, white);
	}
      }
    }
  }

  FILE *pngout;
  pngout = fopen("test.png", "wb");

  gdImagePng(img, pngout);
  fclose(pngout);
  gdImageDestroy(img);

  //  n.dump_states("final_state.tsv");

  gsl_rng_free(rng);
}
