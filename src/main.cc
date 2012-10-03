#include "device.h"
#include "mnist_dataset.h"
#include "mnist_world.h"
#include "monitor.h"
#include "network.h"
#include "training_schedule.h"
#include "test_schedule.h"

#include <iostream>
#include <gsl/gsl_vector.h>
//#include <gd.h>

int main(int argc, char **argv) {
  Monitor m;
  MnistWorld w;
  Device device(&w, &m);
  
  // Schedule *training_schedule = new TrainingSchedule();
  // device.set_schedule(training_schedule);
  // device.run();
  // delete training_schedule;

  device.load_state("final_state_0.1.tsv");

  Schedule *test_schedule;
  int count_correct = 0;
  for(int i = 0; i < 10000; ++i) {
    test_schedule = new TestSchedule(i);
    device.set_schedule(test_schedule);
    device.run();
    int label = m.read_int("label");
    if(label == w.training_data()->get_label(test_schedule->active_image()))
      ++count_correct;
    delete test_schedule;
  }
  std::cout << count_correct << std::endl;

  // // params
  // bool fixed_image = false;
  // // ---

  // Monitor m;
  // Network n(&m);

  // MnistDataset dataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", fixed_image);
  // Schedule s;
  // s.m_debug = false;
  // // n.train(rng, &dataset, &s);
  // n.load_states("final_state_0.1.tsv");

  // MnistDataset test_dataset("", "", false);
  // gsl_vector *input_observations = gsl_vector_alloc(784); // TODO: Should be okay for dataset to own this rather than copying
  // for(int i = 0; i < 10000; ++i) {
  //   test_dataset.get_state(input_observations, i);
  //   int cl = n.classify(input_observations);
  //   std::cout << i << "\t" << test_dataset.get_label(i) << "\t" << cl << std::endl;
  // }
  // gsl_vector_free(input_observations);

  // // gdImagePtr img;
  // // int black;
  // // int white;
  // // img = gdImageCreate(280*2, 280*2);
  // // black = gdImageColorAllocate(img, 0, 0, 0);  
  // // white = gdImageColorAllocate(img, 255, 255, 255);  

  // // for(int ni = 0; ni < 10; ++ni) {
  // //   for(int nj = 0; nj < 10; ++nj) {
  // //     std::cout << "Label: " << ni << std::endl;
  // //     n.sample_input(rng, ni);
  // //     gsl_vector *sample = n.extract_input_states();
      
  // //     for(int i = 0; i < 28; ++i ) {
  // // 	for(int j = 0; j < 28; ++j ) {
  // // 	  int y = 2 * (ni * 28 + i);
  // // 	  int x = 2 * (nj * 28 + j);
  // // 	  if(gsl_vector_get(sample, i*28 + j) == 1.0)
  // // 	    gdImageRectangle(img, x, y, x+1, y+1, black);
  // // 	  else
  // // 	    gdImageRectangle(img, x, y, x+1, y+1, white);
  // // 	}
  // //     }
  // //   }
  // // }

  // // FILE *pngout;
  // // pngout = fopen("test.png", "wb");

  // // gdImagePng(img, pngout);
  // // fclose(pngout);
  // // gdImageDestroy(img);

  // //  n.dump_states("final_state.tsv");

  // gsl_rng_free(rng);
}
