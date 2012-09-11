#ifndef __layer_h__
#define __layer_h__

class Layer {
public:
  Layer(int size);
  ~Layer();
  int size();
  double get_bias(int i);
  void update_biases(double *delta);

private:
  int m_size;
  double *m_biases;
};

#endif
