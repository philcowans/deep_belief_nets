#ifndef __layer_h__
#define __layer_h__

class Layer {
public:
  Layer(int size);
  ~Layer();
  int size();
  double get_bias(int i);
  
  void reset_deltas();
  void update_biases(int i, double delta);
  void commit_deltas();

private:
  int m_size;
  double *m_biases;
  double *m_deltas;
};

#endif
