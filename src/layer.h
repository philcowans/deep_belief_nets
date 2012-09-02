#ifndef __layer_h__
#define __layer_h__

class Layer {
public:
  Layer(int size);
  int size();
  double get_bias(int i);

private:
  int m_size;
};

#endif
