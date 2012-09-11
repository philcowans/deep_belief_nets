#include "connection.h"

Connection::Connection(Layer *below, Layer *above) {
}

double Connection::get_weight(int i, int j) {
  return 0.0;
}

void Connection::update_weights(double *delta) {
}
