#include "mnist_dataset.h"

#include <fstream>

MnistDataset::MnistDataset(const char *images_filename, const char *labels_filename) {
  load_images(images_filename);
  load_labels(labels_filename);
}

MnistDataset::~MnistDataset() {
  for(int i = 0; i < m_num_images; ++i) {
    delete[] m_image_data[i];
  }
  delete[] m_image_data;
  delete[] m_labels;
}

void MnistDataset::load_images(const char *filename) {
  int32_t magic;

  std::ifstream data_file(filename);

  data_file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  data_file.read(reinterpret_cast<char *>(&m_num_images), sizeof(m_num_images));
  data_file.read(reinterpret_cast<char *>(&m_num_rows), sizeof(m_num_rows));
  data_file.read(reinterpret_cast<char *>(&m_num_cols), sizeof(m_num_cols));

  magic = __builtin_bswap32(magic);
  m_num_images = __builtin_bswap32(m_num_images);
  m_num_rows = __builtin_bswap32(m_num_rows);
  m_num_cols = __builtin_bswap32(m_num_cols);
  
  m_image_data = new uint8_t*[m_num_images];
  for(int i = 0; i < m_num_images; ++i) {
    m_image_data[i] = new uint8_t[m_num_rows * m_num_cols];
    for(int j = 0; j < m_num_rows * m_num_cols; ++j) {
      data_file.read(reinterpret_cast<char *>(m_image_data[i] + j), 1);
    }
  }

  data_file.close();
}

void MnistDataset::load_labels(const char *filename) {
  int32_t magic;

  std::ifstream data_file(filename);

  data_file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  data_file.read(reinterpret_cast<char *>(&m_num_images), sizeof(m_num_images));
  
  magic = __builtin_bswap32(magic);
  m_num_images = __builtin_bswap32(m_num_images);

  m_labels = new uint8_t[m_num_images];
  for(int i = 0; i < m_num_images; ++i) {
    data_file.read(reinterpret_cast<char *>(m_labels + i), 1);
  }

  data_file.close();
}

bool MnistDataset::get_value(int i) {
  return false;
}

void MnistDataset::get_sample(gsl_rng *r, bool *sample) {
  int active_image = gsl_rng_uniform_int(r, m_num_images);
  for(int i = 0; i< m_num_rows * m_num_cols; ++i) {
    sample[i] = gsl_rng_uniform_int(r, 255) < m_image_data[active_image][i];
  }
}
