#include "compressed_data.hpp"
#include "huffman_data.hpp"
#include "bitset_data.hpp"
#include <vector>
#include <cmath>
#include <omp.h>

template <typename T>
CompressedData<T>::CompressedData(const std::vector<T>& input, T max_value) {
    std::vector<T> frequency(max_value + 1, 0);  // Initialize with zeros

    #pragma omp parallel
    {
        // Each thread will have its own local dense array
        std::vector<T> local_dense_array(max_value + 1, 0);

        #pragma omp for
        for (size_t i = 0; i < input.size(); i++) {
            local_dense_array[input[i]]++;
        }

        // Now, we'll reduce/merge the local dense arrays into the global one
        #pragma omp critical
        {
            for (T i = 0; i <= max_value; i++) {
                frequency[i] += local_dense_array[i];
            }
        }
    }

    double entropy = 0.0;
    double data_size = static_cast<double>(data.size());

    #pragma omp parallel for reduction(-:entropy)
    for (size_t i = 0; i < frequency.size(); i++) {
        double prob = static_cast<double>(frequency[i]) / data_size;
        entropy -= prob * std::log2(prob);
    }

    double max_entropy = std::log2(max_value + 1);

    if (input.size() < 1000 || entropy / max_entropy > 0.95) {
        data_format_ = std::make_unique<BitsetData<T>>(input, max_value);
    } else {
        data_format_ = std::make_unique<HuffmanData<T>>(input, max_value);
    }
}

template <typename T>
std::vector<T> CompressedData<T>::get_data() const {
    return data_format_->get_data();
}

template <typename T>
size_t CompressedData<T>::get_size() const {
    return data_format_->get_size();
}
