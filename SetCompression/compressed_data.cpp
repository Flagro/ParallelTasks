#include "compressed_data.hpp"
#include "sparse_data.hpp"
#include "dense_data.hpp"
#include <vector>
#include <omp.h>

template <typename T>
CompressedData<T>::CompressedData(const std::vector<T>& input, T max_value) {

    // Create the dense array representation using std::vector
    std::vector<T> dense_array(max_value + 1, 0);  // Initialize with zeros

    // Parallelizing the population of the dense array
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); i++) {
        #pragma omp atomic
        dense_array[input[i]]++;
    }

    // Count non-zero elements in the dense array
    size_t non_zero_count = 0;
    #pragma omp parallel for reduction(+:non_zero_count)
    for (T i = 0; i <= max_value; i++) {
        if (dense_array[i] > 0) {
            ++non_zero_count;
        }
    }

    // If the number of non-zero values is less than or equal to half of max_value, use the sparse representation.
    if (non_zero_count * 2 > static_cast<size_t>(max_value)) {
        data_format_ = std::make_unique<DenseData<T>>(std::move(dense_array), input.size());
        return;
    }

    std::vector<std::pair<T, T>> sparse_array;
    sparse_array.reserve(non_zero_count);
    for (T i = 0; i <= max_value; i++) {
        if (dense_array[i] > 0) {
            sparse_array.emplace_back(i, dense_array[i]);
        }
    }
    data_format_ = std::make_unique<SparseData<T>>(std::move(sparse_array), input.size());
}

template <typename T>
std::vector<T> CompressedData<T>::get_data() const {
    return data_format_->get_data();
}

template <typename T>
size_t CompressedData<T>::get_size() const {
    return data_format_->get_size();
}
