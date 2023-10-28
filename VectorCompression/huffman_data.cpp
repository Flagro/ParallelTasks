#include "huffman_data.hpp"
#include <queue>
#include <omp.h>

template <typename T>
HuffmanData<T>::HuffmanData(const std::vector<T>& input_array, size_t max_value) {
    data_array_.resize(max_value + 1, -1);
    freq_array_.resize(max_value + 1, 0);
    codes_.resize(max_value + 1);

    // Compute frequencies
    #pragma omp parallel for
    for (int i = 0; i < input_array.size(); i++) {
        #pragma omp atomic
        freq_array_[input_array[i]]++;
    }

    buildHuffmanTree(max_value);
    generateCodes(0, std::vector<bool>());
    compressed_data_ = compress(input_array);
}

template <typename T>
int HuffmanData<T>::buildHuffmanTree(size_t max_value) {
    std::priority_queue<int, std::vector<int>, Compare> min_heap{Compare(freq_array_)};

    for (int i = 0; i <= max_value; i++) {
        if (freq_array_[i] > 0) {
            data_array_.push_back(i);
            min_heap.push(data_array_.size() - 1);
        }
    }

    while (min_heap.size() > 1) {
        int left = min_heap.top();
        min_heap.pop();

        int right = min_heap.top();
        min_heap.pop();

        data_array_.push_back(-1);
        freq_array_.push_back(freq_array_[left] + freq_array_[right]);
        min_heap.push(data_array_.size() - 1);
    }

    return min_heap.top();
}

template <typename T>
void HuffmanData<T>::generateCodes(int index, const std::vector<bool>& currentCode) {
    if (index >= data_array_.size()) {
        return;
    }

    if (data_array_[index] != -1) {
        codes_[data_array_[index]] = currentCode;
    }

    std::vector<bool> leftCode = currentCode;
    leftCode.push_back(false);  // 0
    generateCodes(2*index + 1, leftCode);

    std::vector<bool> rightCode = currentCode;
    rightCode.push_back(true);  // 1
    generateCodes(2*index + 2, rightCode);
}

template <typename T>
std::vector<bool> HuffmanData<T>::compress(const std::vector<T>& input) {
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<bool>> temp_results(num_threads);

    #pragma omp parallel for
    for (int t = 0; t < num_threads; t++) {
        int chunk_size = input.size() / num_threads;
        int start_idx = t * chunk_size;
        int end_idx = (t == num_threads - 1) ? input.size() : start_idx + chunk_size;

        for (int i = start_idx; i < end_idx; i++) {
            for (bool bit : codes_[input[i]]) {
                temp_results[t].push_back(bit);
            }
        }
    }

    std::vector<bool> result;
    for (const auto& vec : temp_results) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}

template <typename T>
std::vector<T> HuffmanData<T>::decompress(const std::vector<bool>& compressed) const {
    std::vector<T> result;
    int current_index = 0;  // root of Huffman tree

    for (bool bit : compressed) {
        if (bit) {
            current_index = 2*current_index + 2;
        } else {
            current_index = 2*current_index + 1;
        }

        if (current_index >= data_array_.size()) {
            break;
        }

        if (data_array_[current_index] != -1) {
            result.push_back(data_array_[current_index]);
            current_index = 0;  // return to root
        }
    }
    return result;
}

template <typename T>
std::vector<T> HuffmanData<T>::get_data() const {
    return decompress(compressed_data_);
}

template <typename T>
size_t HuffmanData<T>::get_size() const {
    return compressed_data_.size() + 
           data_array_.capacity() * sizeof(T) + 
           freq_array_.capacity() * sizeof(unsigned);
}
