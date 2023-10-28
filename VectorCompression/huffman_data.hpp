#pragma once

#include "data_format.hpp"
#include <vector>
#include <functional>

template <typename T>
class HuffmanData : public DataFormat<T> {
public:
    explicit HuffmanData(const std::vector<T>& input_array, size_t max_value);
    
    std::vector<T> get_data() const override;
    size_t get_size() const override;

private:
    int buildHuffmanTree(size_t max_value);
    void generateCodes(int index, const std::vector<bool>& currentCode);
    std::vector<bool> compress(const std::vector<T>& input);
    std::vector<T> decompress(const std::vector<bool>& compressed) const;

    std::vector<T> data_array_;
    std::vector<unsigned> freq_array_;
    std::vector<std::vector<bool>> codes_;
    std::vector<bool> compressed_data_;

    struct Compare {
        const std::vector<unsigned>& freqRef;
        Compare(const std::vector<unsigned>& freq) : freqRef(freq) {}
        
        bool operator()(int l, int r) const {
            return freqRef[l] > freqRef[r];
        }
    };
};

// Explicit template instantiation for common types
template class HuffmanData<int>;
template class HuffmanData<long long>;
