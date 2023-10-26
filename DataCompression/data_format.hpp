#pragma once

#include <vector>

template <typename T>
class DataFormat {
public:
    virtual ~DataFormat() = default;

    // Return decompressed data
    virtual std::vector<T> get_data() const = 0;

    // Return the size of the compressed data in memory
    virtual size_t get_size() const = 0;
};
