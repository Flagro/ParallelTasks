#pragma once

#include <vector>

class UniqueFinder {
public:
    UniqueFinder() = default;
    UniqueFinder(const std::vector<int>& data, int nunique);
    ~UniqueFinder();

    std::vector<int> find_unique();

private:
    std::vector<int> data;
    int unique_values;
};
