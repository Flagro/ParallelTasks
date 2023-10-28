#pragma once

#include <vector>

class UniqueFinder {
public:
    UniqueFinder() = default;
    UniqueFinder::UniqueFinder(const std::vector<int>& data, int nunique) : data(data), unique_values(nunique) {}

    std::vector<int> find_unique();

private:
    std::vector<int> data;
    int unique_values;
};
