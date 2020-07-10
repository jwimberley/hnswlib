#include <iostream>
#include <array>
#include <numeric>

#include "hnswlib/hnswlib.h"

using L2Space = hnswlib::L2Space;
using HNSW = hnswlib::HierarchicalNSW<float>;

constexpr size_t N = 100;
constexpr size_t M = 10;

int main(int, char** args) {
    L2Space l2space(1);
    HNSW graph(&l2space,N);
    std::array<float,N> data;
    std::iota(data.begin(),data.end(),1.0);
    for (size_t idx = 0; idx < N; ++idx) {
        graph.addPoint(static_cast<void*>(&data[idx]),idx);
    }

    // Now run a test
    graph.setEf(10);    
    float test_idx = atof(args[1]);
    auto neighbors = graph.approxSortKnn(static_cast<void*>(&test_idx),N);
    for (const auto& n : neighbors) {
        std::cout << n.second << " -- distance " << n.first << std::endl;
    }
    
    return 0;
};
