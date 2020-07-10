#include <iostream>
#include <array>
#include <vector>
#include <numeric>

#include <random>
#include <chrono>
#include <cmath>

#include "hnswlib/hnswlib.h"

using namespace std::chrono;
using Time = std::chrono::system_clock;
using FloatMsec = std::chrono::duration<float,std::milli>;

using L2Space = hnswlib::L2Space;
using HNSW = hnswlib::HierarchicalNSW<float>;

int main(int, char** args) {

    std::random_device rd;
    std::mt19937 g(rd());
    std::normal_distribution<> dist {0,1};

    const size_t dim = 3;
    auto gen_tuple = [&] () -> std::array<float,dim> {
        std::array<float,dim> vals;
        for (size_t k = 0; k < dim; k++) {
            vals[k] = dist(g);
        }
        return vals;
    };

    // Fill dataa
    size_t N = 1;
    while (N <= 100000) {
        N *= 10;

        std::vector<std::array<float,3>> data(N,{0,0,0});
        for (auto& vals : data) {
            vals = gen_tuple();
        }
        
        L2Space l2space(dim);
        HNSW graph(&l2space,N);
        for (size_t idx = 0; idx < N; ++idx) {
            graph.addPoint(static_cast<void*>(&data[idx]),idx);
        }
        
        // Now run a test
        graph.setEf(10);
        auto start = Time::now();
        for (size_t test = 0; test < 10; ++test) {
            auto coords = gen_tuple();
            auto neighbors = graph.approxSortKnn(static_cast<void*>(&coords),N);
        }
        auto stop = Time::now();
        auto elapsed = duration_cast<FloatMsec>(stop-start).count();
        std::cout << "N = " << N
                  << " => " << elapsed << " msec"
                  << ", per N " << (elapsed/N) << " msec"
                  << ", per N log N " << (elapsed/N/log(N)) << " msec"
                  << std::endl;
    }
        
    return 0;
};
