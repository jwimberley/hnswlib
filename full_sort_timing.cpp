#include <iostream>
#include <array>
#include <vector>

#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

using namespace std::chrono;
using Time = std::chrono::system_clock;
using FloatMsec = std::chrono::duration<float,std::milli>;

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

    auto tuple_distance = [] (const std::array<float,dim>& lhs, const std::array<float,dim>& rhs) {
        float dist = 0.0;
        for (size_t k = 0; k < dim; k++) {
            float delta = lhs[k] - rhs[k];
            dist += delta*delta;
        }
        return dist;
    };

    // Fill dataa
    size_t N = 1;
    while (N <= 1000000) {
        N *= 10;

        std::vector<std::array<float,3>> data_original(N,{0,0,0});
        for (auto& vals : data_original) {
            vals = gen_tuple();
        }

        std::vector< std::vector<std::array<float,3>> > datas(10);
        for (size_t test = 0; test < 10; ++test) {
            datas[test] = data_original;
        }
        
        // Now run a test
        auto start = Time::now();
        for (size_t test = 0; test < 10; ++test) {
            auto& test_data = datas[test];
            auto coords = gen_tuple();
            std::sort(test_data.begin(),test_data.end(),[&] (const std::array<float,dim>& lhs, std::array<float,dim>& rhs) {
                    return tuple_distance(lhs,coords) < tuple_distance(rhs,coords);
                });
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
