// CPU FP64 baselines with Eigen (native Eigen GEMM, no BLAS backend) + OpenMP.
//
// Subcommands (inputs/outputs are .npy, timings are printed as one JSON object):
//   dgemm --a A.npy --b B.npy --out C.npy  --threads 6,14,16 --repeats 3
//   adi   --m M.npy --u0 U0.npy --out U.npy --steps 20  --threads ... --repeats 1
//   ftcs  --u0 U0.npy --out U.npy --steps 1000 --r 0.2 --threads ... --repeats 1
//
// Python (src/benchmark_cpu.py) generates the inputs exactly like the GPU benchmarks
// and evaluates the accuracy, so CPU and GPU are compared on identical data.
// The output file holds the result of the last thread configuration; the JSON
// reports the max abs difference between thread configurations (determinism check).

#include <Eigen/Core>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy.hpp"

namespace {

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMat = Eigen::Map<RowMat>;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string command;
    std::map<std::string, std::string> kv;

    std::string get(const std::string& key, const std::string& fallback = "") const {
        auto it = kv.find(key);
        if (it != kv.end()) return it->second;
        if (fallback.empty()) throw std::runtime_error("missing --" + key);
        return fallback;
    }
};

Args parse_args(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("usage: cpu_bench {dgemm|adi|ftcs|version} --key value ...");
    Args a;
    a.command = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument " + key);
        a.kv[key.substr(2)] = argv[i + 1];
    }
    return a;
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) out.push_back(std::stoi(item));
    return out;
}

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

void set_threads(int n) {
    omp_set_num_threads(n);
    Eigen::setNbThreads(n);
}

// Spin up the OpenMP pool so that thread creation is not timed.
void warm_up_threads() {
    RowMat x = RowMat::Ones(256, 256);
    RowMat y = x * x;
    volatile double sink = y(0, 0);
    (void)sink;
}

std::string json_list(const std::vector<double>& v) {
    std::ostringstream os;
    os.precision(9);
    os << "[";
    for (std::size_t i = 0; i < v.size(); ++i) os << (i ? ", " : "") << v[i];
    os << "]";
    return os.str();
}

struct RunResult {
    int threads;
    std::vector<double> times;
};

void print_json(const std::string& command, std::int64_t n, const std::vector<RunResult>& runs,
                double max_thread_diff, const std::string& extra = "") {
    std::ostringstream os;
    os.precision(9);
    os << "{\"command\": \"" << command << "\", \"n\": " << n << ", \"eigen_version\": \"" << EIGEN_WORLD_VERSION
       << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\", \"simd\": \"" << Eigen::SimdInstructionSetsInUse()
       << "\", \"max_abs_diff_between_thread_configs\": " << max_thread_diff << extra << ", \"runs\": [";
    for (std::size_t i = 0; i < runs.size(); ++i) {
        os << (i ? ", " : "") << "{\"threads\": " << runs[i].threads << ", \"times_s\": " << json_list(runs[i].times)
           << "}";
    }
    os << "]}";
    std::cout << os.str() << std::endl;
}

// Run `body` for every thread count; `snapshot` returns the current result so that
// results of different thread counts can be compared.
template <typename Body, typename Result>
std::pair<std::vector<RunResult>, double> sweep_threads(const std::vector<int>& threads, int repeats, Body body,
                                                        Result result) {
    std::vector<RunResult> runs;
    RowMat first;
    double max_diff = 0.0;
    for (int t : threads) {
        set_threads(t);
        warm_up_threads();
        RunResult r{t, {}};
        for (int k = 0; k < repeats; ++k) r.times.push_back(body());
        const RowMat& cur = result();
        if (first.size() == 0) {
            first = cur;
        } else {
            max_diff = std::max(max_diff, (cur - first).cwiseAbs().maxCoeff());
        }
        std::cerr << "[cpu_bench] threads=" << t << " median of " << repeats << " runs done" << std::endl;
        runs.push_back(r);
    }
    return {runs, max_diff};
}

int cmd_dgemm(const Args& args) {
    npy::Array2D a_raw = npy::load(args.get("a"));
    npy::Array2D b_raw = npy::load(args.get("b"));
    if (a_raw.cols != b_raw.rows) throw std::runtime_error("shape mismatch");
    MapMat a(a_raw.data.data(), a_raw.rows, a_raw.cols);
    MapMat b(b_raw.data.data(), b_raw.rows, b_raw.cols);
    RowMat c = RowMat::Zero(a.rows(), b.cols());  // touch the pages before timing

    auto [runs, diff] = sweep_threads(
        parse_int_list(args.get("threads")), std::stoi(args.get("repeats", "3")),
        [&] {
            auto t0 = Clock::now();
            c.noalias() = a * b;
            return seconds_since(t0);
        },
        [&]() -> const RowMat& { return c; });

    npy::save(args.get("out"), c.data(), c.rows(), c.cols());
    print_json("dgemm", a.rows(), runs, diff);
    return 0;
}

int cmd_adi(const Args& args) {
    npy::Array2D m_raw = npy::load(args.get("m"));
    npy::Array2D u0_raw = npy::load(args.get("u0"));
    const int steps = std::stoi(args.get("steps"));
    MapMat m(m_raw.data.data(), m_raw.rows, m_raw.cols);
    MapMat u0(u0_raw.data.data(), u0_raw.rows, u0_raw.cols);
    RowMat u = u0;
    RowMat tmp = RowMat::Zero(u0.rows(), u0.cols());

    auto [runs, diff] = sweep_threads(
        parse_int_list(args.get("threads")), std::stoi(args.get("repeats", "1")),
        [&] {
            u = u0;
            auto t0 = Clock::now();
            for (int s = 0; s < steps; ++s) {
                tmp.noalias() = m * u;
                u.noalias() = tmp * m;
            }
            return seconds_since(t0);
        },
        [&]() -> const RowMat& { return u; });

    npy::save(args.get("out"), u.data(), u.rows(), u.cols());
    print_json("adi", u.rows(), runs, diff, ", \"steps\": " + std::to_string(steps));
    return 0;
}

int cmd_ftcs(const Args& args) {
    npy::Array2D u0_raw = npy::load(args.get("u0"));
    const int steps = std::stoi(args.get("steps"));
    const double r = std::stod(args.get("r"));
    const std::int64_t n = u0_raw.rows;
    const std::int64_t w = n + 2;

    // Padded (n+2)^2 buffers with zero Dirichlet boundary, same layout as the CUDA kernel.
    std::vector<double> init(static_cast<std::size_t>(w * w), 0.0);
    for (std::int64_t i = 0; i < n; ++i)
        std::copy_n(&u0_raw.data[i * n], n, &init[(i + 1) * w + 1]);
    std::vector<double> buf0(init.size()), buf1(init.size());
    double* result_ptr = nullptr;
    RowMat result(n, n);

    auto [runs, diff] = sweep_threads(
        parse_int_list(args.get("threads")), std::stoi(args.get("repeats", "1")),
        [&] {
            std::copy(init.begin(), init.end(), buf0.begin());
            std::fill(buf1.begin(), buf1.end(), 0.0);
            double* u = buf0.data();
            double* v = buf1.data();
            auto t0 = Clock::now();
            for (int s = 0; s < steps; ++s) {
#pragma omp parallel for schedule(static)
                for (std::int64_t i = 1; i <= n; ++i) {
                    const double* up = u + (i - 1) * w;
                    const double* mid = u + i * w;
                    const double* down = u + (i + 1) * w;
                    double* out = v + i * w;
#pragma omp simd
                    for (std::int64_t j = 1; j <= n; ++j)
                        out[j] = mid[j] + r * (up[j] + down[j] + mid[j - 1] + mid[j + 1] - 4.0 * mid[j]);
                }
                std::swap(u, v);
            }
            const double elapsed = seconds_since(t0);
            result_ptr = u;
            return elapsed;
        },
        [&]() -> const RowMat& {
            for (std::int64_t i = 0; i < n; ++i)
                std::copy_n(result_ptr + (i + 1) * w + 1, n, result.data() + i * n);
            return result;
        });

    npy::save(args.get("out"), result.data(), n, n);
    print_json("ftcs", n, runs, diff, ", \"steps\": " + std::to_string(steps));
    return 0;
}

bool defined_avx2() {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
}

bool defined_eigen_fma() {
#ifdef EIGEN_VECTORIZE_FMA
    return true;
#else
    return false;
#endif
}

int cmd_version() {
    // Eigen 3.4's SimdInstructionSetsInUse() does not list AVX2/FMA, so report the macros too.
    std::cout << "{\"eigen_version\": \"" << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << "\", \"avx2\": " << (defined_avx2() ? "true" : "false")
              << ", \"eigen_fma\": " << (defined_eigen_fma() ? "true" : "false")
              << ", \"simd\": \"" << Eigen::SimdInstructionSetsInUse()
              << "\", \"compiler\": \"" << __VERSION__ << "\", \"openmp\": " << _OPENMP
              << ", \"omp_max_threads\": " << omp_get_max_threads() << "}" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.command == "dgemm") return cmd_dgemm(args);
        if (args.command == "adi") return cmd_adi(args);
        if (args.command == "ftcs") return cmd_ftcs(args);
        if (args.command == "version") return cmd_version();
        throw std::runtime_error("unknown command: " + args.command);
    } catch (const std::exception& e) {
        std::cerr << "[cpu_bench] error: " << e.what() << std::endl;
        return 1;
    }
}
