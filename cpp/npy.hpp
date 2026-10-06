// Minimal reader/writer for 2D float64, C-order .npy files (format version 1.0 / 2.0).
#pragma once

#include <cstdint>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

struct Array2D {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<double> data;  // row-major
};

inline Array2D load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error("not a .npy file: " + path);
    unsigned char version[2];
    f.read(reinterpret_cast<char*>(version), 2);

    std::uint32_t header_len = 0;
    if (version[0] == 1) {
        std::uint16_t len16 = 0;
        f.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, ' ');
    f.read(header.data(), header_len);

    if (header.find("'descr': '<f8'") == std::string::npos)
        throw std::runtime_error("expected dtype <f8 in " + path);
    if (header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("expected C-order array in " + path);
    std::smatch m;
    if (!std::regex_search(header, m, std::regex(R"('shape':\s*\((\d+),\s*(\d+)\))")))
        throw std::runtime_error("expected a 2D shape in " + path);

    Array2D a;
    a.rows = std::stoll(m[1]);
    a.cols = std::stoll(m[2]);
    a.data.resize(static_cast<std::size_t>(a.rows * a.cols));
    f.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(a.data.size() * sizeof(double)));
    if (!f) throw std::runtime_error("truncated data in " + path);
    return a;
}

inline void save(const std::string& path, const double* data, std::int64_t rows, std::int64_t cols) {
    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                         std::to_string(cols) + "), }";
    // Pad so that magic(6) + version(2) + len(2) + header is a multiple of 64, ending in '\n'.
    const std::size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write("\x93NUMPY\x01\x00", 8);
    const auto len16 = static_cast<std::uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&len16), 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(rows * cols * sizeof(double)));
    if (!f) throw std::runtime_error("failed writing " + path);
}

}  // namespace npy
