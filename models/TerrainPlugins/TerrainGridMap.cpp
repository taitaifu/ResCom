#include "TerrainPlugins/TerrainGridMap.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace roversim {

namespace {
constexpr double kEps = 1e-12;
constexpr int kColsPerPoint = 10;  // x y z + 7 terrain params

inline bool NearlyEqual(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}
}  // namespace

TerrainGridMap::TerrainGridMap(const Config& cfg) : m_cfg(cfg) {}

void TerrainGridMap::SetConfig(const Config& cfg) {
    m_cfg = cfg;
}

bool TerrainGridMap::LoadFromFile(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        throw std::runtime_error("TerrainGridMap: cannot open file: " + filename);
    }

    std::vector<double> raw;
    raw.reserve(static_cast<std::size_t>(m_cfg.x_grids > 0 ? m_cfg.x_grids : 1024) *
                static_cast<std::size_t>(m_cfg.y_grids > 0 ? m_cfg.y_grids : 1024) *
                kColsPerPoint);

    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        ++line_no;
        if (line.empty())
            continue;

        std::istringstream iss(line);
        double vals[kColsPerPoint];
        for (int i = 0; i < kColsPerPoint; ++i) {
            if (!(iss >> vals[i])) {
                throw std::runtime_error(
                    "TerrainGridMap: invalid row format at line " + std::to_string(line_no) +
                    ", expected 10 numeric columns: x y z p0 p1 p2 p3 p4 p5 p6");
            }
        }

        double extra = 0.0;
        if (iss >> extra) {
            throw std::runtime_error(
                "TerrainGridMap: too many columns at line " + std::to_string(line_no) +
                ", expected exactly 10 columns");
        }

        for (int i = 0; i < kColsPerPoint; ++i) {
            raw.push_back(vals[i]);
        }
    }

    if (raw.empty()) {
        throw std::runtime_error("TerrainGridMap: input file is empty");
    }

    if (raw.size() % kColsPerPoint != 0) {
        throw std::runtime_error("TerrainGridMap: raw data size is not divisible by 10");
    }

    const std::size_t point_count = raw.size() / kColsPerPoint;

    if (m_cfg.x_grids <= 1 || m_cfg.y_grids <= 1) {
        throw std::runtime_error("TerrainGridMap: x_grids and y_grids must both be > 1");
    }

    const std::size_t expected_points =
        static_cast<std::size_t>(m_cfg.x_grids) * static_cast<std::size_t>(m_cfg.y_grids);

    if (point_count != expected_points) {
        throw std::runtime_error(
            "TerrainGridMap: point count mismatch, expected " + std::to_string(expected_points) +
            " rows, got " + std::to_string(point_count));
    }

    // 保存连续数据
    m_data = std::move(raw);

    // // 当前格式固定为 10 列
    // m_cfg.dtm_params = 10;
    // m_cfg.terrain_params_num = 7;

    // 当前索引方式: cell = iy + y_grids * ix
    m_node_coef = m_cfg.y_grids;

    // 从数据中推断边界和步长
    // 假设文件按 ix 外层, iy 内层 或等价的规则网格顺序存储
    auto at_raw = [&](std::size_t cell, int offset) -> double {
        return m_data[cell * static_cast<std::size_t>(kColsPerPoint) + static_cast<std::size_t>(offset)];
    };

    m_x0 = at_raw(0, 0);
    m_y0 = at_raw(0, 1);

    m_x_step = 0.0;
    m_y_step = 0.0;

    // 推断 y_step: 同一 x 下，相邻 y 的差
    for (int iy = 1; iy < m_cfg.y_grids; ++iy) {
        double dy = at_raw(static_cast<std::size_t>(iy), 1) - at_raw(static_cast<std::size_t>(iy - 1), 1);
        if (std::abs(dy) > kEps) {
            m_y_step = std::abs(dy);
            break;
        }
    }

    // 推断 x_step: 相邻 x 行首点的差
    for (int ix = 1; ix < m_cfg.x_grids; ++ix) {
        std::size_t c_cur = static_cast<std::size_t>(ix) * static_cast<std::size_t>(m_node_coef);
        std::size_t c_pre = static_cast<std::size_t>(ix - 1) * static_cast<std::size_t>(m_node_coef);
        double dx = at_raw(c_cur, 0) - at_raw(c_pre, 0);
        if (std::abs(dx) > kEps) {
            m_x_step = std::abs(dx);
            break;
        }
    }

    if (m_x_step <= 0.0 || m_y_step <= 0.0) {
        throw std::runtime_error("TerrainGridMap: failed to infer x_step or y_step from file");
    }

    m_cfg.x_min = m_x0;
    m_cfg.y_min = m_y0;
    m_cfg.x_max = m_x0 + m_x_step * static_cast<double>(m_cfg.x_grids-1);
    m_cfg.y_max = m_y0 + m_y_step * static_cast<double>(m_cfg.y_grids-1);

    std::cout << "TerrainGridMap: boundaries set to [" << m_cfg.x_min << ", " << m_cfg.x_max << "] x [" << m_cfg.y_min << ", " << m_cfg.y_max << "]" << std::endl;
    std::cout << "TerrainGridMap: step sizes inferred as x_step = " << m_x_step << ", y_step = " << m_y_step << std::endl;
    std::cout << "TerrainGridMap: node coefficient set to " << m_node_coef << std::endl;

    // 基本一致性检查
    for (int ix = 0; ix < m_cfg.x_grids; ++ix) {
        for (int iy = 0; iy < m_cfg.y_grids; ++iy) {
            const std::size_t c = static_cast<std::size_t>(FlatIndex(ix, iy));
            const double x_expect = m_x0 + static_cast<double>(ix) * m_x_step;
            const double y_expect = m_y0 + static_cast<double>(iy) * m_y_step;
            const double x_read = at_raw(c, 0);
            const double y_read = at_raw(c, 1);

            if (!NearlyEqual(x_expect, x_read, 1e-6) || !NearlyEqual(y_expect, y_read, 1e-6)) {
                throw std::runtime_error(
                    "TerrainGridMap: file grid order or spacing is inconsistent with x_grids/y_grids");
            }
        }
    }

    return true;
}

void TerrainGridMap::OverrideTerrainParams(const std::vector<double>& terrain_params) {
    if (terrain_params.size() != static_cast<std::size_t>(m_cfg.terrain_params_num)) {
        throw std::runtime_error("TerrainGridMap: terrain parameter override size mismatch");
    }
    if (m_data.empty()) {
        throw std::runtime_error("TerrainGridMap: cannot override terrain parameters before loading data");
    }

    const std::size_t point_count = m_data.size() / static_cast<std::size_t>(m_cfg.dtm_params);
    for (std::size_t cell = 0; cell < point_count; ++cell) {
        const std::size_t base = cell * static_cast<std::size_t>(m_cfg.dtm_params) + 3;
        for (int j = 0; j < m_cfg.terrain_params_num; ++j) {
            m_data[base + static_cast<std::size_t>(j)] = terrain_params[static_cast<std::size_t>(j)];
        }
    }
}

bool TerrainGridMap::InBounds(double x, double y) const {
    return (x >= m_cfg.x_min && x <= m_cfg.x_max &&
            y >= m_cfg.y_min && y <= m_cfg.y_max);
}

int TerrainGridMap::FlatIndex(int ix, int iy) const {
    return m_node_coef * ix + iy;
}

TerrainGridMap::Sample TerrainGridMap::SamplePoint(double x, double y) const {
    Sample out;
    out.terrain_params.assign(static_cast<std::size_t>(m_cfg.terrain_params_num), 0.0);

    if (!InBounds(x, y) || m_data.empty() || m_x_step <= 0.0 || m_y_step <= 0.0) {
        return out;
    }

    // 为了双线性插值，需要 ix+1 和 iy+1 有效
    double fx = (x - m_cfg.x_min) / m_x_step;
    double fy = (y - m_cfg.y_min) / m_y_step;

    int ix = static_cast<int>(std::floor(fx));
    int iy = static_cast<int>(std::floor(fy));

    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (ix >= m_cfg.x_grids - 1) ix = m_cfg.x_grids - 2;
    if (iy >= m_cfg.y_grids - 1) iy = m_cfg.y_grids - 2;

    const double x0 = m_cfg.x_min + ix * m_x_step;
    const double y0 = m_cfg.y_min + iy * m_y_step;

    const double tx = (x - x0) / m_x_step;
    const double ty = (y - y0) / m_y_step;

    const double u0 = (1.0 - tx) * (1.0 - ty);
    const double u1 = (1.0 - tx) * ty;
    const double u2 = tx * (1.0 - ty);
    const double u3 = tx * ty;

    const int c0 = FlatIndex(ix,     iy);
    const int c1 = FlatIndex(ix,     iy + 1);
    const int c2 = FlatIndex(ix + 1, iy);
    const int c3 = FlatIndex(ix + 1, iy + 1);

    const auto at = [&](int cell, int offset) -> double {
        const std::size_t idx =
            static_cast<std::size_t>(m_cfg.dtm_params) * static_cast<std::size_t>(cell) +
            static_cast<std::size_t>(offset);
        if (idx >= m_data.size()) {
            throw std::runtime_error("TerrainGridMap: index overflow while sampling terrain map");
        }
        return m_data[idx];
    };

    // 列定义: x y z p0 p1 p2 p3 p4 p5 p6
    const double z0 = at(c0, 2);
    const double z1 = at(c1, 2);
    const double z2 = at(c2, 2);
    const double z3 = at(c3, 2);
    out.z = u0 * z0 + u1 * z1 + u2 * z2 + u3 * z3;

    for (int j = 0; j < m_cfg.terrain_params_num; ++j) {
        const int offset = 3 + j;
        const double p0 = at(c0, offset);
        const double p1 = at(c1, offset);
        const double p2 = at(c2, offset);
        const double p3 = at(c3, offset);
        out.terrain_params[static_cast<std::size_t>(j)] = u0 * p0 + u1 * p1 + u2 * p2 + u3 * p3;
    }

    out.valid = true;
    return out;
}

}  // namespace roversim
