#pragma once

#include <string>
#include <vector>

namespace roversim {

class TerrainGridMap {
  public:
    struct Config {
        double x_min = -10.0;
        double x_max = 10.0;
        double y_min = -10.0;
        double y_max = 10.0;

        // 网格点数量，不是单元数量
        int x_grids = 0;
        int y_grids = 0;

        // 每个网格点的数据列数:
        // x y z + 7 terrain params = 10
        int dtm_params = 10;

        // 土壤参数数量
        // 当前默认顺序: Kc, Kphi, n0, n1, c, phi, K
        int terrain_params_num = 7;
    };

    struct Sample {
        double z = -100.0;
        std::vector<double> terrain_params;
        bool valid = false;
    };

  public:
    TerrainGridMap() = default;
    explicit TerrainGridMap(const Config& cfg);

    void SetConfig(const Config& cfg);
    const Config& GetConfig() const { return m_cfg; }

    // 读取矩阵格式文件
    // 每行一个网格点，10列:
    // x y z Kc Kphi n0 n1 c phi K
    bool LoadFromFile(const std::string& filename);
    void OverrideTerrainParams(const std::vector<double>& terrain_params);

    // 在任意 (x,y) 位置双线性插值采样
    Sample SamplePoint(double x, double y) const;

    double GetX0() const { return m_x0; }
    double GetY0() const { return m_y0; }
    double GetXStep() const { return m_x_step; }
    double GetYStep() const { return m_y_step; }
    int GetNodeCoef() const { return m_node_coef; }

  private:
    bool InBounds(double x, double y) const;
    int FlatIndex(int ix, int iy) const;

  private:
    Config m_cfg;

    // 从文件和网格配置推断得到
    double m_x0 = 0.0;
    double m_y0 = 0.0;
    double m_x_step = 0.0;
    double m_y_step = 0.0;

    // 当前索引方式: cell = iy + node_coef * ix
    int m_node_coef = 0;

    // 连续一维存储，每个点占 dtm_params 个数
    std::vector<double> m_data;
};

}  // namespace roversim
