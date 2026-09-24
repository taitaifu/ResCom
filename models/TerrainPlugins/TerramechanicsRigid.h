#pragma once

#include "TerrainGridMap.h"

#include <array>
#include <memory>
#include <string>

#include "chrono/core/ChQuaternion.h"
#include "chrono/core/ChVector3.h"
#include "chrono/physics/ChBody.h"

namespace roversim {

class TerramechanicsRigid {
  public:
    struct Params {
        bool enable_model = true;

        // Wheel geometry
        double r = 0.12;
        double b = 0.18;
        double h = 0.01;
        double fn_ave = 220.0;

        // Empirical coefficients
        double c1 = 0.5;
        double c2 = -0.3;
        double c3 = 0.0;
        double c_T1 = 0.214;
        double c_d1 = -0.626;
        double c_d2 = 0.308;
        double c_d3 = -0.448;

        // Numerical limits
        double sinkage_step = 0.01;
        double sinkage_max = 0.15;
        double first_damp_coef = 5000.0;
        double second_damp_coef = 5000.0;
        double fn_max = 5000.0;

        int print_step = 1000;

        double time_step = 5e-3;

    };

    struct WheelKinematics {
        chrono::ChVector3d wheel_pos;
        chrono::ChQuaternion<> wheel_rot;
        chrono::ChVector3d wheel_lin_vel;
        chrono::ChVector3d wheel_ang_vel;

        // Explicit local basis for terramechanics.
        // x: rolling direction, y: lateral direction, z: wheel-up/body-up direction
        chrono::ChVector3d axis_x;
        chrono::ChVector3d axis_y;
        chrono::ChVector3d axis_z;

        // Wheel spin axis in world coordinates.
        chrono::ChVector3d spin_axis;

        bool use_joint_spin = false;
        double joint_spin = 0.0;
    };

    struct Output {
        chrono::ChVector3d force_world{0, 0, 0};
        chrono::ChVector3d torque_world{0, 0, 0};
        chrono::ChVector3d force_local{0, 0, 0};
        chrono::ChVector3d torque_local{0, 0, 0};
        chrono::ChVector3d contact_normal{0, 0, 1};
        chrono::ChVector3d contact_point{0, 0, 0};
        chrono::ChVector3d frame_ex{1, 0, 0};
        chrono::ChVector3d frame_ey{0, 1, 0};
        chrono::ChVector3d frame_ez{0, 0, 1};
        std::array<double, 3> vel_local{0.0, 0.0, 0.0};
        double angle_velocity = 0.0;
        double sinkage = 0.0;
        double slip = 0.0;
        double beta = 0.0;
        double theta1 = 0.0;
        double theta2 = 0.0;
        double thetam = 0.0;
        double sigma_m = 0.0;
        double tao_m = 0.0;
        bool in_contact = false;
    };

    struct TouchArea {
        chrono::ChVector3d normal{0, 0, 1};
        chrono::ChVector3d point{0, 0, 0};
        bool valid = false;
        double Kc = 0.0;
        double Kphi = 0.0;
        double n0 = 0.0;
        double n1 = 0.0;
        double c = 0.0;
        double phi = 0.0;
        double K = 0.0;
    };

  public:
    TerramechanicsRigid();
    explicit TerramechanicsRigid(const Params& params);

    void SetParams(const Params& params) { m_params = params; }
    const Params& GetParams() const { return m_params; }

    void SetTerrain(std::shared_ptr<TerrainGridMap> terrain) { m_terrain = std::move(terrain); }
    std::shared_ptr<TerrainGridMap> GetTerrain() const { return m_terrain; }

    Output Evaluate(const WheelKinematics& kin, int iter = 0);
    void ApplyToBody(const Output& out, chrono::ChBody& wheel_body) const;

    // Convenience builder for a Chrono wheel body + chassis body.
    static WheelKinematics BuildWheelKinematics(const chrono::ChBody& wheel_body,
                                                const chrono::ChBody& chassis_body,
                                                const chrono::ChVector3d& spin_axis_local,
                                                const chrono::ChVector3d& rolling_axis_local = chrono::ChVector3d(1, 0, 0),
                                                const chrono::ChVector3d& lateral_axis_local = chrono::ChVector3d(0, 1, 0),
                                                const chrono::ChVector3d& up_axis_local = chrono::ChVector3d(0, 0, 1));

  private:
    static chrono::ChVector3d NormalizeSafe(const chrono::ChVector3d& v,
                                            const chrono::ChVector3d& fallback = chrono::ChVector3d(0, 0, 1));
    static double Clamp(double v, double lo, double hi);

    TouchArea CalculateTouchArea(const WheelKinematics& kin, double theta1_in) const;

    double CalculateSlip(double angv, const std::array<double, 3>& local_lin_velocity) const;
    double CalculateSFlag(double s) const;
    double CalculateBetaFlag(double s) const;
    double CalculateBeta(const std::array<double, 3>& local_lin_velocity) const;
    double CalculationSinkage(const chrono::ChVector3d& wheel_pos,
                              const chrono::ChVector3d& plane_point,
                              const chrono::ChVector3d& plane_normal,
                              double sinkage_last) const;
    double CalculationTheta1(double z_sinkage) const;
    double LimitDampingCoef(const std::array<double, 3>& local_lin_velocity) const;
    double LimitSustainForce(double fn_in) const;
    double CalculateRj(double s) const;

    void WheelTerrainInteraction(double angle_velocity,
                                 const std::array<double, 3>& vel_local_real,
                                 double s,
                                 double beta,
                                 double theta1,
                                 double theta2,
                                 double thetam,
                                 double sigma_m,
                                 double Kc,
                                 double Kphi,
                                 double c,
                                 double phi,
                                 double K,
                                 double n,
                                 Output& out) const;

    void StaticModel(const std::array<double, 3>& vel_local_real,
                     double theta1,
                     double theta2,
                     double thetam,
                     double Kc,
                     double Kphi,
                     double c,
                     double phi,
                     double K,
                     const chrono::ChVector3d& wheel_pos,
                     const chrono::ChVector3d& wheel_pos_org,
                     const chrono::ChVector3d& normal_vector,
                     const chrono::ChVector3d& contact_force_dir_world,
                     Output& out) const;

  private:
    Params m_params;
    std::shared_ptr<TerrainGridMap> m_terrain;

    // per-wheel memory terms
    double m_theta1_last = 0.01;
    double m_sinkage_last = 0.0;
    bool m_sinkage_initialized = false;
    bool m_in_contact_last = false;
    chrono::ChVector3d m_wheel_pos_org{0, 0, 0};
};

}  // namespace roversim
