#include "TerrainPlugins/TerramechanicsRigid.h"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace roversim {

namespace {
    constexpr double kEps = 1e-12;
    constexpr double kPi = 3.14159265358979323846;
    struct ContactFrame {
        chrono::ChVector3d ex;
        chrono::ChVector3d ey;
        chrono::ChVector3d ez;
        bool valid = false;
    };

    static ContactFrame BuildContactFrameFromNormalAndHeading(
        const chrono::ChVector3d& normal_world,
        const chrono::ChVector3d& heading_hint_world) 
    {
        ContactFrame frame;

        chrono::ChVector3d ez = normal_world;
        const double nz = ez.Length();
        if (nz <= 1e-12) {
            return frame;
        }
        ez /= nz;

        // 先把几何前向轴投影到接触平面内，对应旧版由法向+偏航角共同定局部 x 轴的思路
        chrono::ChVector3d ex = heading_hint_world - chrono::Vdot(heading_hint_world, ez) * ez;
        const double nx = ex.Length();
        if (nx <= 1e-12) {
            chrono::ChVector3d ref(1, 0, 0);
            if (std::fabs(chrono::Vdot(ref, ez)) > 0.9) {
                ref = chrono::ChVector3d(0, 1, 0);
            }
            ex = ref - chrono::Vdot(ref, ez) * ez;
        }
        ex = ex.GetNormalized();

        // 右手系
        chrono::ChVector3d ey = chrono::Vcross(ez, ex);
        const double ny = ey.Length();
        if (ny <= 1e-12) {
            return frame;
        }
        ey /= ny;

        // 再正交化一次
        ex = chrono::Vcross(ey, ez).GetNormalized();

        frame.ex = ex;
        frame.ey = ey;
        frame.ez = ez;
        frame.valid = true;
        return frame;
    }

    static std::array<double, 3> ToLocalVector(
        const chrono::ChVector3d& v_world,
        const ContactFrame& frame) 
    {
        return {
            chrono::Vdot(v_world, frame.ex),
            chrono::Vdot(v_world, frame.ey),
            chrono::Vdot(v_world, frame.ez)
        };
    }

    static chrono::ChVector3d ToWorldVector(
        const std::array<double, 3>& v_local,
        const ContactFrame& frame) 
    {
        return frame.ex * v_local[0] + frame.ey * v_local[1] + frame.ez * v_local[2];
    }

    static chrono::ChVector3d ToWorldVector(
        const chrono::ChVector3d& v_local,
        const ContactFrame& frame) 
    {
        return frame.ex * v_local.x() + frame.ey * v_local.y() + frame.ez * v_local.z();
    }
}

chrono::ChVector3d CalculateProjectedDisplacementOnPlane(
    const chrono::ChVector3d& normal_vector,
    const chrono::ChVector3d& wheel_pos_0,
    const chrono::ChVector3d& wheel_pos_1) 
{
    chrono::ChVector3d wheel_distance = wheel_pos_1 - wheel_pos_0;

    const double nn = chrono::Vdot(normal_vector, normal_vector);
    if (nn <= 1e-12) {
        return chrono::ChVector3d(0, 0, 0);
    }

    const double proj = chrono::Vdot(wheel_distance, normal_vector);
    chrono::ChVector3d j = wheel_distance - normal_vector * (proj / nn);
    return j;
}

TerramechanicsRigid::TerramechanicsRigid() : m_params() {}
TerramechanicsRigid::TerramechanicsRigid(const Params& params) : m_params(params) {}

chrono::ChVector3d TerramechanicsRigid::NormalizeSafe(const chrono::ChVector3d& v,
                                                      const chrono::ChVector3d& fallback) {
    const double n = v.Length();
    if (n <= kEps)
        return fallback;
    return (1.0 / n) * v;
}

double TerramechanicsRigid::Clamp(double v, double lo, double hi) {
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

TerramechanicsRigid::WheelKinematics TerramechanicsRigid::BuildWheelKinematics(
    const chrono::ChBody& wheel_body,
    const chrono::ChBody& chassis_body,
    const chrono::ChVector3d& spin_axis_local,
    const chrono::ChVector3d& rolling_axis_local,
    const chrono::ChVector3d& lateral_axis_local,
    const chrono::ChVector3d& up_axis_local) {
    WheelKinematics kin;
    kin.wheel_pos = wheel_body.GetPos();
    kin.wheel_rot = wheel_body.GetRot();
    kin.wheel_lin_vel = wheel_body.GetPosDt();
    kin.wheel_ang_vel = wheel_body.GetAngVelParent();
    // std::cout << "[BuildWheelKinematics] Wheel velocity = ("
    //           << kin.wheel_lin_vel.x() << ", "
    //           << kin.wheel_lin_vel.y() << ", "
    //           << kin.wheel_lin_vel.z() << ")\n";
    // std::cout << "[BuildWheelKinematics] Wheel angular velocity = ("
    //           << kin.wheel_ang_vel.x() << ", "
    //           << kin.wheel_ang_vel.y() << ", "
    //           << kin.wheel_ang_vel.z() << ")\n";

    // const chrono::ChVector3d axis_x_local = rolling_axis_local;
    // const chrono::ChVector3d axis_x_parent = chassis_body.TransformDirectionLocalToParent(axis_x_local);
    // const chrono::ChVector3d axis_x_norm = NormalizeSafe(axis_x_parent, chrono::ChVector3d(1, 0, 0));
    // std::cout << "[BuildWheelKinematics] axis_x local  = ("
    //         << axis_x_local.x() << ", "
    //         << axis_x_local.y() << ", "
    //         << axis_x_local.z() << ")\n";
    // std::cout << "[BuildWheelKinematics] axis_x parent = ("
    //         << axis_x_parent.x() << ", "
    //         << axis_x_parent.y() << ", "
    //         << axis_x_parent.z() << ")\n";
    // std::cout << "[BuildWheelKinematics] axis_x norm   = ("
    //         << axis_x_norm.x() << ", "
    //         << axis_x_norm.y() << ", "
    //         << axis_x_norm.z() << ")\n";
    // kin.axis_x = axis_x_norm;
    // 这些轴只保留为几何参考，不直接作为 terramechanics 局部系
    kin.axis_x = NormalizeSafe(chassis_body.TransformDirectionLocalToParent(rolling_axis_local), chrono::ChVector3d(1, 0, 0));
    kin.axis_y = NormalizeSafe(chassis_body.TransformDirectionLocalToParent(lateral_axis_local), chrono::ChVector3d(0, 1, 0));
    kin.axis_z = NormalizeSafe(chassis_body.TransformDirectionLocalToParent(up_axis_local), chrono::ChVector3d(0, 0, 1));
    kin.spin_axis = NormalizeSafe(wheel_body.TransformDirectionLocalToParent(spin_axis_local), chrono::ChVector3d(0, 1, 0));
    return kin;
}

TerramechanicsRigid::TouchArea TerramechanicsRigid::CalculateTouchArea(const WheelKinematics& kin, double theta1_in) const {
    if (!m_terrain) {
        throw std::runtime_error("TerramechanicsRigid: terrain map is not set");
    }

    const double r = m_params.r;
    const double b = m_params.b;

    // Build wheel contact frame exactly in the same spirit as the Gazebo code:
    // x: rolling direction, y: lateral direction, z: wheel-up direction.
    const chrono::ChVector3d ex = NormalizeSafe(kin.axis_x, chrono::ChVector3d(1, 0, 0));
    const chrono::ChVector3d ey = NormalizeSafe(kin.axis_y, chrono::ChVector3d(0, 1, 0));
    const chrono::ChVector3d ez = NormalizeSafe(kin.axis_z, chrono::ChVector3d(0, 0, 1));
    // const chrono::ChVector3d ez = NormalizeSafe(kin.axis_z, chrono::ChVector3d(0, 0, 1));
    // chrono::ChVector3d ex = kin.axis_x - chrono::Vdot(kin.axis_x, ez) * ez;
    // ex = NormalizeSafe(ex, chrono::ChVector3d(1, 0, 0));
    // chrono::ChVector3d ey = NormalizeSafe(chrono::Vcross(ez, ex), chrono::ChVector3d(0, 1, 0));
    // ex = NormalizeSafe(chrono::Vcross(ey, ez), chrono::ChVector3d(1, 0, 0));

    const auto to_world = [&](const chrono::ChVector3d& p_local) {
        return kin.wheel_pos + ex * p_local.x() + ey * p_local.y() + ez * p_local.z();
    };

    const chrono::ChVector3d pw1_local(r * std::sin(theta1_in),  b * 0.5, -r * std::cos(theta1_in));
    const chrono::ChVector3d pw2_local(r * std::sin(theta1_in), -b * 0.5, -r * std::cos(theta1_in));
    const chrono::ChVector3d pw3_local(-r * std::sin(theta1_in), 0.0,     -r * std::cos(theta1_in));

    chrono::ChVector3d pa = to_world(pw1_local);
    chrono::ChVector3d pb = to_world(pw2_local);
    chrono::ChVector3d pc = to_world(pw3_local);

    const auto s1 = m_terrain->SamplePoint(pa.x(), pa.y());
    const auto s2 = m_terrain->SamplePoint(pb.x(), pb.y());
    const auto s3 = m_terrain->SamplePoint(pc.x(), pc.y());

    TouchArea ta;
    if (!s1.valid || !s2.valid || !s3.valid) {
        return ta;
    }

    pa.z() = s1.z;
    pb.z() = s2.z;
    pc.z() = s3.z;

    chrono::ChVector3d n = chrono::Vcross(pb - pc, pa - pc);
    if (n.z() < 0.0)
        n = -n;

    ta.normal = NormalizeSafe(n, chrono::ChVector3d(0, 0, 1));
    ta.point = pa;
    ta.valid = true;

    // Terrain params ordering follows the Gazebo implementation: Kc, Kphi, n0, n1, c, phi, K.
    ta.Kc = (s1.terrain_params[0] + s2.terrain_params[0] + s3.terrain_params[0]) / 3.0;
    ta.Kphi = (s1.terrain_params[1] + s2.terrain_params[1] + s3.terrain_params[1]) / 3.0;
    ta.n0 = (s1.terrain_params[2] + s2.terrain_params[2] + s3.terrain_params[2]) / 3.0;
    ta.n1 = (s1.terrain_params[3] + s2.terrain_params[3] + s3.terrain_params[3]) / 3.0;
    ta.c = (s1.terrain_params[4] + s2.terrain_params[4] + s3.terrain_params[4]) / 3.0;
    ta.phi = (s1.terrain_params[5] + s2.terrain_params[5] + s3.terrain_params[5]) / 3.0;
    ta.K = (s1.terrain_params[6] + s2.terrain_params[6] + s3.terrain_params[6]) / 3.0;
    return ta;
}

double TerramechanicsRigid::CalculateSlip(double angv, const std::array<double, 3>& v) const {
    // const double rs = m_params.r + 0.5 * m_params.h;
    const double rs = m_params.r;
    const double vel_real = v[0];
    double del_v = 0.0;
    if (angv > 0.0) {
        del_v = rs * angv - vel_real;
    } else {
        del_v = std::fabs(rs * angv) - std::fabs(vel_real);
    }

    double s = 0.0;
    if (std::fabs(del_v) <= 1e-3 || std::fabs(angv) < 1e-3) {
        s = 0.0;
    } else if (angv * vel_real < 0.0) {
        s = 1.0;
    } else if (del_v > 1e-3) {
        s = 1.0 - std::fabs(vel_real / (rs * angv));
    } else {
        s = std::fabs(rs * angv) / std::fabs(vel_real) - 1.0;
    }

    s = Clamp(s, -1.0, 1.0);
    // std::cout << "[CalculateSlip] angv = " << angv << ", vel_real = " << vel_real << ", slip = " << s << "\n";
    return s;
}

double TerramechanicsRigid::CalculateSFlag(double s) const {
    return (s >= 0.0) ? 1.0 : -1.0;
}

double TerramechanicsRigid::CalculateBetaFlag(double beta) const {
    return (beta >= 0.0) ? -1.0 : 1.0;
}

double TerramechanicsRigid::CalculateBeta(const std::array<double, 3>& v) const {
    if (std::fabs(v[1]) <= 1e-3) {
        return 0.0;
    }
    double beta = kPi / 2.0 - std::atan2(v[0], v[1]);
    if (beta > kPi / 2.0) {
        beta -= kPi;
    }
    if (std::fabs(beta) < 1e-2) {
        return 0.0;
    }
    return beta;
}

double TerramechanicsRigid::CalculationSinkage(const chrono::ChVector3d& wheel_pos,
                                               const chrono::ChVector3d& plane_point,
                                               const chrono::ChVector3d& plane_normal,
                                               double sinkage_last) const {
    double sinkage = m_params.r - chrono::Vdot(plane_normal, wheel_pos - plane_point) / plane_normal.Length();
    if (sinkage < 0.0) {
        sinkage = 0.0;
    } else {
        if (std::fabs(sinkage - sinkage_last) > m_params.sinkage_step) {
            if (sinkage > sinkage_last)
                sinkage = sinkage_last + m_params.sinkage_step;
            else
                sinkage = sinkage_last - m_params.sinkage_step;
        }
        if (sinkage > m_params.sinkage_max)
            sinkage = m_params.sinkage_max;
    }
    return sinkage;
}

double TerramechanicsRigid::CalculationTheta1(double z_sinkage) const {
    return std::acos(Clamp((m_params.r - z_sinkage) / m_params.r, -1.0, 1.0));
}

double TerramechanicsRigid::LimitDampingCoef(const std::array<double, 3>& v) const {
    return (v[2] > 0.0) ? m_params.first_damp_coef : m_params.second_damp_coef;
}

double TerramechanicsRigid::LimitSustainForce(double fn_in) const {
    return std::min(fn_in, m_params.fn_max);
}

double TerramechanicsRigid::CalculateRj(double s) const {
    if (s < 0.15)
        return m_params.r;
    if (s > 0.5)
        return m_params.r + m_params.h;
    return m_params.r + m_params.h * (s - 0.15) / (0.5 - 0.15);
}

void TerramechanicsRigid::WheelTerrainInteraction(double angle_velocity,
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
                                                  Output& out) const {
    (void)Kc;
    (void)Kphi;
    (void)n;

    // std::cout << "[WheelTerrainInteraction] theta1 = " << theta1 << ", theta2 = " << theta2 << ", thetam = " << thetam << "\n";
    // std::cout << "[WheelTerrainInteraction] sigma_m = " << sigma_m << "\n";

    const double r = m_params.r;
    const double b = m_params.b;
    const double h = m_params.h;
    const double rs = r + 0.5 * h;

    const double dz = LimitDampingCoef(vel_local_real);
    const double theta_m2 = std::max(thetam - theta2, 1e-6);
    const double theta_1m = std::max(theta1 - thetam, 1e-6);
    const double theta_12 = std::max(theta1 - theta2, 1e-6);

    const double A_1 = (std::cos(thetam) - std::cos(theta2)) / theta_m2;
    const double A_2 = (std::cos(thetam) - std::cos(theta1)) / theta_1m;
    const double A = A_1 + A_2;
    const double B_1 = (std::sin(thetam) - std::sin(theta2)) / theta_m2;
    const double B_2 = (std::sin(thetam) - std::sin(theta1)) / theta_1m;
    const double B = B_1 + B_2;
    const double C = theta_12 / 2.0;

    const double Rj = CalculateRj(s);
    const double theta11 = std::acos(Clamp(r * std::cos(theta1) / Rj, -1.0, 1.0));

    double jx = 0.0;
    double jy = 0.0;
    if (s >= 0.0) {
        jx = rs * ((theta11 - thetam) - (1.0 - s) * (std::sin(theta11) - std::sin(thetam)));
        jy = r * (1.0 - s) * (theta11 - thetam) * std::tan(std::fabs(beta));
    } else {
        jx = -rs * ((theta11 - thetam) - (std::sin(theta11) - std::sin(thetam)) / (1.0 + 0.9 * s));
        jy = r * (theta11 - thetam) * std::tan(std::fabs(beta)) / (1.0 + 0.99 * s);
    }
    const double exp_jk_x = 1.0 - std::exp(-jx / std::max(K, 1e-8));
    const double exp_jk_y = 1.0 - std::exp(-jy / std::max(K, 1e-8));
    const double tao_m = (c + sigma_m * std::tan(phi)) * exp_jk_x;
    // std::cout<<"[WheelTerrainInteraction] j = " << j << ", one_jdk = " << one_jdk << ", tao_m = " << tao_m << "\n";

    // double Fs_add1 = 1 - std::exp(-r * (1 - s) * (theta11 - thetam) * std::tan(std::fabs(beta)) / K);
    // double Fs = 0.5 * r * b * theta_12 * (c + sigma_m * std::tan(0.5 * phi)) * Fs_add1;
    const double Fs = r * b * C * (c + sigma_m * std::tan(0.5 * phi)) * exp_jk_y;
    // std::cout << "[WheelTerrainInteraction] C = " << C << ", one_jdk = " << one_jdk << ", Fs = " << Fs << "\n";

    const double Fn1 = r * b * A * sigma_m + rs * b * tao_m * B;
    const double Fn = LimitSustainForce(Fn1);

    const double coef_tempT = 1.0 + (m_params.c_T1 * ((m_params.fn_ave - Fn) / m_params.fn_ave));
    const double Mr1 = rs * rs * C * (b * c + coef_tempT * Fn * std::tan(phi) / std::max(r * A, 1e-6)) * exp_jk_x;
    const double Mr2 = 1.0 + rs * B * std::tan(phi) * exp_jk_x / std::max(r * A, 1e-6);
    double Mr = Mr1 / std::max(Mr2, 1e-6);

    const double coef_s = m_params.c_d1 + m_params.c_d2 * s;
    const double coef_t = m_params.c_d3 * ((m_params.fn_ave - Fn) / m_params.fn_ave);
    const double Fdp1 = (A / C + B * B / std::max(A * C, 1e-6)) * Mr / rs - B * Fn / std::max(A, 1e-6);
    const double Fdp = Fdp1 * ((1.0 + coef_s) * (1.0 + coef_t));

    double FX_local = (s >= 0.0) ? Fdp : -Fdp;
    if (s < 0.0) {
        if (FX_local > 0.0) {
            FX_local = 0.0;
        }
        FX_local -= 200.0 * std::fabs(vel_local_real[0]);
    } else if (FX_local < -100.0) {
        FX_local = -100.0;
    }

    double FY_local = (std::fabs(vel_local_real[1]) > 1e-6)
                          ? -std::copysign(std::fabs(Fs), vel_local_real[1])
                          : 0.0;
    FY_local -= Clamp(600.0 * vel_local_real[1], -0.25 * Fn, 0.25 * Fn);

    const double FZ_sub = Clamp(dz * vel_local_real[2], -dz * 5.0, dz * 5.0);
    FX_local = Clamp(FX_local, -0.8 * Fn, 0.8 * Fn);
    FY_local = Clamp(FY_local, -0.3 * Fn, 0.3 * Fn);
    double FZ_local = Clamp(std::max(0.0, Fn - FZ_sub), 0.0, m_params.fn_max);

    if (angle_velocity < 0.0) {
        Mr = -Mr;
    }

    const double MX_local = r * FY_local;
    const double MY_local = Clamp(-Mr, -r * Fn, r * Fn);
    double MZ_local = Clamp(std::sin(thetam) * r * FY_local, -0.4 * r * Fn, 0.4 * r * Fn);

    if (angle_velocity < 0.0) {
        FX_local = -FX_local;
        MZ_local = -MZ_local;
    }
    if (angle_velocity * vel_local_real[0] < 0.0) {
        FX_local -= 200.0 * vel_local_real[0];
    }

    out.force_local = chrono::ChVector3d(FX_local, FY_local, FZ_local);
    out.torque_local = chrono::ChVector3d(MX_local, MY_local, MZ_local);
    out.tao_m = tao_m;
}

void TerramechanicsRigid::StaticModel(const std::array<double, 3>& vel_local_real,
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
                                      Output& out) const {
    const double r = m_params.r;
    const double b = m_params.b;

    const double theta_12 = std::max(theta1 - theta2, 1e-6);

    const double dz = LimitDampingCoef(vel_local_real);

    const double sigma_m = (Kc / b + Kphi) * r *
                           std::max(std::cos(thetam) - std::cos(theta1), 0.0);
    out.sigma_m = sigma_m;

    const double Fn1 = r * b * theta_12 * sigma_m;
    const double Fn = LimitSustainForce(Fn1);

    chrono::ChVector3d delta_world = wheel_pos - wheel_pos_org;
    if (normal_vector.Length() > 1e-12) {
        const auto normal = normal_vector.GetNormalized();
        delta_world -= chrono::Vdot(delta_world, normal) * normal;
    }
    const double delta_norm = delta_world.Length();

    chrono::ChVector3d friction_world(0, 0, 0);
    if (delta_norm > 1e-6) {
        friction_world = (-1.0 / delta_norm) * delta_world;
    }

    const double friction_force =
        (r * b * theta_12 * c + Fn * std::tan(phi)) *
        (1.0 - std::exp(-delta_norm / std::max(K, 1e-8)));
    chrono::ChVector3d ex = NormalizeSafe(contact_force_dir_world, chrono::ChVector3d(1, 0, 0));
    chrono::ChVector3d ez = NormalizeSafe(normal_vector, chrono::ChVector3d(0, 0, 1));
    chrono::ChVector3d ey = NormalizeSafe(chrono::Vcross(ez, ex), chrono::ChVector3d(0, 1, 0));
    ex = NormalizeSafe(chrono::Vcross(ey, ez), chrono::ChVector3d(1, 0, 0));

    const chrono::ChVector3d friction = friction_force * friction_world;
    double FX = chrono::Vdot(friction, ex) - Clamp(1200.0 * vel_local_real[0], -0.25 * Fn, 0.25 * Fn);
    double FY = chrono::Vdot(friction, ey) - Clamp(1800.0 * vel_local_real[1], -0.35 * Fn, 0.35 * Fn);
    const double FZ_sub = Clamp(dz * vel_local_real[2], -dz * 5.0, dz * 5.0);
    double FZ = Clamp(std::max(0.0, Fn - FZ_sub), 0.0, m_params.fn_max);

    FX = Clamp(FX, -0.35 * Fn, 0.35 * Fn);
    FY = Clamp(FY, -0.35 * Fn, 0.35 * Fn);

    const double MX = r * FY;
    const double MY = -r * FX;
    const double MZ = std::sin(out.thetam) * r * FY;

    out.force_local = chrono::ChVector3d(FX, FY, FZ);
    out.torque_local = chrono::ChVector3d(MX, MY, MZ);
}

// void TerramechanicsRigid::StaticModel(const std::array<double, 3>& vel_local_real,
//                                       double theta1,
//                                       double theta2,
//                                       double sigma_m,
//                                       double c,
//                                       double phi,
//                                       const chrono::ChVector3d& wheel_pos,
//                                       const chrono::ChVector3d& wheel_pos_org,
//                                       const chrono::ChVector3d& normal_vector,
//                                       const chrono::ChVector3d& contact_force_dir_world,
//                                       Output& out) const {
//     const double r = m_params.r;
//     const double b = m_params.b;

//     const double theta_m2 = out.thetam - theta2;
//     const double theta_1m = theta1 - out.thetam;
//     const double theta_12 = theta1 - theta2;
//     const double A_1 = (std::cos(out.thetam) - std::cos(theta2)) / theta_m2;
//     const double A_2 = (std::cos(out.thetam) - std::cos(theta1)) / theta_1m;
//     const double A = A_1 + A_2;

//     const double dz = LimitDampingCoef(vel_local_real);
//     const double Fn1 = r * b * A * sigma_m;
//     const double Fn = LimitSustainForce(Fn1);

//     chrono::ChVector3d deltaj = wheel_pos - wheel_pos_org;
//     const double deltaj_len = deltaj.Length();
//     chrono::ChVector3d fri_dir = chrono::ChVector3d(0, 0, 0);
//     if (deltaj_len > 1e-4)
//         fri_dir = (-1.0 / deltaj_len) * deltaj;

//     const double friction_force = (r * b * theta_12 * c + Fn * std::tan(phi)) *
//                                   (1.0 - std::exp(-deltaj_len / std::max(1e-8, 1.0)));
//     const chrono::ChVector3d friction_world = friction_force * fri_dir;

//     const double FX_sub = Clamp(2000.0 * vel_local_real[0], -1000.0, 1000.0);
//     const double FY_sub = Clamp(10000.0 * vel_local_real[1], -1000.0, 1000.0);
//     const double FZ_sub = Clamp(dz * vel_local_real[2], -dz * 5.0, dz * 5.0);

//     // Project friction onto user-provided local frame.
//     // const double fx = chrono::Vdot(friction_world, contact_force_dir_world) - FX_sub;
//     // const double fy = -FY_sub;
//     // const double fz = Fn - FZ_sub;
//     const double fx = chrono::Vdot(friction_world, contact_force_dir_world);
//     const double fy = 0;
//     const double fz = Fn;

//     out.force_local = chrono::ChVector3d(fx, fy, fz);
//     out.torque_local = chrono::ChVector3d(r * fy, -(r * fx), std::sin(out.thetam) * r * fy);
//     (void)normal_vector;
// }
TerramechanicsRigid::Output TerramechanicsRigid::Evaluate(const WheelKinematics& kin, int iter) {
    if (!m_terrain) {
        throw std::runtime_error("TerramechanicsRigid: terrain not set before Evaluate");
    }

    Output out;
    if (!m_params.enable_model)
        return out;

    // 先用上一时刻 theta1 去找接触面
    TouchArea ta = CalculateTouchArea(kin, m_theta1_last);
    if (!ta.valid) {
        out.in_contact = false;
        m_in_contact_last = false;
        return out;
    }

    out.contact_normal = ta.normal;
    out.contact_point = ta.point;

    // 用接触面法向 + 几何前向提示轴，构造轮地接触局部系
    ContactFrame contact_frame =
        BuildContactFrameFromNormalAndHeading(ta.normal, kin.axis_x);

    if (!contact_frame.valid) {
        return out;
    }

    out.frame_ex = contact_frame.ex;
    out.frame_ey = contact_frame.ey;
    out.frame_ez = contact_frame.ez;

    // 世界系线速度和角速度转到接触局部系
    const std::array<double, 3> vel_local_real =
        ToLocalVector(kin.wheel_lin_vel, contact_frame);

    const double angle_velocity =
        kin.use_joint_spin ? kin.joint_spin
                           : chrono::Vdot(kin.wheel_ang_vel, NormalizeSafe(kin.spin_axis, contact_frame.ey));
    out.vel_local = vel_local_real;
    out.angle_velocity = angle_velocity;

    const double s = CalculateSlip(angle_velocity, vel_local_real);
    const double beta = CalculateBeta(vel_local_real);

    out.slip = s;
    out.beta = beta;

    double raw_sinkage = CalculationSinkage(kin.wheel_pos, ta.point, ta.normal, m_sinkage_last);
    if (!m_sinkage_initialized) {
        raw_sinkage = m_params.r - chrono::Vdot(ta.normal, kin.wheel_pos - ta.point) / ta.normal.Length();
        raw_sinkage = Clamp(raw_sinkage, 0.0, m_params.sinkage_max);
        m_sinkage_initialized = true;
    }
    out.sinkage = raw_sinkage;

    const double contact_threshold = 5e-5;
    const bool has_contact = (raw_sinkage > contact_threshold);

    if (!has_contact) {
        out.in_contact = false;
        m_in_contact_last = false;
        m_sinkage_last = raw_sinkage;
        m_theta1_last = 0.01;
        m_wheel_pos_org = kin.wheel_pos;
        return out;
    }

    out.in_contact = true;
    m_in_contact_last = true;

    const double effective_sinkage = raw_sinkage - contact_threshold;
    const double theta1 = CalculationTheta1(effective_sinkage);
    const double theta2 = m_params.c3 * theta1;
    const double thetam = (m_params.c1 + m_params.c2 * s) * theta1;
    const double n = ta.n0 + ta.n1 * std::fabs(s);
    const double sigma_m = (ta.Kc / m_params.b + ta.Kphi) * std::pow(m_params.r, n) *
                           std::pow(std::cos(thetam) - std::cos(theta1), n);
    out.theta1 = theta1;
    out.theta2 = theta2;
    out.thetam = thetam;
    out.sigma_m = sigma_m;

    // std::cout << "[Evaluate] angle_velocity = " << angle_velocity << "\n";

    if (std::fabs(angle_velocity) >= 0.01) {
        WheelTerrainInteraction(angle_velocity, vel_local_real, s, beta, theta1, theta2, thetam,
                                sigma_m, ta.Kc, ta.Kphi, ta.c, ta.phi, ta.K, n, out);
        // std::cout << "[Evaluate] Using dynamic model\n";
    } else {
        StaticModel(vel_local_real, theta1, theta2, thetam, ta.Kc, ta.Kphi, ta.c, ta.phi, ta.K,
                    kin.wheel_pos, m_wheel_pos_org, ta.normal, contact_frame.ex, out);
        // std::cout << "[Evaluate] Using static model\n";
    }
    // std::cout << "[Evaluate] Force world = ("
    //           << out.force_local.x() << ", "
    //           << out.force_local.y() << ", "
    //           << out.force_local.z() << ")\n";
    // Transform local force/torque into world frame with the explicit terramechanics basis.
    // 局部力和局部力矩转回世界系
    out.force_world = ToWorldVector(out.force_local, contact_frame);
    out.torque_world = ToWorldVector(out.torque_local, contact_frame);
    // std::cout << "[Evaluate] Force world = ("
    //           << out.force_world.x() << ", "
    //           << out.force_world.y() << ", "
    //           << out.force_world.z() << ")\n";

    m_sinkage_last = raw_sinkage;
    m_theta1_last = theta1;
    m_wheel_pos_org = kin.wheel_pos;

    (void)iter;
    return out;
}
// TerramechanicsRigid::Output TerramechanicsRigid::Evaluate(const WheelKinematics& kin, int iter) {
//     if (!m_terrain) {
//         throw std::runtime_error("TerramechanicsRigid: terrain not set before Evaluate");
//     }

//     Output out;
//     if (!m_params.enable_model)
//         return out;

//     const chrono::ChVector3d ex = NormalizeSafe(kin.axis_x, chrono::ChVector3d(1, 0, 0));
//     const chrono::ChVector3d ey = NormalizeSafe(kin.axis_y, chrono::ChVector3d(0, 1, 0));
//     const chrono::ChVector3d ez = NormalizeSafe(kin.axis_z, chrono::ChVector3d(0, 0, 1));
//     const double angle_velocity = chrono::Vdot(kin.wheel_ang_vel, NormalizeSafe(kin.spin_axis, ey));
//     // std::cout << "[Evaluate] Wheel angular velocity = " << kin.wheel_ang_vel
//     //           << " after normalization = " << angle_velocity << "\n";

//     const std::array<double, 3> vel_local_real = {
//         chrono::Vdot(kin.wheel_lin_vel, ex),
//         chrono::Vdot(kin.wheel_lin_vel, ey),
//         chrono::Vdot(kin.wheel_lin_vel, ez)
//     };

//     const double s = CalculateSlip(angle_velocity, vel_local_real);
    
//     const double beta = CalculateBeta(vel_local_real);

//     out.slip = s;
//     out.beta = beta;

//     TouchArea ta = CalculateTouchArea(kin, m_theta1_last);
//     // std::cout << "[Evaluate] TouchArea = ("
//     //           << ta.point.x() << ", "
//     //           << ta.point.y() << ", "
//     //           << ta.point.z() << ")\n"
//     //           << "Wheel pos = ("
//     //           << kin.wheel_pos.x() << ", "
//     //           << kin.wheel_pos.y() << ", "
//     //           << kin.wheel_pos.z() << ")\n";
//     if (!ta.valid) {
//         // std::cout << "[Evaluate] TouchArea is invalid\n";
//         return out;
//     }

//     // std::cout << "[Evaluate] TouchArea = ("
//     //           << ta.point.x() << ", "
//     //           << ta.point.y() << ", "
//     //           << ta.point.z() << ")\n"
//     //           << "Wheel pos = ("
//     //           << kin.wheel_pos.x() << ", "
//     //           << kin.wheel_pos.y() << ", "
//     //           << kin.wheel_pos.z() << ")\n";

//     out.contact_normal = ta.normal;
//     out.contact_point = ta.point;
//     // std::cout << "[Evaluate] Contact point = ("
//     //           << ta.point.x() << ", "
//     //           << ta.point.y() << ", "
//     //           << ta.point.z() << ")\n";

//     const double sinkage = CalculationSinkage(kin.wheel_pos, ta.point, ta.normal, m_sinkage_last);
//     out.sinkage = sinkage;

//     if (sinkage <= 2e-5) {
//         m_sinkage_last = 2e-4;
//         m_theta1_last = 0.01;
//         m_wheel_pos_org = kin.wheel_pos;
//         return out;
//     }
//     double sinkage_rate = 0.0;

//     if (m_params.time_step > 1e-12) {
//         sinkage_rate = (sinkage - m_sinkage_last) / m_params.time_step;
//     }
//     const std::array<double, 3> vel_local_wheel = {
//         chrono::Vdot(kin.wheel_lin_vel, ex),
//         chrono::Vdot(kin.wheel_lin_vel, ey),
//         sinkage_rate
//     };
//     // std::cout << "[Evaluate] vel_local_wheel = ("
//     //           << vel_local_wheel[0] << ", "
//     //           << vel_local_wheel[1] << ", "
//     //           << vel_local_wheel[2] << ")\n";

//     const double theta1 = CalculationTheta1(sinkage);
//     const double theta2 = m_params.c3 * theta1;
//     const double thetam = (m_params.c1 + m_params.c2 * s) * theta1;
//     const double n = ta.n0 + ta.n1 * std::fabs(s);
//     const double sigma_m = (ta.Kc / m_params.b + ta.Kphi) * std::pow(m_params.r, n) *
//                            std::pow(std::cos(thetam) - std::cos(theta1), n);
//     // std::cout << "[Evaluate] theta1 = " << theta1 << ", theta2 = " << theta2 << ", thetam = " << thetam << "\n";
//     // std::cout << "[Evaluate] sigma_m = " << sigma_m << "s = " << s <<"\n";
//     out.theta1 = theta1;
//     out.theta2 = theta2;
//     out.thetam = thetam;
//     out.sigma_m = sigma_m;
//     out.in_contact = true;

//     std::cout << "[Evaluate] angle_velocity = " << angle_velocity << "\n";

//     if (std::fabs(angle_velocity) >= 0.01) {
//         WheelTerrainInteraction(angle_velocity, vel_local_wheel, s, beta, theta1, theta2, thetam,
//                                 sigma_m, ta.Kc, ta.Kphi, ta.c, ta.phi, ta.K, n, out);
//         // std::cout << "[Evaluate] Using dynamic model\n";
//     } else {
//         StaticModel(vel_local_wheel, theta1, theta2, sigma_m, ta.c, ta.phi, ta.K,
//                     kin.wheel_pos, m_wheel_pos_org, ta.normal, ex, out);
//         // std::cout << "[Evaluate] Using static model\n";
//     }
//     // std::cout << "[Evaluate] Force world = ("
//     //           << out.force_local.x() << ", "
//     //           << out.force_local.y() << ", "
//     //           << out.force_local.z() << ")\n";
//     // Transform local force/torque into world frame with the explicit terramechanics basis.
//     out.force_world = ex * out.force_local.x() + ey * out.force_local.y() + ez * out.force_local.z();
//     out.torque_world = ex * out.torque_local.x() + ey * out.torque_local.y() + ez * out.torque_local.z();
//     // std::cout << "[Evaluate] Force world = ("
//     //           << out.force_world.x() << ", "
//     //           << out.force_world.y() << ", "
//     //           << out.force_world.z() << ")\n";

//     m_sinkage_last = sinkage;
//     m_theta1_last = theta1;
//     m_wheel_pos_org = kin.wheel_pos;

//     (void)iter;
//     return out;
// }

void TerramechanicsRigid::ApplyToBody(const Output& out, chrono::ChBody& wheel_body) const {
    if (!out.in_contact)
        return;

    chrono::ChVector3d F = out.force_world;
    chrono::ChVector3d T = out.torque_world;

    auto clamp_comp = [](double v, double lim) {
        if (v > lim) return lim;
        if (v < -lim) return -lim;
        return v;
    };
    // std::cout << "[ApplyToBody] Force world before clamping = ("
    //           << F.x() << ", "
    //           << F.y() << ", "
    //           << F.z() << ")\n";

    // F.x() = clamp_comp(F.x(), 0.0);
    // F.y() = clamp_comp(F.y(), 0.0);
    // F.z() = clamp_comp(F.z(), 200.0);
    F.x() = 0.0;
    F.y() = 0.0;
    F.z() = 10.0;

    T.x() = clamp_comp(T.x(), 0.0);
    T.y() = clamp_comp(T.y(), 0.0);
    T.z() = clamp_comp(T.z(), 0.0);

    static std::unordered_map<chrono::ChBody*, unsigned int> accumulator_indices;
    auto [it, inserted] = accumulator_indices.emplace(&wheel_body, 0);
    if (inserted) {
        it->second = wheel_body.AddAccumulator();
    }

    wheel_body.EmptyAccumulator(it->second);
    wheel_body.AccumulateForce(it->second, F, wheel_body.GetPos(), false);
    wheel_body.AccumulateTorque(it->second, T, false);
}

}  // namespace roversim
