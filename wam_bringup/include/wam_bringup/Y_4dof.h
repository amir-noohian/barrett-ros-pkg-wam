//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// calculate_Y.cpp
//
// Code generation for function 'calculate_Y'
//

// Include files
#include <eigen3/Eigen/Dense>
#include <array>
#include <cmath>

// Function Definitions
void calculate_Y(const double theta[4], const double thetad[4],
                 const double thetadd[4], double Y_b[120])
{
  static const signed char b[1440]{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double dv[192];
  double Y44_tmp;
  double Y44_tmp_tmp;
  double b_Y44_tmp;
  double b_Y44_tmp_tmp;
  double c_Y44_tmp;
  double c_Y44_tmp_tmp;
  double d;
  double d1;
  double d10;
  double d11;
  double d12;
  double d13;
  double d14;
  double d15;
  double d16;
  double d17;
  double d18;
  double d19;
  double d2;
  double d20;
  double d21;
  double d22;
  double d23;
  double d24;
  double d25;
  double d26;
  double d27;
  double d28;
  double d29;
  double d3;
  double d30;
  double d31;
  double d32;
  double d33;
  double d34;
  double d35;
  double d36;
  double d37;
  double d38;
  double d39;
  double d4;
  double d40;
  double d41;
  double d42;
  double d43;
  double d44;
  double d45;
  double d46;
  double d47;
  double d48;
  double d49;
  double d5;
  double d50;
  double d51;
  double d52;
  double d53;
  double d54;
  double d55;
  double d56;
  double d57;
  double d58;
  double d59;
  double d6;
  double d60;
  double d61;
  double d62;
  double d63;
  double d64;
  double d65;
  double d66;
  double d67;
  double d68;
  double d69;
  double d7;
  double d70;
  double d71;
  double d72;
  double d73;
  double d74;
  double d75;
  double d76;
  double d77;
  double d78;
  double d79;
  double d8;
  double d80;
  double d81;
  double d82;
  double d83;
  double d84;
  double d85;
  double d86;
  double d9;
  double d_Y44_tmp;
  double d_Y44_tmp_tmp;
  double e_Y44_tmp;
  double e_Y44_tmp_tmp;
  double f_Y44_tmp;
  double f_Y44_tmp_tmp;
  int i1;
  int i2;
  Y44_tmp_tmp = std::sin(theta[3]);
  Y44_tmp = std::cos(2.0 * theta[1]);
  b_Y44_tmp_tmp = std::cos(theta[3]);
  c_Y44_tmp_tmp = std::cos(theta[2]);
  d_Y44_tmp_tmp = std::sin(theta[1]);
  e_Y44_tmp_tmp = std::sin(theta[2]);
  f_Y44_tmp_tmp = std::cos(theta[1]);
  b_Y44_tmp = std::cos(2.0 * theta[2]);
  c_Y44_tmp = std::sin(2.0 * theta[2]);
  d_Y44_tmp = std::sin(2.0 * theta[1]);
  e_Y44_tmp = std::sin(2.0 * theta[3]);
  f_Y44_tmp = std::cos(2.0 * theta[3]);
  dv[0] = 0.0;
  dv[4] = 0.0;
  dv[8] = 0.0;
  dv[12] = 0.0;
  dv[16] = 0.0;
  dv[20] = 0.0;
  dv[24] = 0.0;
  dv[28] = 0.0;
  dv[32] = 0.0;
  dv[36] = thetadd[0];
  dv[40] = thetad[0];
  dv[44] = std::tanh(100.0 * thetad[0]);
  dv[48] = 0.0;
  dv[52] = 0.0;
  dv[56] = 0.0;
  dv[60] = 0.0;
  d = thetad[0] * thetad[1] * d_Y44_tmp;
  dv[64] = (-thetadd[0] * Y44_tmp / 2.0 + d) + thetadd[0] / 2.0;
  dv[68] = 2.0 * thetad[1] * thetad[0] * Y44_tmp + thetadd[0] * d_Y44_tmp;
  d1 = thetad[1] * thetad[1];
  dv[72] = -thetadd[1] * d_Y44_tmp_tmp - d1 * f_Y44_tmp_tmp;
  d2 = thetadd[0] * Y44_tmp / 2.0 - d;
  dv[76] = d2 + thetadd[0] / 2.0;
  d3 = thetadd[1] * f_Y44_tmp_tmp;
  dv[80] = d1 * d_Y44_tmp_tmp - d3;
  dv[84] = 0.0;
  dv[88] = 0.0;
  dv[92] = 0.0;
  dv[96] = 0.0;
  dv[100] = 0.0;
  dv[104] = 0.0;
  dv[108] = 0.0;
  d4 = 80000.0 * thetad[0] * thetad[2];
  d5 = d4 * c_Y44_tmp;
  d6 = 160000.0 * thetad[2] * thetad[1];
  d7 = d6 * d_Y44_tmp_tmp;
  d8 = 80000.0 * d1;
  d9 = d8 * f_Y44_tmp_tmp;
  d /= 2.0;
  d10 = 80000.0 * thetad[1] * thetad[0];
  dv[112] =
      (((((-40000.0 * thetadd[0] * b_Y44_tmp + d5) - 40000.0 * thetadd[0]) *
             Y44_tmp / 160000.0 +
         ((d10 * d_Y44_tmp - d7) + 40000.0 * thetadd[0]) * b_Y44_tmp /
             160000.0) +
        ((-80000.0 * thetadd[1] * d_Y44_tmp_tmp - d9) - d4) * c_Y44_tmp /
            160000.0) +
       d) +
      thetadd[0] / 4.0;
  d11 = 160000.0 * thetad[0] * thetad[2];
  d12 = 160000.0 * d1;
  d13 = 320000.0 * thetad[2] * thetad[1];
  d14 = d12 * f_Y44_tmp_tmp;
  dv[116] =
      ((d11 * b_Y44_tmp + 80000.0 * thetadd[0] * c_Y44_tmp) * Y44_tmp /
           160000.0 +
       ((-160000.0 * thetadd[1] * d_Y44_tmp_tmp - d14) - d11) * b_Y44_tmp /
           160000.0) +
      ((-160000.0 * thetad[1] * thetad[0] * d_Y44_tmp + d13 * d_Y44_tmp_tmp) -
       80000.0 * thetadd[0]) *
          c_Y44_tmp / 160000.0;
  d15 = thetad[2] * thetad[2];
  d16 = 160000.0 * d15;
  d17 = -160000.0 * d1;
  d18 = d17 + d16;
  d19 = 160000.0 * thetadd[0] * c_Y44_tmp_tmp;
  d20 = 160000.0 * thetadd[2] * c_Y44_tmp_tmp;
  dv[120] = ((-2.0 * thetad[0] * thetad[1] * c_Y44_tmp_tmp * Y44_tmp +
              (d11 * e_Y44_tmp_tmp - d19) * d_Y44_tmp / 160000.0) +
             (d18 * e_Y44_tmp_tmp - d20) * d_Y44_tmp_tmp / 160000.0) +
            d3 * e_Y44_tmp_tmp;
  d9 = (80000.0 * thetadd[1] * d_Y44_tmp_tmp + d9) + d4;
  d += (((40000.0 * thetadd[0] * b_Y44_tmp - d5) - 40000.0 * thetadd[0]) *
            Y44_tmp / 160000.0 +
        ((-80000.0 * thetad[1] * thetad[0] * d_Y44_tmp + d7) -
         40000.0 * thetadd[0]) *
            b_Y44_tmp / 160000.0) +
       d9 * c_Y44_tmp / 160000.0;
  dv[124] = d + thetadd[0] / 4.0;
  d5 = d11 * c_Y44_tmp_tmp;
  dv[128] =
      ((2.0 * thetad[0] * thetad[1] * e_Y44_tmp_tmp * Y44_tmp +
        (160000.0 * thetadd[0] * e_Y44_tmp_tmp + d5) * d_Y44_tmp / 160000.0) +
       (160000.0 * thetadd[2] * e_Y44_tmp_tmp + d18 * c_Y44_tmp_tmp) *
           d_Y44_tmp_tmp / 160000.0) +
      d3 * c_Y44_tmp_tmp;
  d3 = thetad[1] * thetad[2] * d_Y44_tmp_tmp;
  d18 = thetadd[2] * f_Y44_tmp_tmp;
  dv[132] = ((d2 - d3) + d18) + thetadd[0] / 2.0;
  dv[136] = 0.0;
  dv[140] = 0.0;
  d2 = 162.0 * thetad[0] * thetad[2];
  d21 = 324.0 * thetad[2] * thetad[1];
  d22 = 3960.0 * d15;
  d23 = 162.0 * d1;
  d24 = 3960.0 * thetadd[0] * c_Y44_tmp_tmp;
  dv[144] =
      (((((((((81.0 * thetadd[0] * b_Y44_tmp - d2 * c_Y44_tmp) +
              7920.0 * thetad[0] * thetad[1] * c_Y44_tmp_tmp) -
             24119.0 * thetadd[0]) *
                Y44_tmp / 160000.0 +
            ((-162.0 * thetad[1] * thetad[0] * d_Y44_tmp +
              d21 * d_Y44_tmp_tmp) -
             81.0 * thetadd[0]) *
                b_Y44_tmp / 160000.0) +
           ((162.0 * thetadd[1] * d_Y44_tmp_tmp + d23 * f_Y44_tmp_tmp) + d2) *
               c_Y44_tmp / 160000.0) +
          ((-3960.0 * thetad[0] * thetad[2] * e_Y44_tmp_tmp + d24) +
           48238.0 * thetad[1] * thetad[0]) *
              d_Y44_tmp / 160000.0) +
         (((3960.0 * d1 - d22) * e_Y44_tmp_tmp +
           3960.0 * thetadd[2] * c_Y44_tmp_tmp) -
          d21) *
             d_Y44_tmp_tmp / 160000.0) -
        0.02475 * f_Y44_tmp_tmp * thetadd[1] * e_Y44_tmp_tmp) +
       0.002025 * thetadd[2] * f_Y44_tmp_tmp) +
      0.15276875 * thetadd[0];
  d2 = 3600.0 * thetadd[0] * b_Y44_tmp_tmp;
  d25 = 3600.0 * thetad[3] * thetad[0];
  d26 = d25 * Y44_tmp_tmp;
  d27 = 7200.0 * thetad[0] * thetad[2];
  d28 = 14400.0 * thetad[2] * thetad[1];
  d29 = thetad[3] * thetad[3];
  d30 = 88000.0 * thetad[0] * thetad[3];
  d31 = (-14400.0 * thetad[0] * thetad[1] * c_Y44_tmp_tmp -
         3600.0 * thetad[0] * thetad[3]) +
        88000.0 * thetadd[0];
  d32 = -7200.0 * d1;
  d33 = (d27 * e_Y44_tmp_tmp +
         (-88000.0 * thetad[0] * thetad[3] - 7200.0 * thetadd[0]) *
             c_Y44_tmp_tmp) -
        176000.0 * thetad[1] * thetad[0];
  d34 = 7200.0 * thetad[1] * thetad[0];
  d35 = 7200.0 * d15;
  d36 = 7200.0 * d29;
  d37 =
      (((d32 + d35) - d36) - 88000.0 * thetadd[3]) * e_Y44_tmp_tmp +
      (-176000.0 * thetad[3] * thetad[2] - 7200.0 * thetadd[2]) * c_Y44_tmp_tmp;
  d38 = 88000.0 * d15;
  d39 = 88000.0 * d29;
  d40 = 88000.0 * thetadd[2] * c_Y44_tmp_tmp;
  d41 = 14400.0 * thetad[3] * thetad[2];
  d42 = ((7200.0 * f_Y44_tmp_tmp * thetadd[1] * e_Y44_tmp_tmp -
          d41 * f_Y44_tmp_tmp) -
         10800.0 * thetad[0] * thetad[3]) -
        88000.0 * thetadd[0];
  d43 = 14400.0 * thetadd[2] * f_Y44_tmp_tmp;
  d44 = 176000.0 * thetad[0] * thetad[1];
  d45 = 7200.0 * d1;
  d46 = d45 * f_Y44_tmp_tmp;
  d47 = 88000.0 * d1 - d38;
  d48 = 7200.0 * thetadd[1] * b_Y44_tmp_tmp;
  d49 = 7200.0 * thetad[3] * thetad[1];
  d50 = d49 * Y44_tmp_tmp;
  d51 = d28 * b_Y44_tmp_tmp;
  d52 = 14400.0 * f_Y44_tmp_tmp;
  d53 = thetad[1] * thetad[3];
  dv[148] =
      (((((((((d2 - d26) * b_Y44_tmp - d27 * b_Y44_tmp_tmp * c_Y44_tmp) +
             ((d44 * c_Y44_tmp_tmp + d30) + 3600.0 * thetadd[0]) *
                 b_Y44_tmp_tmp) +
            d31 * Y44_tmp_tmp) *
               Y44_tmp / 160000.0 +
           (((-7200.0 * thetad[0] * thetad[1] * b_Y44_tmp_tmp * d_Y44_tmp +
              d51 * d_Y44_tmp_tmp) -
             d2) +
            d26) *
               b_Y44_tmp / 160000.0) +
          ((d48 - d50) * d_Y44_tmp_tmp + (d46 + d27) * b_Y44_tmp_tmp) *
              c_Y44_tmp / 160000.0) +
         (((-88000.0 * thetad[0] * thetad[2] * e_Y44_tmp_tmp +
            (-7200.0 * thetad[0] * thetad[3] + 88000.0 * thetadd[0]) *
                c_Y44_tmp_tmp) -
           d34) *
              b_Y44_tmp_tmp +
          d33 * Y44_tmp_tmp) *
             d_Y44_tmp / 160000.0) +
        (((((d47 - d39) + 7200.0 * thetadd[3]) * e_Y44_tmp_tmp + d40) - d28) *
             b_Y44_tmp_tmp +
         d37 * Y44_tmp_tmp) *
            d_Y44_tmp_tmp / 160000.0) +
       (((d52 * (d53 - 6.1111111111111107 * thetadd[1]) * e_Y44_tmp_tmp + d43) -
         d30) +
        10800.0 * thetadd[0]) *
           b_Y44_tmp_tmp / 160000.0) +
      d42 * Y44_tmp_tmp / 160000.0;
  d2 = 3600.0 * thetadd[0] * Y44_tmp_tmp;
  d26 = 88000.0 * thetad[0] * thetad[2];
  d54 = -7200.0 * thetad[3] * thetad[1];
  d55 = 7200.0 * thetadd[1] * Y44_tmp_tmp;
  d56 = d28 * Y44_tmp_tmp;
  dv[152] =
      (((((((((-3600.0 * thetad[3] * thetad[0] * b_Y44_tmp_tmp - d2) *
                  b_Y44_tmp +
              d27 * Y44_tmp_tmp * c_Y44_tmp) +
             d31 * b_Y44_tmp_tmp) +
            ((-176000.0 * thetad[0] * thetad[1] * c_Y44_tmp_tmp - d30) -
             3600.0 * thetadd[0]) *
                Y44_tmp_tmp) *
               Y44_tmp / 160000.0 +
           (((7200.0 * thetad[0] * thetad[1] * Y44_tmp_tmp * d_Y44_tmp -
              d56 * d_Y44_tmp_tmp) +
             d25 * b_Y44_tmp_tmp) +
            d2) *
               b_Y44_tmp / 160000.0) +
          ((d54 * b_Y44_tmp_tmp - d55) * d_Y44_tmp_tmp +
           (d32 * f_Y44_tmp_tmp - d27) * Y44_tmp_tmp) *
              c_Y44_tmp / 160000.0) +
         (d33 * b_Y44_tmp_tmp +
          ((d26 * e_Y44_tmp_tmp +
            (7200.0 * thetad[0] * thetad[3] - 88000.0 * thetadd[0]) *
                c_Y44_tmp_tmp) +
           d34) *
              Y44_tmp_tmp) *
             d_Y44_tmp / 160000.0) +
        (d37 * b_Y44_tmp_tmp +
         (((((-88000.0 * d1 + d38) + d39) - 7200.0 * thetadd[3]) *
               e_Y44_tmp_tmp -
           d40) +
          d28) *
             Y44_tmp_tmp) *
            d_Y44_tmp_tmp / 160000.0) +
       d42 * b_Y44_tmp_tmp / 160000.0) +
      (((7200.0 * f_Y44_tmp_tmp *
             (-2.0 * thetad[3] * thetad[1] + 12.222222222222221 * thetadd[1]) *
             e_Y44_tmp_tmp -
         d43) +
        d30) -
       10800.0 * thetadd[0]) *
          Y44_tmp_tmp / 160000.0;
  d2 = -88000.0 * thetadd[0] * e_Y44_tmp_tmp;
  dv[156] = ((((((-7200.0 * thetad[0] * thetad[2] * b_Y44_tmp -
                  3600.0 * thetadd[0] * c_Y44_tmp) -
                 d44 * e_Y44_tmp_tmp) *
                    Y44_tmp / 160000.0 +
                ((7200.0 * thetadd[1] * d_Y44_tmp_tmp + d46) + d27) *
                    b_Y44_tmp / 160000.0) +
               ((d34 * d_Y44_tmp - d28 * d_Y44_tmp_tmp) + 3600.0 * thetadd[0]) *
                   c_Y44_tmp / 160000.0) +
              (d2 - d26 * c_Y44_tmp_tmp) * d_Y44_tmp / 160000.0) +
             (-88000.0 * thetadd[2] * e_Y44_tmp_tmp + d47 * c_Y44_tmp_tmp) *
                 d_Y44_tmp_tmp / 160000.0) -
            0.55 * f_Y44_tmp_tmp * thetadd[1] * c_Y44_tmp_tmp;
  d25 = 40000.0 * thetad[0] * thetad[2];
  d26 = 40000.0 * thetad[0] * thetad[3];
  d27 = d26 * e_Y44_tmp;
  d30 = 40000.0 * d1;
  d31 = d30 * f_Y44_tmp_tmp;
  d32 = 80000.0 * thetad[2] * thetad[1];
  d33 = 160000.0 * thetad[3] * thetad[2];
  d34 = 160000.0 * f_Y44_tmp_tmp;
  d37 = 60000.0 * thetadd[0] * f_Y44_tmp;
  d40 = 120000.0 * thetad[0] * thetad[3];
  d42 = -162.0 * thetad[0] * thetad[1];
  d43 = d32 * d_Y44_tmp_tmp;
  d44 = 40000.0 * thetadd[1] * d_Y44_tmp_tmp;
  d46 = 80000.0 * thetad[3] * thetad[1];
  d47 = d46 * d_Y44_tmp_tmp * e_Y44_tmp;
  d57 = 120000.0 * thetad[1] * thetad[0];
  d58 = 80000.0 * thetadd[0] * c_Y44_tmp_tmp;
  d59 = 40000.0 * thetad[1] * thetad[0];
  d60 = d34 * thetad[3] * thetad[1] * e_Y44_tmp_tmp;
  d61 = 80000.0 * thetadd[2] * f_Y44_tmp_tmp;
  d62 = thetad[1] - thetad[2];
  d63 = thetad[1] + thetad[2];
  d64 = 80000.0 * thetadd[2] * c_Y44_tmp_tmp;
  d65 = 80000.0 * f_Y44_tmp_tmp;
  d66 = d65 * thetadd[1] * e_Y44_tmp_tmp;
  d67 = d33 * f_Y44_tmp_tmp;
  d3 /= 2.0;
  d18 /= 2.0;
  d68 = 160000.0 * thetad[0] * thetad[1];
  d69 = d33 * c_Y44_tmp_tmp;
  d70 = -80000.0 * thetad[0] * thetad[2];
  d71 = 160000.0 * thetad[3] * thetad[0];
  d72 = d71 * c_Y44_tmp_tmp;
  dv[160] =
      (((((((((((((-20000.0 * thetadd[0] * f_Y44_tmp + d27) -
                  20000.0 * thetadd[0]) *
                     b_Y44_tmp +
                 (d25 * f_Y44_tmp + d25) * c_Y44_tmp) -
                d37) +
               (d68 * c_Y44_tmp_tmp + d40) * e_Y44_tmp) +
              20000.0 * thetadd[0]) *
                 Y44_tmp / 160000.0 +
             ((((d42 * (-246.91358024691357 * f_Y44_tmp - 246.91358024691357) *
                     d_Y44_tmp +
                 (-80000.0 * thetad[2] * thetad[1] * d_Y44_tmp_tmp +
                  20000.0 * thetadd[0]) *
                     f_Y44_tmp) -
                d27) -
               d43) +
              20000.0 * thetadd[0]) *
                 b_Y44_tmp / 160000.0) +
            ((((((-40000.0 * thetadd[1] * d_Y44_tmp_tmp - d31) - d25) *
                    f_Y44_tmp +
                d47) -
               d44) -
              d31) -
             d25) *
                c_Y44_tmp / 160000.0) +
           (((d72 + d57) * f_Y44_tmp +
             (d70 * e_Y44_tmp_tmp + d58) * e_Y44_tmp) -
            d59) *
               d_Y44_tmp / 160000.0) +
          ((((d69 + d32) * d_Y44_tmp_tmp - d60) - d61) - 20000.0 * thetadd[0]) *
              f_Y44_tmp / 160000.0) +
         ((((80000.0 * d62 * d63 * e_Y44_tmp_tmp + d64) * d_Y44_tmp_tmp - d66) +
           d67) +
          d26) *
             e_Y44_tmp / 160000.0) -
        d3) +
       d18) +
      0.375 * thetadd[0];
  d73 = 80000.0 * thetad[0] * thetad[3];
  d74 = d73 * f_Y44_tmp;
  d75 = 320000.0 * thetad[0] * thetad[1];
  d76 = 160000.0 * thetad[3] * thetad[1];
  d77 = d12 - d16;
  d78 = -160000.0 * thetad[0] * thetad[2];
  d79 = 320000.0 * f_Y44_tmp_tmp * thetad[3] * thetad[1];
  d80 = 320000.0 * thetad[3] * thetad[2];
  d81 = -320000.0 * thetad[3] * thetad[0];
  d82 = d81 * c_Y44_tmp_tmp;
  dv[164] =
      ((((((((d74 + 40000.0 * thetadd[0] * e_Y44_tmp) * b_Y44_tmp -
             d4 * e_Y44_tmp * c_Y44_tmp) +
            (d75 * c_Y44_tmp_tmp + 240000.0 * thetad[0] * thetad[3]) *
                f_Y44_tmp) +
           120000.0 * thetadd[0] * e_Y44_tmp) *
              Y44_tmp / 160000.0 +
          ((-80000.0 * thetad[0] * thetad[1] * e_Y44_tmp * d_Y44_tmp - d74) +
           (d7 - 40000.0 * thetadd[0]) * e_Y44_tmp) *
              b_Y44_tmp / 160000.0) +
         (d76 * d_Y44_tmp_tmp * f_Y44_tmp + d9 * e_Y44_tmp) * c_Y44_tmp /
             160000.0) +
        ((d78 * e_Y44_tmp_tmp + d19) * f_Y44_tmp +
         (d82 - 240000.0 * thetad[1] * thetad[0]) * e_Y44_tmp) *
            d_Y44_tmp / 160000.0) +
       ((((d77 * e_Y44_tmp_tmp + d20) * d_Y44_tmp_tmp -
          d34 * thetadd[1] * e_Y44_tmp_tmp) +
         d80 * f_Y44_tmp_tmp) +
        d73) *
           f_Y44_tmp / 160000.0) +
      ((((-320000.0 * thetad[3] * thetad[2] * c_Y44_tmp_tmp - d6) *
             d_Y44_tmp_tmp +
         d79 * e_Y44_tmp_tmp) +
        160000.0 * thetadd[2] * f_Y44_tmp_tmp) +
       40000.0 * thetadd[0]) *
          e_Y44_tmp / 160000.0;
  d7 = 80000.0 * thetadd[0] * b_Y44_tmp_tmp;
  d9 = 80000.0 * thetad[3] * thetad[0];
  d19 = d9 * Y44_tmp_tmp;
  d20 = 160000.0 * d29;
  d73 = -160000.0 * thetad[0] * thetad[1];
  d74 = d75 * e_Y44_tmp_tmp;
  d5 = -160000.0 * thetadd[0] * e_Y44_tmp_tmp - d5;
  d75 = (-160000.0 * thetadd[2] * e_Y44_tmp_tmp + (d77 + d20) * c_Y44_tmp_tmp) +
        d76;
  d77 = -160000.0 * f_Y44_tmp_tmp * thetadd[1] * c_Y44_tmp_tmp -
        160000.0 * thetadd[3] * f_Y44_tmp_tmp;
  d83 = d20 * f_Y44_tmp_tmp;
  d84 = d13 * b_Y44_tmp_tmp;
  d85 = d76 * Y44_tmp_tmp;
  d86 = -160000.0 * thetadd[1] * b_Y44_tmp_tmp;
  dv[168] =
      (((((((d11 * b_Y44_tmp_tmp * b_Y44_tmp + (d7 - d19) * c_Y44_tmp) -
            d74 * Y44_tmp_tmp) *
               Y44_tmp / 160000.0 +
           ((d86 + d85) * d_Y44_tmp_tmp +
            (d17 * f_Y44_tmp_tmp - d11) * b_Y44_tmp_tmp) *
               b_Y44_tmp / 160000.0) +
          (((d73 * b_Y44_tmp_tmp * d_Y44_tmp + d84 * d_Y44_tmp_tmp) - d7) +
           d19) *
              c_Y44_tmp / 160000.0) +
         (-160000.0 * thetad[0] * thetad[3] * e_Y44_tmp_tmp * b_Y44_tmp_tmp +
          d5 * Y44_tmp_tmp) *
             d_Y44_tmp / 160000.0) +
        (-160000.0 * thetadd[3] * c_Y44_tmp_tmp * b_Y44_tmp_tmp +
         d75 * Y44_tmp_tmp) *
            d_Y44_tmp_tmp / 160000.0) +
       (-320000.0 * f_Y44_tmp_tmp * thetad[3] * thetad[1] * c_Y44_tmp_tmp -
        d83) *
           b_Y44_tmp_tmp / 160000.0) +
      d77 * Y44_tmp_tmp / 160000.0;
  d7 = -160000.0 * thetad[3] * thetad[0];
  d17 = d7 * c_Y44_tmp_tmp;
  dv[172] =
      (((((((((((((20000.0 * thetadd[0] * f_Y44_tmp - d27) -
                  20000.0 * thetadd[0]) *
                     b_Y44_tmp +
                 (-40000.0 * thetad[0] * thetad[2] * f_Y44_tmp + d25) *
                     c_Y44_tmp) +
                d37) +
               (d73 * c_Y44_tmp_tmp - d40) * e_Y44_tmp) +
              20000.0 * thetadd[0]) *
                 Y44_tmp / 160000.0 +
             ((((d42 * (246.91358024691357 * f_Y44_tmp - 246.91358024691357) *
                     d_Y44_tmp +
                 (d43 - 20000.0 * thetadd[0]) * f_Y44_tmp) +
                d27) -
               d43) +
              20000.0 * thetadd[0]) *
                 b_Y44_tmp / 160000.0) +
            ((((((d44 + d31) + d25) * f_Y44_tmp - d47) - d44) - d31) - d25) *
                c_Y44_tmp / 160000.0) +
           (((d17 - d57) * f_Y44_tmp + (d4 * e_Y44_tmp_tmp - d58) * e_Y44_tmp) -
            d59) *
               d_Y44_tmp / 160000.0) +
          ((((-160000.0 * thetad[3] * thetad[2] * c_Y44_tmp_tmp - d32) *
                 d_Y44_tmp_tmp +
             d60) +
            d61) +
           20000.0 * thetadd[0]) *
              f_Y44_tmp / 160000.0) +
         ((((-80000.0 * d62 * d63 * e_Y44_tmp_tmp - d64) * d_Y44_tmp_tmp +
            d66) -
           d67) -
          d26) *
             e_Y44_tmp / 160000.0) -
        d3) +
       d18) +
      0.375 * thetadd[0];
  d3 = 80000.0 * thetadd[0] * Y44_tmp_tmp;
  d18 = d13 * Y44_tmp_tmp;
  d19 = d76 * b_Y44_tmp_tmp;
  d25 = 160000.0 * thetadd[1] * Y44_tmp_tmp;
  dv[176] = (((((((d78 * Y44_tmp_tmp * b_Y44_tmp +
                   (-80000.0 * thetad[3] * thetad[0] * b_Y44_tmp_tmp - d3) *
                       c_Y44_tmp) -
                  d74 * b_Y44_tmp_tmp) *
                     Y44_tmp / 160000.0 +
                 ((d19 + d25) * d_Y44_tmp_tmp + (d14 + d11) * Y44_tmp_tmp) *
                     b_Y44_tmp / 160000.0) +
                (((d68 * Y44_tmp_tmp * d_Y44_tmp - d18 * d_Y44_tmp_tmp) +
                  d9 * b_Y44_tmp_tmp) +
                 d3) *
                    c_Y44_tmp / 160000.0) +
               (d5 * b_Y44_tmp_tmp + 160000.0 * thetad[0] * thetad[3] *
                                         e_Y44_tmp_tmp * Y44_tmp_tmp) *
                   d_Y44_tmp / 160000.0) +
              (d75 * b_Y44_tmp_tmp +
               160000.0 * thetadd[3] * c_Y44_tmp_tmp * Y44_tmp_tmp) *
                  d_Y44_tmp_tmp / 160000.0) +
             d77 * b_Y44_tmp_tmp / 160000.0) +
            (d79 * c_Y44_tmp_tmp + d83) * Y44_tmp_tmp / 160000.0;
  dv[180] = ((d + (160000.0 * thetadd[3] * e_Y44_tmp_tmp + d69) *
                      d_Y44_tmp_tmp / 160000.0) +
             f_Y44_tmp_tmp * thetad[3] * thetad[1] * e_Y44_tmp_tmp) +
            thetadd[0] / 4.0;
  dv[184] = 0.0;
  dv[188] = 0.0;
  dv[49] = 0.0;
  d = -f_Y44_tmp_tmp * 9.81;
  dv[53] = d;
  dv[57] = d_Y44_tmp_tmp * 9.81;
  dv[61] = 0.0;
  d3 = thetad[0] * thetad[0];
  dv[65] = -d3 * d_Y44_tmp / 2.0;
  dv[69] = -d3 * Y44_tmp;
  dv[73] = -d_Y44_tmp_tmp * thetadd[0];
  d5 = d3 * d_Y44_tmp;
  d14 = d5 / 2.0;
  dv[77] = d14;
  dv[81] = -f_Y44_tmp_tmp * thetadd[0];
  dv[85] = thetadd[1];
  dv[89] = thetad[1];
  dv[93] = std::tanh(100.0 * thetad[1]);
  dv[97] = 0.0;
  dv[101] = d * c_Y44_tmp_tmp;
  dv[105] = f_Y44_tmp_tmp * 9.81 * e_Y44_tmp_tmp;
  dv[109] = -d_Y44_tmp_tmp * 9.81;
  d = d11 * d_Y44_tmp_tmp;
  d5 /= 4.0;
  d11 = -40000.0 * d3;
  dv[113] =
      ((((d11 * d_Y44_tmp - d) - 80000.0 * thetadd[1]) * b_Y44_tmp / 160000.0 +
        (-80000.0 * d_Y44_tmp_tmp * thetadd[0] + d6) * c_Y44_tmp / 160000.0) -
       d5) +
      thetadd[1] / 2.0;
  d26 = 320000.0 * thetad[0] * thetad[2];
  d27 = 80000.0 * d3;
  dv[117] =
      (-160000.0 * d_Y44_tmp_tmp * thetadd[0] + d13) * b_Y44_tmp / 160000.0 +
      ((d27 * d_Y44_tmp + d26 * d_Y44_tmp_tmp) + 160000.0 * thetadd[1]) *
          c_Y44_tmp / 160000.0;
  d13 = d34 * thetadd[0];
  d31 = d13 + 160000.0 * thetadd[2];
  d34 = d26 * f_Y44_tmp_tmp + d16;
  dv[121] = (d3 * c_Y44_tmp_tmp * Y44_tmp + d34 * c_Y44_tmp_tmp / 160000.0) +
            d31 * e_Y44_tmp_tmp / 160000.0;
  d37 = 40000.0 * d3;
  d6 = 80000.0 * d_Y44_tmp_tmp * thetadd[0] - d6;
  d5 = (((d37 * d_Y44_tmp + d) + 80000.0 * thetadd[1]) * b_Y44_tmp / 160000.0 +
        d6 * c_Y44_tmp / 160000.0) -
       d5;
  dv[125] = d5 + thetadd[1] / 2.0;
  d40 = -d3 * e_Y44_tmp_tmp;
  dv[129] = (d40 * Y44_tmp + d31 * c_Y44_tmp_tmp / 160000.0) +
            (-320000.0 * thetad[0] * thetad[2] * f_Y44_tmp_tmp - d16) *
                e_Y44_tmp_tmp / 160000.0;
  d16 = thetad[0] * thetad[2];
  d31 = d16 * d_Y44_tmp_tmp;
  dv[133] = d14 + d31;
  dv[137] = 0.0;
  dv[141] = 0.0;
  d14 = 324.0 * thetad[0] * thetad[2];
  d42 = 81.0 * d3;
  dv[145] =
      ((((((((d42 * d_Y44_tmp + d14 * d_Y44_tmp_tmp) + 162.0 * thetadd[1]) *
                b_Y44_tmp / 160000.0 +
            (162.0 * d_Y44_tmp_tmp * thetadd[0] - d21) * c_Y44_tmp / 160000.0) -
           0.15074375 * d3 * d_Y44_tmp) -
          0.02475 * d3 * c_Y44_tmp_tmp * Y44_tmp) +
         (d14 - 863280.0) * d_Y44_tmp_tmp / 160000.0) +
        ((-7920.0 * thetad[0] * thetad[2] - 70632.0) * f_Y44_tmp_tmp - d22) *
            c_Y44_tmp_tmp / 160000.0) +
       (-3960.0 * f_Y44_tmp_tmp * thetadd[0] - 3960.0 * thetadd[2]) *
           e_Y44_tmp_tmp / 160000.0) +
      0.3035125 * thetadd[1];
  d14 = 14400.0 * thetad[0] * thetad[2];
  d21 = d14 * b_Y44_tmp_tmp;
  d22 = 0.55 * d3;
  d35 = ((d14 * f_Y44_tmp_tmp + d35) - d36) - 88000.0 * thetadd[3];
  d36 = ((176000.0 * thetad[0] * thetad[3] + 7200.0 * thetadd[0]) *
             f_Y44_tmp_tmp +
         176000.0 * thetad[3] * thetad[2]) +
        7200.0 * thetadd[2];
  d43 = d54 - 176000.0 * thetadd[1];
  d44 = -88000.0 * f_Y44_tmp_tmp * thetadd[0] - 88000.0 * thetadd[2];
  d47 = 3600.0 * d3;
  d54 = d47 * b_Y44_tmp_tmp;
  dv[149] =
      ((((((((((d54 * d_Y44_tmp + d21 * d_Y44_tmp_tmp) + d48) - d50) *
                 b_Y44_tmp / 160000.0 +
             ((7200.0 * thetadd[0] * b_Y44_tmp_tmp -
               7200.0 * thetad[3] * thetad[0] * Y44_tmp_tmp) *
                  d_Y44_tmp_tmp -
              d51) *
                 c_Y44_tmp / 160000.0) -
            d3 * (-0.045 * b_Y44_tmp_tmp - 1.1 * Y44_tmp_tmp) * d_Y44_tmp /
                2.0) +
           d22 * (-b_Y44_tmp_tmp + 0.081818181818181818 * Y44_tmp_tmp) *
               c_Y44_tmp_tmp * Y44_tmp) +
          (d21 + 1.5696E+6 * Y44_tmp_tmp) * d_Y44_tmp_tmp / 160000.0) +
         (((((-176000.0 * thetad[0] * thetad[2] - 1.5696E+6) * f_Y44_tmp_tmp -
             d38) -
            d39) +
           7200.0 * thetadd[3]) *
              b_Y44_tmp_tmp +
          d35 * Y44_tmp_tmp) *
             c_Y44_tmp_tmp / 160000.0) +
        (d44 * b_Y44_tmp_tmp + d36 * Y44_tmp_tmp) * e_Y44_tmp_tmp / 160000.0) +
       (-176000.0 * thetad[3] * thetad[1] + 7200.0 * thetadd[1]) *
           b_Y44_tmp_tmp / 160000.0) +
      d43 * Y44_tmp_tmp / 160000.0;
  d21 = d14 * Y44_tmp_tmp;
  d50 = -3600.0 * d3;
  d38 += (176000.0 * thetad[0] * thetad[2] + 1.5696E+6) * f_Y44_tmp_tmp;
  d51 = 7200.0 * thetadd[0] * Y44_tmp_tmp;
  d57 = d50 * Y44_tmp_tmp;
  dv[153] =
      ((((((((((d57 * d_Y44_tmp - d21 * d_Y44_tmp_tmp) - d49 * b_Y44_tmp_tmp) -
              d55) *
                 b_Y44_tmp / 160000.0 +
             ((-7200.0 * thetad[3] * thetad[0] * b_Y44_tmp_tmp - d51) *
                  d_Y44_tmp_tmp +
              d56) *
                 c_Y44_tmp / 160000.0) -
            d3 * (-1.1 * b_Y44_tmp_tmp + 0.045 * Y44_tmp_tmp) * d_Y44_tmp /
                2.0) +
           d22 * (0.081818181818181818 * b_Y44_tmp_tmp + Y44_tmp_tmp) *
               c_Y44_tmp_tmp * Y44_tmp) +
          (1.5696E+6 * b_Y44_tmp_tmp - d21) * d_Y44_tmp_tmp / 160000.0) +
         (d35 * b_Y44_tmp_tmp +
          ((d38 + d39) - 7200.0 * thetadd[3]) * Y44_tmp_tmp) *
             c_Y44_tmp_tmp / 160000.0) +
        (d36 * b_Y44_tmp_tmp +
         (88000.0 * f_Y44_tmp_tmp * thetadd[0] + 88000.0 * thetadd[2]) *
             Y44_tmp_tmp) *
            e_Y44_tmp_tmp / 160000.0) +
       d43 * b_Y44_tmp_tmp / 160000.0) +
      (176000.0 * thetad[3] * thetad[1] - 7200.0 * thetadd[1]) * Y44_tmp_tmp /
          160000.0;
  dv[157] =
      ((((7200.0 * d_Y44_tmp_tmp * thetadd[0] - d28) * b_Y44_tmp / 160000.0 +
         ((d50 * d_Y44_tmp - d14 * d_Y44_tmp_tmp) - 7200.0 * thetadd[1]) *
             c_Y44_tmp / 160000.0) +
        d22 * e_Y44_tmp_tmp * Y44_tmp) +
       d44 * c_Y44_tmp_tmp / 160000.0) +
      d38 * e_Y44_tmp_tmp / 160000.0;
  d14 = d70 * d_Y44_tmp_tmp;
  d21 = f_Y44_tmp_tmp * thetad[0];
  d22 = d21 + thetad[2] / 2.0;
  d4 *= d_Y44_tmp_tmp;
  d28 = d46 * e_Y44_tmp;
  d35 = 40000.0 * d_Y44_tmp_tmp * thetadd[0];
  d9 = d9 * d_Y44_tmp_tmp * e_Y44_tmp;
  d36 = 160000.0 * d22 * thetad[2] * c_Y44_tmp_tmp;
  d31 /= 2.0;
  d38 = -80000.0 * d3;
  d7 = (d7 * f_Y44_tmp_tmp - d33) * e_Y44_tmp_tmp;
  dv[161] =
      (((((((((d11 * (f_Y44_tmp / 2.0 + 0.5) * d_Y44_tmp +
               (d14 - 40000.0 * thetadd[1]) * f_Y44_tmp) +
              d28) -
             d4) -
            40000.0 * thetadd[1]) *
               b_Y44_tmp / 160000.0 +
           ((((-40000.0 * d_Y44_tmp_tmp * thetadd[0] + d32) * f_Y44_tmp + d9) -
             d35) +
            d32) *
               c_Y44_tmp / 160000.0) -
          d3 * (0.75 * f_Y44_tmp - 0.25) * d_Y44_tmp / 2.0) +
         ((d14 + d7) + 40000.0 * thetadd[1]) * f_Y44_tmp / 160000.0) +
        (((d38 * c_Y44_tmp_tmp * Y44_tmp - d36) +
          (-80000.0 * f_Y44_tmp_tmp * thetadd[0] - 80000.0 * thetadd[2]) *
              e_Y44_tmp_tmp) -
         d46) *
            e_Y44_tmp / 160000.0) +
       d31) +
      thetadd[1] / 4.0;
  d14 = -160000.0 * f_Y44_tmp_tmp * thetadd[0] - 160000.0 * thetadd[2];
  d39 = 320000.0 * thetad[3] * thetad[0];
  d43 = d39 * f_Y44_tmp_tmp;
  dv[165] = (((((d37 * e_Y44_tmp * d_Y44_tmp + d76 * f_Y44_tmp) +
                (d + 80000.0 * thetadd[1]) * e_Y44_tmp) *
                   b_Y44_tmp / 160000.0 +
               (d71 * d_Y44_tmp_tmp * f_Y44_tmp + d6 * e_Y44_tmp) * c_Y44_tmp /
                   160000.0) +
              0.75 * d3 * e_Y44_tmp * d_Y44_tmp) +
             (((-160000.0 * d3 * c_Y44_tmp_tmp * Y44_tmp -
                320000.0 * d22 * thetad[2] * c_Y44_tmp_tmp) +
               d14 * e_Y44_tmp_tmp) -
              d76) *
                 f_Y44_tmp / 160000.0) +
            ((d + (d43 + d80) * e_Y44_tmp_tmp) - 80000.0 * thetadd[1]) *
                e_Y44_tmp / 160000.0;
  d = thetad[0] * thetad[3];
  d6 = d34 - d20;
  d20 = -160000.0 * thetadd[0] * b_Y44_tmp_tmp;
  d22 = d3 * Y44_tmp_tmp * e_Y44_tmp_tmp;
  dv[169] = ((((((d20 + d71 * Y44_tmp_tmp) * d_Y44_tmp_tmp + d84) * b_Y44_tmp /
                    160000.0 +
                (((d27 * b_Y44_tmp_tmp * d_Y44_tmp +
                   d26 * b_Y44_tmp_tmp * d_Y44_tmp_tmp) +
                  160000.0 * thetadd[1] * b_Y44_tmp_tmp) -
                 d85) *
                    c_Y44_tmp / 160000.0) +
               d22 * Y44_tmp) -
              d * Y44_tmp_tmp * d_Y44_tmp_tmp) +
             d14 * Y44_tmp_tmp * c_Y44_tmp_tmp / 160000.0) +
            (160000.0 * thetadd[3] * b_Y44_tmp_tmp + d6 * Y44_tmp_tmp) *
                e_Y44_tmp_tmp / 160000.0;
  d34 = d65 * thetadd[0];
  d44 = d71 * f_Y44_tmp_tmp;
  dv[173] = (((((((((d11 * (-f_Y44_tmp / 2.0 + 0.5) * d_Y44_tmp +
                     (d4 + 40000.0 * thetadd[1]) * f_Y44_tmp) -
                    d28) -
                   d4) -
                  40000.0 * thetadd[1]) *
                     b_Y44_tmp / 160000.0 +
                 ((((d35 - d32) * f_Y44_tmp - d9) - d35) + d32) * c_Y44_tmp /
                     160000.0) -
                d3 * (-0.75 * f_Y44_tmp - 0.25) * d_Y44_tmp / 2.0) +
               ((d4 + (d44 + d33) * e_Y44_tmp_tmp) - 40000.0 * thetadd[1]) *
                   f_Y44_tmp / 160000.0) +
              (((d27 * c_Y44_tmp_tmp * Y44_tmp + d36) +
                (d34 + 80000.0 * thetadd[2]) * e_Y44_tmp_tmp) +
               d46) *
                  e_Y44_tmp / 160000.0) +
             d31) +
            thetadd[1] / 4.0;
  d4 = 160000.0 * thetadd[0] * Y44_tmp_tmp;
  d9 = d3 * b_Y44_tmp_tmp * e_Y44_tmp_tmp;
  dv[177] = ((((((d71 * b_Y44_tmp_tmp + d4) * d_Y44_tmp_tmp - d18) * b_Y44_tmp /
                    160000.0 +
                (((d38 * Y44_tmp_tmp * d_Y44_tmp -
                   d26 * Y44_tmp_tmp * d_Y44_tmp_tmp) -
                  d19) -
                 d25) *
                    c_Y44_tmp / 160000.0) +
               d9 * Y44_tmp) -
              d * b_Y44_tmp_tmp * d_Y44_tmp_tmp) +
             d14 * b_Y44_tmp_tmp * c_Y44_tmp_tmp / 160000.0) +
            (d6 * b_Y44_tmp_tmp - 160000.0 * thetadd[3] * Y44_tmp_tmp) *
                e_Y44_tmp_tmp / 160000.0;
  dv[181] =
      ((d5 + thetadd[3] * c_Y44_tmp_tmp) + d7 / 160000.0) + thetadd[1] / 2.0;
  dv[185] = 0.0;
  dv[189] = 0.0;
  dv[98] = 0.0;
  d5 = e_Y44_tmp_tmp * d_Y44_tmp_tmp;
  dv[102] = d5 * 9.81;
  d6 = c_Y44_tmp_tmp * d_Y44_tmp_tmp;
  dv[106] = d6 * 9.81;
  dv[110] = 0.0;
  d7 = d_Y44_tmp_tmp * thetad[0] * thetad[1];
  d14 = d7 * b_Y44_tmp;
  dv[114] = ((d11 * Y44_tmp + d37) - d8) * c_Y44_tmp / 160000.0 + d14;
  dv[118] = -2.0 * thetad[0] * d_Y44_tmp_tmp * thetad[1] * c_Y44_tmp +
            ((d38 * Y44_tmp + d27) - d12) * b_Y44_tmp / 160000.0;
  d12 = 2.0 * f_Y44_tmp_tmp * thetad[0] * thetad[1];
  dv[122] =
      ((d40 * d_Y44_tmp / 2.0 - d6 * thetadd[0]) + e_Y44_tmp_tmp * thetadd[1]) -
      d12 * c_Y44_tmp_tmp;
  d6 = ((d37 * Y44_tmp - d37) + d8) * c_Y44_tmp / 160000.0 - d14;
  dv[126] = d6;
  d8 = c_Y44_tmp_tmp * thetadd[1];
  dv[130] = ((-d3 * c_Y44_tmp_tmp * d_Y44_tmp / 2.0 + d5 * thetadd[0]) +
             d12 * e_Y44_tmp_tmp) +
            d8;
  d5 = f_Y44_tmp_tmp * thetadd[0];
  dv[134] = (d5 - d7) + thetadd[2];
  dv[138] = thetad[2];
  dv[142] = std::tanh(100.0 * thetad[2]);
  dv[146] =
      ((((((((d42 * Y44_tmp - d42) + d23) * c_Y44_tmp / 160000.0 -
            0.002025 * d_Y44_tmp_tmp * thetad[0] * thetad[1] * b_Y44_tmp) +
           0.012375 * d3 * e_Y44_tmp_tmp * d_Y44_tmp) +
          ((70632.0 * e_Y44_tmp_tmp + d24) - 324.0 * thetad[1] * thetad[0]) *
              d_Y44_tmp_tmp / 160000.0) -
         0.02475 * e_Y44_tmp_tmp * thetadd[1]) +
        0.0495 * f_Y44_tmp_tmp * thetad[0] * thetad[1] * c_Y44_tmp_tmp) +
       0.002025 * f_Y44_tmp_tmp * thetadd[0]) +
      0.002025 * thetadd[2];
  d12 = 2.0 * d1;
  d18 = d3 - d12;
  d19 = 3600.0 * d18;
  d23 = 0.09 * d_Y44_tmp_tmp * thetad[0] * thetad[1];
  d24 = 176000.0 * f_Y44_tmp_tmp * thetad[0] * thetad[1];
  d26 = -14400.0 * thetad[3] * thetad[0] * f_Y44_tmp_tmp - d41;
  dv[150] =
      (((((((d54 * Y44_tmp - d19 * b_Y44_tmp_tmp) * c_Y44_tmp / 160000.0 -
            d23 * b_Y44_tmp_tmp * b_Y44_tmp) -
           d3 * (-0.55 * b_Y44_tmp_tmp + 0.045 * Y44_tmp_tmp) * e_Y44_tmp_tmp *
               d_Y44_tmp / 2.0) +
          ((1.5696E+6 * b_Y44_tmp_tmp * e_Y44_tmp_tmp +
            ((-14400.0 * thetad[0] * thetad[3] + 88000.0 * thetadd[0]) *
                 b_Y44_tmp_tmp -
             d51) *
                c_Y44_tmp_tmp) -
           14400.0 * thetad[1] * b_Y44_tmp_tmp * thetad[0]) *
              d_Y44_tmp_tmp / 160000.0) +
         ((14400.0 * thetad[3] * thetad[1] - 88000.0 * thetadd[1]) *
              b_Y44_tmp_tmp +
          d55) *
             e_Y44_tmp_tmp / 160000.0) +
        (d24 * b_Y44_tmp_tmp -
         14400.0 * f_Y44_tmp_tmp * thetad[0] * thetad[1] * Y44_tmp_tmp) *
            c_Y44_tmp_tmp / 160000.0) +
       (d52 * thetadd[0] + 14400.0 * thetadd[2]) * b_Y44_tmp_tmp / 160000.0) +
      d26 * Y44_tmp_tmp / 160000.0;
  dv[154] =
      (((((((d57 * Y44_tmp + d19 * Y44_tmp_tmp) * c_Y44_tmp / 160000.0 +
            d23 * Y44_tmp_tmp * b_Y44_tmp) -
           d3 * (0.045 * b_Y44_tmp_tmp + 0.55 * Y44_tmp_tmp) * e_Y44_tmp_tmp *
               d_Y44_tmp / 2.0) +
          ((-1.5696E+6 * Y44_tmp_tmp * e_Y44_tmp_tmp +
            (-7200.0 * thetadd[0] * b_Y44_tmp_tmp +
             (14400.0 * thetad[0] * thetad[3] - 88000.0 * thetadd[0]) *
                 Y44_tmp_tmp) *
                c_Y44_tmp_tmp) +
           14400.0 * thetad[1] * Y44_tmp_tmp * thetad[0]) *
              d_Y44_tmp_tmp / 160000.0) +
         (d48 + (-14400.0 * thetad[3] * thetad[1] + 88000.0 * thetadd[1]) *
                    Y44_tmp_tmp) *
             e_Y44_tmp_tmp / 160000.0) +
        (-14400.0 * f_Y44_tmp_tmp * thetad[0] * thetad[1] * b_Y44_tmp_tmp -
         d24 * Y44_tmp_tmp) *
            c_Y44_tmp_tmp / 160000.0) +
       d26 * b_Y44_tmp_tmp / 160000.0) +
      (-14400.0 * f_Y44_tmp_tmp * thetadd[0] - 14400.0 * thetadd[2]) *
          Y44_tmp_tmp / 160000.0;
  dv[158] = ((((0.09 * thetad[0] * d_Y44_tmp_tmp * thetad[1] * c_Y44_tmp +
                ((d47 * Y44_tmp - d47) + d45) * b_Y44_tmp / 160000.0) +
               0.275 * d3 * c_Y44_tmp_tmp * d_Y44_tmp) +
              (d2 + 1.5696E+6 * c_Y44_tmp_tmp) * d_Y44_tmp_tmp / 160000.0) -
             1.1 * f_Y44_tmp_tmp * thetad[0] * thetad[1] * e_Y44_tmp_tmp) -
            0.55 * c_Y44_tmp_tmp * thetadd[1];
  d2 = 20000.0 * d3;
  d12 += d3 * Y44_tmp - d3;
  d19 = d2 * Y44_tmp;
  d23 = d76 * e_Y44_tmp_tmp;
  d14 /= 2.0;
  d24 = 80000.0 * c_Y44_tmp_tmp * d_Y44_tmp_tmp * thetadd[0];
  d26 = 80000.0 * e_Y44_tmp_tmp * thetadd[1];
  d28 = 160000.0 * f_Y44_tmp_tmp * thetad[0] * thetad[1] * c_Y44_tmp_tmp;
  d7 /= 2.0;
  d5 /= 2.0;
  dv[162] =
      ((((((((-20000.0 * d12 * f_Y44_tmp - d19) + d2) - d30) * c_Y44_tmp /
               160000.0 +
           ((((80000.0 * d_Y44_tmp_tmp * thetad[0] * thetad[1] * b_Y44_tmp +
               (d72 + d10) * d_Y44_tmp_tmp) -
              d23) -
             d34) -
            80000.0 * thetadd[2]) *
               f_Y44_tmp / 160000.0) +
          d14) +
         (((((d37 * e_Y44_tmp_tmp * d_Y44_tmp + d24) - d26) + d28) + d44) +
          d33) *
             e_Y44_tmp / 160000.0) -
        d7) +
       d5) +
      thetadd[2] / 2.0;
  d31 = 320000.0 * thetad[3] * thetad[1];
  dv[166] =
      ((d12 * e_Y44_tmp * c_Y44_tmp / 4.0 +
        (((((d27 * e_Y44_tmp_tmp * d_Y44_tmp +
             160000.0 * c_Y44_tmp_tmp * d_Y44_tmp_tmp * thetadd[0]) -
            160000.0 * e_Y44_tmp_tmp * thetadd[1]) +
           320000.0 * f_Y44_tmp_tmp * thetad[0] * thetad[1] * c_Y44_tmp_tmp) +
          d43) +
         d80) *
            f_Y44_tmp / 160000.0) -
       e_Y44_tmp * d_Y44_tmp_tmp * thetad[0] * thetad[1] * b_Y44_tmp) +
      ((((d82 - 160000.0 * thetad[1] * thetad[0]) * d_Y44_tmp_tmp +
         d31 * e_Y44_tmp_tmp) +
        d13) +
       160000.0 * thetadd[2]) *
          e_Y44_tmp / 160000.0;
  d13 = 80000.0 * d18;
  dv[170] = ((((((-2.0 * d_Y44_tmp_tmp * thetad[0] * thetad[1] * b_Y44_tmp_tmp *
                      c_Y44_tmp +
                  (d38 * b_Y44_tmp_tmp * Y44_tmp + d13 * b_Y44_tmp_tmp) *
                      b_Y44_tmp / 160000.0) +
                 Y44_tmp_tmp * d_Y44_tmp * c_Y44_tmp_tmp * d3 / 2.0) +
                (d81 * b_Y44_tmp_tmp - d4) * e_Y44_tmp_tmp * d_Y44_tmp_tmp /
                    160000.0) -
               2.0 * Y44_tmp_tmp * f_Y44_tmp_tmp * e_Y44_tmp_tmp * thetad[0] *
                   thetad[1]) +
              (-320000.0 * thetad[3] * thetad[1] * b_Y44_tmp_tmp - d25) *
                  c_Y44_tmp_tmp / 160000.0) -
             d29 * b_Y44_tmp_tmp) -
            thetadd[3] * Y44_tmp_tmp;
  dv[174] =
      ((((((((20000.0 * d12 * f_Y44_tmp - d19) + d2) - d30) * c_Y44_tmp /
               160000.0 +
           ((((-80000.0 * d_Y44_tmp_tmp * thetad[0] * thetad[1] * b_Y44_tmp +
               (d17 - d10) * d_Y44_tmp_tmp) +
              d23) +
             d34) +
            80000.0 * thetadd[2]) *
               f_Y44_tmp / 160000.0) +
          d14) +
         (((((d11 * e_Y44_tmp_tmp * d_Y44_tmp - d24) + d26) - d28) - d44) -
          d33) *
             e_Y44_tmp / 160000.0) -
        d7) +
       d5) +
      thetadd[2] / 2.0;
  dv[178] = ((((((2.0 * d_Y44_tmp_tmp * thetad[0] * thetad[1] * Y44_tmp_tmp *
                      c_Y44_tmp +
                  (d27 * Y44_tmp_tmp * Y44_tmp - d13 * Y44_tmp_tmp) *
                      b_Y44_tmp / 160000.0) +
                 b_Y44_tmp_tmp * d_Y44_tmp * c_Y44_tmp_tmp * d3 / 2.0) +
                (d20 + d39 * Y44_tmp_tmp) * e_Y44_tmp_tmp * d_Y44_tmp_tmp /
                    160000.0) -
               2.0 * b_Y44_tmp_tmp * f_Y44_tmp_tmp * e_Y44_tmp_tmp * thetad[0] *
                   thetad[1]) +
              (d86 + d31 * Y44_tmp_tmp) * c_Y44_tmp_tmp / 160000.0) -
             thetadd[3] * b_Y44_tmp_tmp) +
            d29 * Y44_tmp_tmp;
  dv[182] = (d6 - d * c_Y44_tmp_tmp * d_Y44_tmp_tmp) + d53 * e_Y44_tmp_tmp;
  dv[186] = 0.0;
  dv[190] = 0.0;
  for (int i{0}; i < 12; i++) {
    i1 = i << 2;
    dv[i1 + 1] = 0.0;
    dv[i1 + 2] = 0.0;
    i2 = (i + 12) << 2;
    dv[i2 + 2] = 0.0;
    dv[i1 + 3] = 0.0;
    dv[i2 + 3] = 0.0;
    dv[((i + 24) << 2) + 3] = 0.0;
  }
  dv[147] = 0.0;
  d = 72.0 * thetad[0] * thetad[2];
  d2 = 9.0 * d3;
  d4 = 18.0 * d1;
  d5 = 0.045 * d_Y44_tmp_tmp * thetad[0] * thetad[1];
  d6 = 0.0225 * d3;
  d7 = ((((((7848.0 * c_Y44_tmp_tmp - 440.0 * thetadd[0] * e_Y44_tmp_tmp) *
                d_Y44_tmp_tmp -
            440.0 * c_Y44_tmp_tmp * thetadd[1]) -
           880.0 * f_Y44_tmp_tmp * thetad[0] * thetad[1] * e_Y44_tmp_tmp) +
          d * f_Y44_tmp_tmp) +
         27.0 * d3) +
        d4) +
       36.0 * d15;
  d10 = 36.0 * thetadd[0] * e_Y44_tmp_tmp;
  d11 = 36.0 * c_Y44_tmp_tmp * thetadd[1];
  d13 = 72.0 * thetad[2] * thetad[1] * e_Y44_tmp_tmp;
  d14 = 800.0 * f_Y44_tmp_tmp * 9.81;
  d17 = 220.0 * d3;
  d19 = 440.0 * d1;
  dv[151] =
      (((((d2 * Y44_tmp_tmp * Y44_tmp - 9.0 * d18 * Y44_tmp_tmp) * b_Y44_tmp /
              800.0 -
          0.275 * (b_Y44_tmp_tmp - 0.040909090909090909 * Y44_tmp_tmp) * d3 *
              Y44_tmp) +
         d5 * Y44_tmp_tmp * c_Y44_tmp) +
        d6 *
            (b_Y44_tmp_tmp * c_Y44_tmp_tmp +
             12.222222222222221 * Y44_tmp_tmp * c_Y44_tmp_tmp) *
            d_Y44_tmp) +
       ((((((d * c_Y44_tmp_tmp + d10) * d_Y44_tmp_tmp + d11) - d13) - d14) +
         d17) +
        d19) *
           b_Y44_tmp_tmp / 800.0) +
      d7 * Y44_tmp_tmp / 800.0;
  dv[155] =
      (((((d2 * b_Y44_tmp_tmp * Y44_tmp + (-9.0 * d3 + d4) * b_Y44_tmp_tmp) *
              b_Y44_tmp / 800.0 -
          0.275 * (-0.040909090909090909 * b_Y44_tmp_tmp - Y44_tmp_tmp) * d3 *
              Y44_tmp) +
         d5 * b_Y44_tmp_tmp * c_Y44_tmp) +
        d6 *
            (12.222222222222221 * b_Y44_tmp_tmp * c_Y44_tmp_tmp -
             Y44_tmp_tmp * c_Y44_tmp_tmp) *
            d_Y44_tmp) +
       d7 * b_Y44_tmp_tmp / 800.0) +
      ((((((-72.0 * thetad[0] * thetad[2] * c_Y44_tmp_tmp - d10) *
               d_Y44_tmp_tmp -
           d11) +
          d13) +
         d14) -
        d17) -
       d19) *
          Y44_tmp_tmp / 800.0;
  dv[159] = 0.0;
  d = d21 + thetad[2];
  d2 = 300.0 * d3 * Y44_tmp;
  d4 = 400.0 * thetad[0] * d_Y44_tmp_tmp * thetad[1] * c_Y44_tmp;
  d5 = 800.0 * thetad[0] * thetad[2] * f_Y44_tmp_tmp;
  d6 = 100.0 * d3;
  d7 = 200.0 * d1;
  d10 = 400.0 * d15;
  d11 = 800.0 * thetad[2] * thetad[0] * c_Y44_tmp_tmp * d_Y44_tmp_tmp;
  d13 = 800.0 * d * thetad[1] * e_Y44_tmp_tmp;
  dv[163] =
      ((((((-100.0 * d12 * b_Y44_tmp - d2) - d4) - d5) - d6) + d7) - d10) *
          e_Y44_tmp / 800.0 +
      ((-400.0 * d3 * c_Y44_tmp_tmp * d_Y44_tmp - d11) + d13) * f_Y44_tmp /
          800.0;
  d14 = 1600.0 * thetad[0] * thetad[2];
  d1 *= 400.0;
  dv[167] = (((800.0 * d3 * c_Y44_tmp_tmp * d_Y44_tmp +
               1600.0 * thetad[2] * thetad[0] * c_Y44_tmp_tmp * d_Y44_tmp_tmp) -
              1600.0 * d * thetad[1] * e_Y44_tmp_tmp) *
                 e_Y44_tmp / 800.0 -
             d12 * f_Y44_tmp * b_Y44_tmp / 4.0) +
            (((((-600.0 * d3 * Y44_tmp -
                 800.0 * thetad[0] * d_Y44_tmp_tmp * thetad[1] * c_Y44_tmp) -
                d14 * f_Y44_tmp_tmp) -
               200.0 * d3) +
              d1) -
             800.0 * d15) *
                f_Y44_tmp / 800.0;
  d = -d_Y44_tmp_tmp * thetad[0] * thetad[1];
  d15 = (800.0 * d_Y44_tmp_tmp * thetad[0] * thetad[1] -
         800.0 * f_Y44_tmp_tmp * thetadd[0]) -
        800.0 * thetadd[2];
  d14 *= e_Y44_tmp_tmp;
  d17 = 1600.0 * thetad[2] * thetad[1] * c_Y44_tmp_tmp;
  d19 = 800.0 * e_Y44_tmp_tmp * thetadd[1];
  dv[171] =
      ((((d * Y44_tmp_tmp * b_Y44_tmp +
          Y44_tmp_tmp * c_Y44_tmp * d3 * Y44_tmp / 4.0) -
         d18 * Y44_tmp_tmp * c_Y44_tmp / 4.0) +
        d9 * d_Y44_tmp / 2.0) +
       (((-800.0 * thetadd[0] * c_Y44_tmp_tmp + d14) * d_Y44_tmp_tmp + d17) +
        d19) *
           b_Y44_tmp_tmp / 800.0) +
      d15 * Y44_tmp_tmp / 800.0;
  dv[175] = ((((((100.0 * d12 * b_Y44_tmp + d2) + d4) + d5) + d6) - d7) + d10) *
                e_Y44_tmp / 800.0 +
            ((400.0 * d3 * c_Y44_tmp_tmp * d_Y44_tmp + d11) - d13) * f_Y44_tmp /
                800.0;
  dv[179] =
      ((((d * b_Y44_tmp_tmp * b_Y44_tmp +
          b_Y44_tmp_tmp * c_Y44_tmp * d3 * Y44_tmp / 4.0) +
         (-200.0 * d3 + d1) * b_Y44_tmp_tmp * c_Y44_tmp / 800.0) -
        d22 * d_Y44_tmp / 2.0) +
       d15 * b_Y44_tmp_tmp / 800.0) +
      (((800.0 * thetadd[0] * c_Y44_tmp_tmp - d14) * d_Y44_tmp_tmp - d17) -
       d19) *
          Y44_tmp_tmp / 800.0;
  dv[183] =
      (((d16 * c_Y44_tmp_tmp + thetadd[0] * e_Y44_tmp_tmp) * d_Y44_tmp_tmp +
        thetad[1] * (d21 - thetad[2]) * e_Y44_tmp_tmp) +
       d8) +
      thetadd[3];
  dv[187] = thetad[3];
  dv[191] = std::tanh(100.0 * thetad[3]);
  for (int i{0}; i < 4; i++) {
    for (i1 = 0; i1 < 30; i1++) {
      d = 0.0;
      for (i2 = 0; i2 < 48; i2++) {
        d += dv[i + (i2 << 2)] * static_cast<double>(b[i2 + 48 * i1]);
      }
      Y_b[i + (i1 << 2)] = d;
    }
  }
}

// extern "C" {
//     void calculate_Y(const double theta[4], const double thetad[4],
//                      const double thetadd[4], double Y_b[120]); // Y_b is 4*30 (120 elements)
// }

Eigen::MatrixXd calculate_Y_eigen(const Eigen::VectorXd& theta, 
                                  const Eigen::VectorXd& thetad, 
                                  const Eigen::VectorXd& thetadd) {
    // Ensure input size is 4
    assert(theta.size() == 4 && thetad.size() == 4 && thetadd.size() == 4);

    // Convert Eigen::VectorXd to C-style array
    std::array<double, 4> theta_arr, thetad_arr, thetadd_arr;
    for (int i = 0; i < 4; i++) {
        theta_arr[i] = theta(i);
        thetad_arr[i] = thetad(i);
        thetadd_arr[i] = thetadd(i);
    }

    // Output array (120 elements)
    std::array<double, 120> Y_b_arr;

    // Call the C function
    calculate_Y(theta_arr.data(), thetad_arr.data(), thetadd_arr.data(), Y_b_arr.data());

    // Convert output to Eigen::MatrixXd (4x30 structure)
    Eigen::MatrixXd Y_b = Eigen::Map<Eigen::MatrixXd>(Y_b_arr.data(), 4, 30);

    return Y_b;
}

// End of code generation (calculate_Y.cpp)
