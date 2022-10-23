#include "helper_math.h"
// cgs unit
// deltat should be smaller than 1e-14 when using Euler method. Should be smaller than 1e-12 when using Heun method.
// If not consider thermal effect, change T to 0 and thermal_step to 1;
// When considering thermal effect, change eta and J steps to small numbers, e.g., 10 and 1000, respectively.

void mag_dynamics_Euler(float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot);
void mag_dynamics_Heun(float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot);
void mag_dynamics_Heun_stochastic(float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot);
void mag_dynamics_Heun_singlecase(float3 Mini, float eta, float J, float3 Hexternal, float3 Anisotropy, float* mx, float* my, float* mz);
void mag_dynamics_Heun_singleeta(float3 Mini, float eta, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot);
void mag_dynamics_Heun_risefall(float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot);
#define pi 3.141592653589793
#define damping 0.005
#define Msat 1000
#define anisotropyField (4000)
#define tFM (1*powf(10,-7))
#define gyromagneticRatio (1.76*powf(10,7))
#define e_charge (1.6*powf(10,-19))
#define h_bar (1.054*powf(10,-27))
#define mu0 (4*pi*powf(10,-7))
#define tp (200*powf(10,-9))
#define ttot (240*powf(10,-9))
#define deltat (10*powf(10,-13))
#define deltat2 (10*powf(10,-13)) // for single case calculation GPU and CPU
#define DL 0.3
#define FL 0
#define kB (1.380649*powf(10,-16))
#define T 0
#define Area_cross_section (900*powf(10,-14))
#define thermal_step 1
#define eta_start 0.1
#define eta_end 90
#define eta_step 100
#define J_start (8*powf(10,5))
#define J_end (3*powf(10,8))
#define J_step 1000
#define Hx 0
#define rise_time (0.2*powf(10,-9))
#define fall_time (0.2*powf(10,-9))

#define eta_fixed 60
#define J_fixed (3*powf(10,7))

#define BLOCK_SIZE 1024