#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include "helper_math.h"
#include "header.h"


__global__ void macrospin_kernel_Euler(
    float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    float dt = deltat;
    int num_step_tot = (int)(ttot/dt);
    int num_step_pulse = (int)(tp/dt);
    int num_step_relax = num_step_tot - num_step_pulse;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_eta = i/Jstep;
    int idx_J = i - idx_eta*Jstep;
    //float eta = etastart + idx_eta*(etaend-etastart)/(etastep-1);
    float eta = etastart * expf((float)idx_eta/(etastep-1)*log(etaend/etastart));
    float J = Jstart + idx_J*(Jend-Jstart)/(Jstep-1);
    float alpha = damping;
    float Hk = anisotropyField;
    float Ms = Msat;
    float gamma = gyromagneticRatio;
    float coeff_DLT = 0.5 * h_bar * DL * J /(Ms*tFM*e_charge);
    float coeff_FLT = 0.5 * h_bar * FL * J /(Ms*tFM*e_charge);
    float3 s = make_float3(0, cosf(eta), sinf(eta));
    float3 A = Anisotropy;
    float3 Hext = Hexternal;

    float3 M = Ms*Mini;
    float3 Mnorm;
    float3 Heff;
    float3 T_precession;
    float3 T_damping;
    float3 DLT;
    float3 FLT;
    float3 dM;
    float factor;
    if(i < num_tot){
        for(int t=0; t<num_step_pulse; t++){
            factor = 1;
            factor = ((t*dt)<rise_time) ? ((t*dt)/rise_time) : 1;
            factor = ((tp-t*dt)<fall_time) ? ((tp-t*dt)/fall_time) : factor;
            Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = factor*coeff_DLT*cross(Mnorm,cross(M,s));
            FLT = factor*coeff_FLT*cross(M,s);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M += dM;

        }

        for(int t=0; t<num_step_relax; t++){
            Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M += dM;
        }

        mz[i] = normalize(M).z;
    }
}


__global__ void macrospin_kernel_Heun(
    float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    float dt = deltat;
    int num_step_tot = (int)(ttot/dt);
    int num_step_pulse = (int)(tp/dt);
    int num_step_relax = num_step_tot - num_step_pulse;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_eta = i/Jstep;
    int idx_J = i - idx_eta*Jstep;
    //float eta = etastart + idx_eta*(etaend-etastart)/(etastep-1);
    float eta = etastart * expf((float)idx_eta/(etastep-1)*log(etaend/etastart));
    float J = Jstart + idx_J*(Jend-Jstart)/(Jstep-1);
    float alpha = damping;
    float Hk = anisotropyField;
    float Ms = Msat;
    float gamma = gyromagneticRatio;
    float coeff_DLT = 0.5 * h_bar * DL * J /(Ms*tFM*e_charge);
    float coeff_FLT = 0.5 * h_bar * FL * J /(Ms*tFM*e_charge);
    float3 s = make_float3(0, cosf(eta), sinf(eta));
    float3 A = Anisotropy;
    float3 Hext = Hexternal;

    float3 M = Ms*Mini;
    float3 M_interm;
    float3 Mnorm;
    float3 Heff;
    float3 T_precession;
    float3 T_damping;
    float3 DLT;
    float3 FLT;
    float3 dM;
    float3 dM_interm;
    if(i < num_tot){
        for(int t=0; t<num_step_pulse; t++){
            Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = coeff_DLT*cross(Mnorm,cross(M,s));
            FLT = coeff_FLT*cross(M,s);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = coeff_DLT*cross(Mnorm,cross(M_interm,s));
            FLT = coeff_FLT*cross(M_interm,s);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M += 0.5 * (dM+dM_interm);

        }

        for(int t=0; t<num_step_relax; t++){
            Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M += 0.5 * (dM+dM_interm);
        }

        mz[i] = normalize(M).z;
    }
}


__global__ void macrospin_kernel_Heun_stochastic(
    float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    float dt = deltat;
    int num_step_tot = (int)(ttot/dt);
    int num_step_pulse = (int)(tp/dt);
    int num_step_relax = num_step_tot - num_step_pulse;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_thermal = i/(Jstep*etastep);
    int idx_eta = (i-idx_thermal*Jstep*etastep)/Jstep;
    int idx_J = i-idx_thermal*Jstep*etastep - idx_eta*Jstep;
    //float eta = etastart + idx_eta*(etaend-etastart)/(etastep-1);
    float eta = etastart * expf((float)idx_eta/(etastep-1)*log(etaend/etastart));
    float J = Jstart + idx_J*(Jend-Jstart)/(Jstep-1);
    float alpha = damping;
    float Hk = anisotropyField;
    float Ms = Msat;
    float gamma = gyromagneticRatio;
    float coeff_DLT = 0.5 * h_bar * DL * J /(Ms*tFM*e_charge);
    float coeff_FLT = 0.5 * h_bar * FL * J /(Ms*tFM*e_charge);
    float3 s = make_float3(0, cosf(eta), sinf(eta));
    float3 A = Anisotropy;
    float3 Hext = Hexternal;
    float3 Hthermal;
    float Hthermal_amplitude = sqrtf((2*alpha*kB*T)/(gamma*Ms*Area_cross_section*tFM*dt));
    curandState_t state;

    float3 M = Ms*Mini;
    float3 M_interm;
    float3 Mnorm;
    float3 Heff;
    float3 T_precession;
    float3 T_damping;
    float3 DLT;
    float3 FLT;
    float3 dM;
    float3 dM_interm;
    if(i < num_tot){
        for(int t=0; t<num_step_pulse; t++){
            Mnorm = normalize(M);
            M = Mnorm*Ms;

            curand_init(t*1+i, 0, 0, &state);
            Hthermal.x = Hthermal_amplitude * curand_normal(&state);
            curand_init(t*2+i, 0, 0, &state);
            Hthermal.y = Hthermal_amplitude * curand_normal(&state);
            curand_init(t*3+i, 0, 0, &state);
            Hthermal.z = Hthermal_amplitude * curand_normal(&state);

            Heff = Hk*dot(A,Mnorm)*A + Hext + Hthermal;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = coeff_DLT*cross(Mnorm,cross(M,s));
            FLT = coeff_FLT*cross(M,s);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;

            curand_init((t+1)*1+i, 0, 0, &state);
            Hthermal.x = Hthermal_amplitude * curand_normal(&state);
            curand_init((t+1)*2+i, 0, 0, &state);
            Hthermal.y = Hthermal_amplitude * curand_normal(&state);
            curand_init((t+1)*3+i, 0, 0, &state);
            Hthermal.z = Hthermal_amplitude * curand_normal(&state);

            Heff = Hk*dot(A,Mnorm)*A + Hext + Hthermal;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = coeff_DLT*cross(Mnorm,cross(M_interm,s));
            FLT = coeff_FLT*cross(M_interm,s);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M += 0.5 * (dM+dM_interm);

        }

        for(int t=0; t<num_step_relax; t++){
            Mnorm = normalize(M);
            M = Mnorm*Ms;

            curand_init(t*1+i+num_step_pulse, 0, 0, &state);
            Hthermal.x = Hthermal_amplitude * curand_normal(&state);
            curand_init(t*2+i+num_step_pulse, 0, 0, &state);
            Hthermal.y = Hthermal_amplitude * curand_normal(&state);
            curand_init(t*3+i+num_step_pulse, 0, 0, &state);
            Hthermal.z = Hthermal_amplitude * curand_normal(&state);

            Heff = Hk*dot(A,Mnorm)*A + Hthermal;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;

            curand_init((t+1)*1+i+num_step_pulse, 0, 0, &state);
            Hthermal.x = Hthermal_amplitude * curand_normal(&state);
            curand_init((t+1)*2+i+num_step_pulse, 0, 0, &state);
            Hthermal.y = Hthermal_amplitude * curand_normal(&state);
            curand_init((t+1)*3+i+num_step_pulse, 0, 0, &state);
            Hthermal.z = Hthermal_amplitude * curand_normal(&state);

            Heff = Hk*dot(A,Mnorm)*A + Hthermal;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M += 0.5 * (dM+dM_interm);
        }

        mz[i] = normalize(M).z;
    }
}

__global__ void macrospin_kernel_Heun_singlecase(
    float3 Mini, float eta, float J, float3 Hexternal, float3 Anisotropy, float* mx, float* my, float* mz) {
    float dt = deltat2;
    int num_step_tot = (int)(ttot/dt);
    int num_step_pulse = (int)(tp/dt);
    int num_step_relax = num_step_tot - num_step_pulse;

    float alpha = damping;
    float Hk = anisotropyField;
    float Ms = Msat;
    float gamma = gyromagneticRatio;
    float coeff_DLT = 0.5 * h_bar * DL * J /(Ms*tFM*e_charge);
    float coeff_FLT = 0.5 * h_bar * FL * J /(Ms*tFM*e_charge);
    float3 s = make_float3(0, cosf(eta), sinf(eta));
    float3 A = Anisotropy;
    float3 Hext = Hexternal;

    float3 M = Ms*Mini;
    float3 M_interm;
    float3 Mnorm;
    float3 Heff;
    float3 T_precession;
    float3 T_damping;
    float3 DLT;
    float3 FLT;
    float3 dM;
    float3 dM_interm;
    Mnorm = normalize(M);
    float factor = 1;

    for(int t=0; t<num_step_pulse; t++){
            //Mnorm = normalize(M);
            factor = 1;
            factor = ((t*dt)<rise_time) ? ((t*dt)/rise_time) : 1;
            factor = ((tp-t*dt)<fall_time) ? ((tp-t*dt)/fall_time) : factor;
            //mx[t] = factor;
            //continue;
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = factor*coeff_DLT*cross(Mnorm,cross(M,s));
            FLT = factor*coeff_FLT*cross(M,s);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = factor*coeff_DLT*cross(Mnorm,cross(M_interm,s));
            FLT = factor*coeff_FLT*cross(M_interm,s);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M += 0.5 * (dM+dM_interm);
            Mnorm = normalize(M);
            mx[t] = Mnorm.x;
            my[t] = Mnorm.y;
            mz[t] = Mnorm.z;
    }

    Mnorm = normalize(M);
    for(int t=0; t<num_step_relax; t++){
            //Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M += 0.5 * (dM+dM_interm);
            Mnorm = normalize(M);
            mx[t+num_step_pulse] = Mnorm.x;
            my[t+num_step_pulse] = Mnorm.y;
            mz[t+num_step_pulse] = Mnorm.z;

    }

}

__global__ void macrospin_kernel_Heun_singleeta(
    float3 Mini, float eta, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    float dt = deltat;
    int num_step_tot = (int)(ttot/dt);
    int num_step_pulse = (int)(tp/dt);
    int num_step_relax = num_step_tot - num_step_pulse;

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float J = Jstart + i*(Jend-Jstart)/(Jstep-1);
    float alpha = damping;
    float Hk = anisotropyField;
    float Ms = Msat;
    float gamma = gyromagneticRatio;
    float coeff_DLT = 0.5 * h_bar * DL * J /(Ms*tFM*e_charge);
    float coeff_FLT = 0.5 * h_bar * FL * J /(Ms*tFM*e_charge);
    float3 s = make_float3(0, cosf(eta), sinf(eta));
    float3 A = Anisotropy;
    float3 Hext = Hexternal;

    float3 M = Ms*Mini;
    float3 M_interm;
    float3 Mnorm;
    float3 Heff;
    float3 T_precession;
    float3 T_damping;
    float3 DLT;
    float3 FLT;
    float3 dM;
    float3 dM_interm;
    float factor;
    if(i < num_tot){
        for(int t=0; t<num_step_pulse; t++){
            factor = 1;
            factor = ((t*dt)<rise_time) ? ((t*dt)/rise_time) : 1;
            factor = ((tp-t*dt)<fall_time) ? ((tp-t*dt)/fall_time) : factor;
            Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = factor*coeff_DLT*cross(Mnorm,cross(M,s));
            FLT = factor*coeff_FLT*cross(M,s);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = factor*coeff_DLT*cross(Mnorm,cross(M_interm,s));
            FLT = factor*coeff_FLT*cross(M_interm,s);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M += 0.5 * (dM+dM_interm);

        }

        for(int t=0; t<num_step_relax; t++){
            Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M += 0.5 * (dM+dM_interm);
        }

        mz[i] = normalize(M).z;
    }
}

__global__ void macrospin_kernel_Heun_risefall(
    float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    float dt = deltat;
    int num_step_tot = (int)(ttot/dt);
    int num_step_pulse = (int)(tp/dt);
    int num_step_relax = num_step_tot - num_step_pulse;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_eta = i/Jstep;
    int idx_J = i - idx_eta*Jstep;
    //float eta = etastart + idx_eta*(etaend-etastart)/(etastep-1);
    float eta = etastart * expf((float)idx_eta/(etastep-1)*log(etaend/etastart));
    float J = Jstart + idx_J*(Jend-Jstart)/(Jstep-1);
    float alpha = damping;
    float Hk = anisotropyField;
    float Ms = Msat;
    float gamma = gyromagneticRatio;
    float coeff_DLT = 0.5 * h_bar * DL * J /(Ms*tFM*e_charge);
    float coeff_FLT = 0.5 * h_bar * FL * J /(Ms*tFM*e_charge);
    float3 s = make_float3(0, cosf(eta), sinf(eta));
    float3 A = Anisotropy;
    float3 Hext = Hexternal;

    float3 M = Ms*Mini;
    float3 M_interm;
    float3 Mnorm;
    float3 Heff;
    float3 T_precession;
    float3 T_damping;
    float3 DLT;
    float3 FLT;
    float3 dM;
    float3 dM_interm;
    float factor = 1;
    if(i < num_tot){
        for(int t=0; t<num_step_pulse; t++){
            factor = 1;
            factor = ((t*dt)<rise_time) ? ((t*dt)/rise_time) : 1;
            factor = ((tp-t*dt)<fall_time) ? ((tp-t*dt)/fall_time) : factor;
            Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = factor*coeff_DLT*cross(Mnorm,cross(M,s));
            FLT = factor*coeff_FLT*cross(M,s);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A + Hext;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            DLT = factor*coeff_DLT*cross(Mnorm,cross(M_interm,s));
            FLT = factor*coeff_FLT*cross(M_interm,s);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping + DLT + FLT);
            M += 0.5 * (dM+dM_interm);

        }

        for(int t=0; t<num_step_relax; t++){
            Mnorm = normalize(M);
            M = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M_interm = M + dM;
            Mnorm = normalize(M_interm);
            M_interm = Mnorm*Ms;
            Heff = Hk*dot(A,Mnorm)*A;
            T_precession = (-1)*cross(M_interm,Heff);
            T_damping = alpha*cross(Mnorm,T_precession);
            dM_interm = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);
            M += 0.5 * (dM+dM_interm);
        }

        mz[i] = normalize(M).z;
    }
}

void mag_dynamics_Euler(
    float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    dim3 GridDim(ceil((float)num_tot/BLOCK_SIZE), 1, 1);
    dim3 BlockDim(BLOCK_SIZE, 1, 1);
    macrospin_kernel_Euler<<<GridDim, BlockDim>>>(Mini, etastart, etaend, etastep, Jstart, Jend, Jstep, Hexternal, Anisotropy, mz, num_tot);
}

void mag_dynamics_Heun(
    float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    dim3 GridDim(ceil((float)num_tot/BLOCK_SIZE), 1, 1);
    dim3 BlockDim(BLOCK_SIZE, 1, 1);
    macrospin_kernel_Heun<<<GridDim, BlockDim>>>(Mini, etastart, etaend, etastep, Jstart, Jend, Jstep, Hexternal, Anisotropy, mz, num_tot);
}

// consider thermal effect, set T, thermal_step in the header
void mag_dynamics_Heun_stochastic(
    float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    dim3 GridDim(ceil((float)num_tot/BLOCK_SIZE), 1, 1);
    dim3 BlockDim(BLOCK_SIZE, 1, 1);
    macrospin_kernel_Heun_stochastic<<<GridDim, BlockDim>>>(Mini, etastart, etaend, etastep, Jstart, Jend, Jstep, Hexternal, Anisotropy, mz, num_tot);
}

// output mx, my, mz for a single (J, eta)
void mag_dynamics_Heun_singlecase(
    float3 Mini, float eta, float J, float3 Hexternal, float3 Anisotropy, float* mx, float* my, float* mz) {
    dim3 GridDim(1, 1, 1);
    dim3 BlockDim(1, 1, 1);
    macrospin_kernel_Heun_singlecase<<<GridDim, BlockDim>>>(Mini, eta, J, Hexternal, Anisotropy, mx, my, mz);
}

// output mz for different J but a single eta
void mag_dynamics_Heun_singleeta(
    float3 Mini, float eta, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    dim3 GridDim(ceil((float)num_tot/BLOCK_SIZE), 1, 1);
    dim3 BlockDim(BLOCK_SIZE, 1, 1);
    macrospin_kernel_Heun_singleeta<<<GridDim, BlockDim>>>(Mini, eta, Jstart, Jend, Jstep, Hexternal, Anisotropy, mz, num_tot);
}

void mag_dynamics_Heun_risefall(
    float3 Mini, float etastart, float etaend, int etastep, float Jstart, float Jend, int Jstep, float3 Hexternal, float3 Anisotropy, float* mz, int num_tot) {
    dim3 GridDim(ceil((float)num_tot/BLOCK_SIZE), 1, 1);
    dim3 BlockDim(BLOCK_SIZE, 1, 1);
    macrospin_kernel_Heun_risefall<<<GridDim, BlockDim>>>(Mini, etastart, etaend, etastep, Jstart, Jend, Jstep, Hexternal, Anisotropy, mz, num_tot);
}



