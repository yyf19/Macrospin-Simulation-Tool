#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "helper_math.h"
#include "header.h"

extern "C"
void computeGold(float* mz_c, int num_tot);
void computeTime(float* m_x, float* m_y, float* m_z);

// write an array to file
void WriteFile(float* arr, char* file_name, int num_elements)
{
    FILE* output = fopen(file_name, "w");
    if (output == NULL) {
        printf("Error opening file %s\n", file_name);
        exit(1);
    }
    for (unsigned i = 0; i < num_elements; i++) {
        fprintf(output, "%f ", arr[i]);
    }
}


int main (int argc, char *argv[])
{
    if(argc != 2){
        printf("Error reading the argument\n");
        return 0;
    }
    long argument = strtol(argv[1], NULL, 10);
    if(argument<=0 || argument>7){
        printf("Argument is not in the proper range [1-7]\n");
        return 0;
    }

    float3 Mini = make_float3(0.02,0,1);   // initial magnetization direction
    float etastart = (float)eta_start/180*pi;
    float etaend = (float)eta_end/180*pi;
    float etafixed = (float)eta_fixed/180*pi;
    int etastep = eta_step;
    float Jstart = J_start;
    float Jend = J_end;
    int Jstep = J_step;
    float3 Hext = make_float3(Hx,0,0);
    float3 Anisotropy = make_float3(0.0,0.0,1.0);
    int num_tot = etastep*Jstep;
    num_tot = etastep*Jstep*thermal_step; //for stochastic computing
    int N = (int)(ttot/deltat2); //number of time steps for computeTime (single case computation (CPU))

    if(argument == 1){
      // GPU computation, Euler method, consider rise and fall time, output final state mz
      float* mz_h= (float*) malloc(num_tot*sizeof(float));
      float* mz_d = NULL;
      cudaMalloc((void**)&mz_d, num_tot*sizeof(float));
      cudaEvent_t start_d, stop_d;
      cudaEventCreate(&start_d);
      cudaEventCreate(&stop_d);
      cudaEventRecord(start_d);

      mag_dynamics_Euler(Mini, etastart, etaend, etastep, Jstart, Jend, Jstep, Hext, Anisotropy, mz_d, num_tot);

      cudaEventRecord(stop_d);
      cudaEventSynchronize(stop_d);
      float device_ms = 0;
      cudaEventElapsedTime(&device_ms, start_d, stop_d);
      cudaEventDestroy(start_d);
      cudaEventDestroy(stop_d);
      printf("GPU Processing time: %f (ms)\n", device_ms);

      cudaMemcpy(mz_h, mz_d, num_tot*sizeof(float), cudaMemcpyDeviceToHost);
      WriteFile(mz_h, (char*)"mz.txt", num_tot);
      cudaFree(mz_d);
      free(mz_h);
    }

    if(argument == 2){
      // GPU computation, Heun method, stochastic mode choice, output final state mz
      float* mz_h= (float*) malloc(num_tot*sizeof(float));
      float* mz_d = NULL;
      cudaMalloc((void**)&mz_d, num_tot*sizeof(float));
      cudaEvent_t start_d, stop_d;
      cudaEventCreate(&start_d);
      cudaEventCreate(&stop_d);
      cudaEventRecord(start_d);
      if(thermal_step != 1)
          mag_dynamics_Heun_stochastic(Mini, etastart, etaend, etastep, Jstart, Jend, Jstep, Hext, Anisotropy, mz_d, num_tot);
      else
          mag_dynamics_Heun(Mini, etastart, etaend, etastep, Jstart, Jend, Jstep, Hext, Anisotropy, mz_d, num_tot);

      cudaEventRecord(stop_d);
      cudaEventSynchronize(stop_d);
      float device_ms = 0;
      cudaEventElapsedTime(&device_ms, start_d, stop_d);
      cudaEventDestroy(start_d);
      cudaEventDestroy(stop_d);
      printf("GPU Processing time: %f (ms)\n", device_ms);

      cudaMemcpy(mz_h, mz_d, num_tot*sizeof(float), cudaMemcpyDeviceToHost);
      WriteFile(mz_h, (char*)"mz.txt", num_tot);
      cudaFree(mz_d);
      free(mz_h);
    }

    if(argument == 3){
      // CPU computation, output final state mz
      float* mz_c= (float*) malloc(num_tot*sizeof(float));
      cudaEvent_t start_h, stop_h;
      cudaEventCreate(&start_h);
      cudaEventCreate(&stop_h);
      cudaEventRecord(start_h);

      computeGold(mz_c, num_tot);

      cudaEventRecord(stop_h);
      cudaEventSynchronize(stop_h);
      float host_ms = 0;
      cudaEventElapsedTime(&host_ms, start_h, stop_h);
      cudaEventDestroy(start_h);
      cudaEventDestroy(stop_h);
      printf("CPU Processing time: %f (ms)\n", host_ms);
      //printf("Speedup: %fX\n", host_ms/device_ms);
      free(mz_c);
    }

    if(argument == 4){
      // single case (etafixed, J_fixed) computation (CPU), output trajectory
      float* m_x= (float*) malloc(N*sizeof(float));
      float* m_y= (float*) malloc(N*sizeof(float));
      float* m_z= (float*) malloc(N*sizeof(float));
      cudaEvent_t start_h, stop_h;
      cudaEventCreate(&start_h);
      cudaEventCreate(&stop_h);
      cudaEventRecord(start_h);

      computeTime(m_x, m_y, m_z);

      cudaEventRecord(stop_h);
      cudaEventSynchronize(stop_h);
      float host_ms = 0;
      cudaEventElapsedTime(&host_ms, start_h, stop_h);
      cudaEventDestroy(start_h);
      cudaEventDestroy(stop_h);
      printf("CPU Processing time: %f (ms)\n", host_ms);

      free(m_x);
      free(m_y);
      free(m_z);
    }

    if(argument == 5){
      // GPU computation for single case (etafixed, J_fixed), Heun method, consider rise/fall time, output trajectory
      float* mx_h= (float*) malloc(N*sizeof(float));
      float* mx_d = NULL;
      cudaMalloc((void**)&mx_d, N*sizeof(float));
      float* my_h= (float*) malloc(N*sizeof(float));
      float* my_d = NULL;
      cudaMalloc((void**)&my_d, N*sizeof(float));
      float* mz_h= (float*) malloc(N*sizeof(float));
      float* mz_d = NULL;
      cudaMalloc((void**)&mz_d, N*sizeof(float));

      cudaEvent_t start_d, stop_d;
      cudaEventCreate(&start_d);
      cudaEventCreate(&stop_d);
      cudaEventRecord(start_d);

      mag_dynamics_Heun_singlecase(Mini, etafixed, J_fixed, Hext, Anisotropy, mx_d, my_d, mz_d);

      cudaEventRecord(stop_d);
      cudaEventSynchronize(stop_d);
      float device_ms = 0;
      cudaEventElapsedTime(&device_ms, start_d, stop_d);
      cudaEventDestroy(start_d);
      cudaEventDestroy(stop_d);
      printf("GPU Processing time: %f (ms)\n", device_ms);

      cudaMemcpy(mx_h, mx_d, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(my_h, my_d, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(mz_h, mz_d, N*sizeof(float), cudaMemcpyDeviceToHost);
      WriteFile(mx_h, (char*)"m_x.txt", N);
      WriteFile(my_h, (char*)"m_y.txt", N);
      WriteFile(mz_h, (char*)"m_z.txt", N);
      cudaFree(mx_d);
      cudaFree(my_d);
      cudaFree(mz_d);
      free(mx_h);
      free(my_h);
      free(mz_h);
    }

    if(argument == 6){
      // GPU computation, Heun method for single eta (etastart), consider rise/fall time
      float* mz_h= (float*) malloc(num_tot*sizeof(float));
      float* mz_d = NULL;
      cudaMalloc((void**)&mz_d, num_tot*sizeof(float));
      cudaEvent_t start_d, stop_d;
      cudaEventCreate(&start_d);
      cudaEventCreate(&stop_d);
      cudaEventRecord(start_d);

      mag_dynamics_Heun_singleeta(Mini, etastart, Jstart, Jend, Jstep, Hext, Anisotropy, mz_d, num_tot);

      cudaEventRecord(stop_d);
      cudaEventSynchronize(stop_d);
      float device_ms = 0;
      cudaEventElapsedTime(&device_ms, start_d, stop_d);
      cudaEventDestroy(start_d);
      cudaEventDestroy(stop_d);
      printf("GPU Processing time: %f (ms)\n", device_ms);

      cudaMemcpy(mz_h, mz_d, num_tot*sizeof(float), cudaMemcpyDeviceToHost);
      WriteFile(mz_h, (char*)"mz_eta0.txt", num_tot);
      cudaFree(mz_d);
      free(mz_h);
    }

    if(argument == 7){
      // GPU computation, Heun method, consider rise and fall time, output final state mz
      float* mz_h= (float*) malloc(num_tot*sizeof(float));
      float* mz_d = NULL;
      cudaMalloc((void**)&mz_d, num_tot*sizeof(float));
      cudaEvent_t start_d, stop_d;
      cudaEventCreate(&start_d);
      cudaEventCreate(&stop_d);
      cudaEventRecord(start_d);

      mag_dynamics_Heun_risefall(Mini, etastart, etaend, etastep, Jstart, Jend, Jstep, Hext, Anisotropy, mz_d, num_tot);

      cudaEventRecord(stop_d);
      cudaEventSynchronize(stop_d);
      float device_ms = 0;
      cudaEventElapsedTime(&device_ms, start_d, stop_d);
      cudaEventDestroy(start_d);
      cudaEventDestroy(stop_d);
      printf("GPU Processing time: %f (ms)\n", device_ms);

      cudaMemcpy(mz_h, mz_d, num_tot*sizeof(float), cudaMemcpyDeviceToHost);
      WriteFile(mz_h, (char*)"mz.txt", num_tot);
      cudaFree(mz_d);
      free(mz_h);
    }

    return EXIT_SUCCESS;
}
