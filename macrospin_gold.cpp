#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "header.h"
#include <iostream>
#include <fstream>
using namespace std;
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold(float* mz_c, int num_tot);
void computeTime(float* m_x, float* m_y, float* m_z);



struct vector{
	float x;
	float y;
	float z;
};

float Dot(vector v1, vector v2){
	vector vr;
    float value;
	vr.x = v1.x*v2.x;
	vr.y = v1.y*v2.y;
	vr.z = v1.z*v2.z;
	value=vr.x+vr.y+vr.z;
	return value;

}
vector Normalized(vector v1){
	vector vr;
	vr.x = v1.x*v1.x;
	vr.y = v1.y*v1.y;
	vr.z = v1.z*v1.z;
	float mag=sqrt(vr.x+vr.y+vr.z);
    vr.x = v1.x/mag;
	vr.y = v1.y/mag;
	vr.z = v1.z/mag;
	return vr;
}
vector Cross(vector v1, vector v2){
	vector vr;
	vr.x = (v1.y*v2.z - v1.z*v2.y);
	vr.y = -(v1.x*v2.z - v1.z*v2.x);
	vr.z = (v1.x*v2.y - v1.y*v2.x);
	return vr;
}
vector add(vector v1, vector v2){
	vector vr;
	vr.x = v1.x+v2.x;
	vr.y = v1.y+v2.y;
	vr.z = v1.z+v2.z;
	return vr;
}

vector times(vector v, float n){
	vector vr;
	vr.x = v.x*n;
	vr.y = v.y*n;
	vr.z = v.z*n;

	return vr;
}
vector div(vector v, float n){
	vector vr;
	vr.x = v.x/n;
	vr.y = v.y/n;
	vr.z = v.z/n;

	return vr;
}

vector make_vec3(vector v , float x, float y, float z)
{
   // vector vc;
    v.x =x;
    v.y =y;
    v.z =z;
    return v;
}


void computeTime(float* m_x, float* m_y, float* m_z)
{
	vector mini = make_vec3(mini, 0.02, 0, 1);
	vector hext = make_vec3(hext, 0, 0, 0);
    vector anisotropy = make_vec3(anisotropy, 0.0, 0.0, 1.0);

    float dt =deltat2;
    int num_step_tot = (int)(ttot/dt);
    int N1=num_step_tot;
    int num_step_pulse = (int)(tp/dt);
    int N2=num_step_pulse;
    int num_step_relax = num_step_tot - num_step_pulse;
	float alpha = damping;
    float Hk = anisotropyField;
    float Ms = Msat;
    float gamma = gyromagneticRatio;
	vector A = anisotropy;
	vector Hext = hext;
	vector M = times(mini, Ms);
  	int i=0;
	vector Mnorm, Heff, T_precession, T_damping, DLT, FLT, dM;

    float eta =(float)eta_fixed / 180 * pi;
	float J =(float)J_fixed;

    float coeff_DLT = 0.5 * h_bar * DL * J /(Ms*tFM*e_charge);
    float coeff_FLT = 0.5 * h_bar * FL * J /(Ms*tFM*e_charge);
    vector s = make_vec3(s, 0.0, cosf(eta), sinf(eta));


    for(int t=0; t<N1; t++){
			if(t<N2)
            {
			Mnorm =Normalized(M);
            M = times(Mnorm, Ms);
            Heff = add((times(A, Dot(A, Mnorm)* Hk)), Hext);
            T_precession = times(Cross(M,Heff),-1);
            T_damping = times(Cross(Mnorm,T_precession), alpha);
            DLT = times(Cross(Mnorm,Cross(M,s)), coeff_DLT);
            FLT =times(Cross(M,s), coeff_FLT);
			//dM = gamma / (1 + powf(alpha, 2)) * dt * (T_precession + T_damping + DLT + FLT);
			float cdm = gamma / (1 + powf(alpha, 2)) * dt;
			vector tpart = add(T_precession, T_damping);
			vector DFLT= add (DLT,FLT);
			vector lastpart = add(tpart, DFLT);
			dM = times(lastpart, cdm);
			M =add(M, dM);
			}

            else
            {

            Mnorm = Normalized(M);
			M = times(Mnorm, Ms);
            Heff =times(A, Dot(A, Mnorm)* Hk);
			T_precession = times(Cross(M, Heff), -1);
			T_damping = times(Cross(Mnorm, T_precession), alpha);
            //dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);

			float cdm = gamma / (1 + powf(alpha, 2)) * dt;
			vector tpart = add(T_precession, T_damping);
			dM = times(tpart, cdm);

			M = add(M, dM);
			}

            m_x[t] = Normalized(M).x;
            m_y[t] = Normalized(M).y;
            m_z[t] = Normalized(M).z;


	}

ofstream outdatax ("m_x.txt");
 for (i=0; i<N1; ++i)
      outdatax << m_x[i] << endl;
   outdatax.close();
ofstream outdatay ("m_y.txt");
 for (i=0; i<N1; ++i)
      outdatay << m_y[i] << endl;
   outdatay.close();

ofstream outdataz ("m_z.txt");
 for (i=0; i<N1; ++i)
      outdataz << m_z[i] << endl;
   outdataz.close();

}






void computeGold(float* mz_c, int num_tot)
{


   	float etastart = (float)eta_start / 180 * pi;
	float etaend = (float)eta_end / 180 * pi;
	int etastep = eta_step;
	float Jstart = J_start;
	float Jend = J_end;
	int Jstep = J_step;
	int n=etastep*Jstep;

	vector mini = make_vec3(mini, 0.02, 0, 1);
	vector hext = make_vec3(hext, 0, 0, 0);
    vector anisotropy = make_vec3(anisotropy, 0.0, 0.0, 1.0);
    float dt = deltat;
    int num_step_tot = (int)(ttot/dt);
    int num_step_pulse = (int)(tp/dt);
    int num_step_relax = num_step_tot - num_step_pulse;
	float alpha = damping;
    float Hk = anisotropyField;
    float Ms = Msat;
    float gamma = gyromagneticRatio;
	vector A = anisotropy;
	vector Hext = hext;
	vector Mnorm, Heff, T_precession, T_damping, DLT, FLT, dM;
	int i=0;

   for (int k=0; k<etastep; k++){


     float eta = etastart * expf((float)k/(etastep-1)*log(etaend/etastart));


	 for (int j=0; j<Jstep; j++)
	 {
       vector M = times(mini, Ms);
       float J = Jstart + j*(Jend-Jstart)/(Jstep-1);

       float coeff_DLT = 0.5 * h_bar * DL * J /(Ms*tFM*e_charge);
       float coeff_FLT = 0.5 * h_bar * FL * J /(Ms*tFM*e_charge);
       vector s = make_vec3(s, 0.0, cosf(eta), sinf(eta));


        for(int t=0; t<num_step_pulse; t++)
	    {
	        Mnorm =Normalized(M);
            M = times(Mnorm, Ms);
            Heff = add((times(A, Dot(A, Mnorm)* Hk)), Hext);
            T_precession = times(Cross(M,Heff),-1);
            T_damping = times(Cross(Mnorm,T_precession), alpha);
            DLT = times(Cross(Mnorm,Cross(M,s)), coeff_DLT);
            FLT =times(Cross(M,s), coeff_FLT);
			//dM = gamma / (1 + powf(alpha, 2)) * dt * (T_precession + T_damping + DLT + FLT);
			float cdm = gamma / (1 + powf(alpha, 2)) * dt;
			vector tpart = add(T_precession, T_damping);
			vector DFLT= add (DLT,FLT);
			vector lastpart = add(tpart, DFLT);
			dM = times(lastpart, cdm);
			M =add(M, dM);

        }

        for(int t=0; t<num_step_relax; t++)
		{
            Mnorm = Normalized(M);
			M = times(Mnorm, Ms);
            Heff =times(A, Dot(A, Mnorm)* Hk);
			T_precession = times(Cross(M, Heff), -1);
			T_damping = times(Cross(Mnorm, T_precession), alpha);
            //dM = gamma / (1+powf(alpha,2)) * dt * (T_precession + T_damping);

			float cdm = gamma / (1 + powf(alpha, 2)) * dt;
			vector tpart = add(T_precession, T_damping);
			dM = times(tpart, cdm);

			M = add(M, dM);
       	}
        i=j+k*Jstep;
        mz_c[i] = Normalized(M).z;

     }

 ofstream outdata ("CMz.txt");
 for (i=0; i<n; ++i)
      outdata << mz_c[i]<< endl;
   outdata.close();

   }


}