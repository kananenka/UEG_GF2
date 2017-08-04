#include <cmath>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <blitz/array.h>

using namespace blitz;

void constructGF(UEG sim,GfImFreq &gf,GfImFreq Sigma,double mu);
double EnergyFock(Array<double,1> Fock,Array<double,1> Dmat,int ncount,double pref);

double EnergyFock(Array<double,1> Fock,Array<double,1> Dmat,int ncount,double pref){
  double energy = 0.0;
  for(int ind=0;ind<ncount;++ind)
  energy += Fock(ind)*Dmat(ind)*pref;
  return energy;

}

void bisecton(UEG sim,GfImFreq gf,GfImFreq Sigma){

   int nitermax = 100;
   int niter = 0;
   int nel = sim.Nel;
   double mu = gf.mu;
   double tol=1.0e-5;
   double mu_min = -10.0;
   double mu_max = 10.0;
   constructGF(sim,gf,Sigma,mu);
   double dens = gf.density();

   //std::cout << " gf.gf " << gf.gf(0,0) << std::endl;
   if(abs(dens-nel)>tol){
      std::cout << " Bisection will be performed: Nel(GF) = " << dens << " Nel = " << nel << std::endl;
      mu = 0.0;
      do{
        constructGF(sim,gf,Sigma,mu);
        dens = gf.density();
        constructGF(sim,gf,Sigma,mu);
        //std::cout << " gf.gf " << gf.gf(0,0) << std::endl;
        dens = gf.density();
        //std::cout << " mu = " << mu << " nel = " << dens << std::endl; 
        //if(abs(dens-nel)<=tol) break;
        if(dens>nel){
           mu_max = mu;  
        }else{
           mu_min = mu;
        } 
        mu = (mu_max + mu_min)/2.0;
      }
      while(abs(dens-nel)>tol || niter > nitermax);
   }
   std::cout << " Bisection converged for mu = " << mu << " Nel = " << dens << std::endl;
}

void constructGF(UEG sim,GfImFreq &gf,GfImFreq Sigma,double mu){
   int iwmax = gf.iwmax;
   int norb = sim.ncount;
   gf.clear();

   std::complex<double> val=0.0;
   std::complex<double> iw=0.0;
   std::complex<double> II=-1.0;
   II = sqrt(II);

#pragma omp parallel for default(none) shared(gf,Sigma,mu,sim,iwmax,norb,II) private(iw,val)
   for(int w=0;w<iwmax;++w){
      iw = II*gf.omegas(w);
      for(int i=0;i<norb;++i){
         val = iw + mu - sim.Fock(i) - Sigma.gf(i,w);
         gf.gf(i,w) = 1.0/val;
      }
   } 

   gf.SetupHFT(norb);
   gf.mu=mu;
   gf.density_matrix();

}

double twobodyenergy(const GfImFreq Sigma, const GfImFreq gf,UEG sim,double prefactor){

   double beta = Sigma.beta;
   double norb = Sigma.ncount;
   double iwmax = Sigma.iwmax;
   double nel = sim.Nel;

   double energy=0.0;
   double trace=0.0;
   for(int w=0;w<iwmax;++w){
      for(int k=0;k<norb;++k){
        trace += prefactor*(gf.gf(k,w).real()*Sigma.gf(k,w).real() - gf.gf(k,w).imag()*Sigma.gf(k,w).imag());
      }
  }
  energy = trace/(beta*nel);

  double hifreq_integral_square = 1/((2.0*iwmax + 1)*M_PI/beta)*beta/(2*M_PI);
  double hifreq_integral_fourth = 1/(3*std::pow((2.0*iwmax+1)*M_PI/beta,3))*beta/(2*M_PI);

  double energy_hifreq_i;
  double energy_hifreq_r;
  double energy_hifreq;

  for(int k=0;k<norb;++k)
     energy_hifreq_i -= prefactor*(gf.c1(k)*Sigma.c1(k));

  for(int k=0;k<norb;++k)
     energy_hifreq_r += prefactor*(gf.c2(k)*Sigma.c2(k));

  energy_hifreq_i*=hifreq_integral_square/beta;
  energy_hifreq_r*=hifreq_integral_fourth/beta;
  energy_hifreq = energy_hifreq_i/nel + energy_hifreq_r/nel;
  std::cout<<"----------------------------------" <<std::endl;
  std::cout<<"E2b/N           = "<<       energy        << std::endl;
  std::cout<<"E_hft/N         = "<<       energy_hifreq << std::endl;
  std::cout<<"(E2b + E_hft)/N = "<< energy + energy_hifreq<<std::endl;
  std::cout<<"----------------------------------" <<std::endl;
  return energy+energy_hifreq;

}

double onebodyenergy(UEG sim,Array<double,1> Dmat,Array<double,1> Fock,Array<double,1> Epsilon){
   double energy=0.0;
   int norb= sim.ncount;
   for(int k=0;k<norb;++k)
      energy += 0.5*Dmat(k)*(Fock(k) + Epsilon(k));

   energy = energy/sim.Nel;
   std::cout << " One-body energy: " << std::endl;
   std::cout << "------------------" << std::endl;
   std::cout << " E1b/N        = " << energy <<std::endl;
   return energy;
}
   
