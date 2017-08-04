#include <cmath>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <blitz/array.h>

using namespace blitz;

class FIRST {

   double energy;
   double mu;
   double toccup;
   double ennel;
   int ncount;
   static const double alpha = 2.837297;
   double um;

   public:
   
   Array<double,1> Dmat;
   Array<double,1> Sigma;

   FIRST(UEG,ZEROTH);
   ~FIRST();
   void SelfEnergy(UEG,ZEROTH);
   void Energy(UEG sim);

};

FIRST :: FIRST(UEG sim,ZEROTH h0){
   
   Sigma.resize(shape(sim.ncount));
   Dmat.resize(shape(sim.ncount));
   Dmat=0.0;
   Sigma=0.0;
   ncount = sim.ncount;
   Dmat = h0.Dmat;
   um = alpha/sim.length;

   double toadd;
   for(int v1=0;v1<ncount;++v1){
      for(int v2=0;v2<ncount;++v2){
         if(v1 != v2){
            Sigma(v1) -= Vint(sim.Kvecs,v1,v2,-1.0)*Dmat(v2)/2.0;
         }
      }
      Sigma(v1) = Sigma(v1)/sim.volume;
   }

   for(int k=0;k<ncount;++k)
      sim.Fock(k) += Sigma(k) - Dmat(k)*um/2.0;

/*
 *  for(int v1=0;v1<ncount;++v1){
 *     Sigma(v1) = 0.0;
 *     double y = sqrt(sim.Kvecs(v1,0)*sim.Kvecs(v1,0) + 
 *                     sim.Kvecs(v1,1)*sim.Kvecs(v1,1) + 
 *                     sim.Kvecs(v1,2)*sim.Kvecs(v1,2));
 *     y = y/kf;
 *     if(y!=0){
 *        if(abs(y-1.0)<1.0e-10){
 *           Sigma(v1) = -2.0*kf/M_PI;
 *        }else{
 *           Sigma(v1) = -2.0*(kf/M_PI)*(1.0 + (1.0 - y*y)/(2.0*y)*log(abs((1.0+y)/(1.0-y))));
 *        }
 *     }
 *  }           
 */
}

void FIRST::Energy(UEG sim){

   energy = EnergyFock(Sigma,Dmat,sim.ncount,0.5);
   ennel = energy/sim.Nel;
   std::cout << " First-order energy: " << std::endl;
   std::cout << "---------------------" << std::endl;
   std::cout << " E_1/N             = "  << ennel << " a.u. " << std::endl;
   std::cout << " E_Madelung        = "  << um/2.0 << " a.u. " << std::endl;

   energy -= sim.Nel*um/2.0;
   ennel = energy/sim.Nel;

   std::cout << " (E_1 - 1/2 E_M)/N = " << ennel << std::endl;
   std::cout << " E_1/N from TDL    = " << -0.4582/sim.rs << " a.u. " << std::endl;
   std::cout << " Error wrt TDL     = " << abs(ennel+0.4582/sim.rs) << " a.u. " << std::endl;

   std::ofstream f1file("EpsHF.dat");
   f1file << "# HF eigenvalues: Nx, Ny, Nz, Eps " << std::endl;
   for(int ind=0;ind<ncount;++ind){
      f1file <<ind << " \t " << sim.Kvecs(ind,0)/sim.pref
                  <<  " \t " << sim.Kvecs(ind,1)/sim.pref
                  <<  " \t " << sim.Kvecs(ind,2)/sim.pref
                  <<  " \t " << Dmat(ind)
                  <<  " \t " << sim.Fock(ind) << std::endl;
   }
   f1file.close();
   std::cout << " Hartree--Fock eigenvalues are written to EpsHF.dat"<< std::endl; 
}

FIRST::~FIRST(void){ }
