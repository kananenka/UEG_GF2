#include <cmath>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <blitz/array.h>

class ZEROTH {

   double energy;
   double mu;
   double toccup;
   double ennel;

   public:
   
   Array<double,1> Dmat;
   Array<double,1> Epsilon;

   ZEROTH(UEG);
   ~ZEROTH();
   void checkNe(UEG);

};

ZEROTH :: ZEROTH(UEG sim){

   Dmat.resize(shape(sim.ncount));
   Epsilon.resize(shape(sim.ncount));

   Dmat=0.0;
   Epsilon=0.0;

   toccup=0.0;
   std::cout << "----------------------------------------------------------"<<std::endl;
   std::cout << "                    Occupied states:            " << std::endl;
   std::cout << " Nx \t Ny \t Nz \t Energy \t Occupation      " << std::endl;
   std::cout << "----------------------------------------------------------"<<std::endl;
   double ek=0.0;
   energy = 0.0;
   double enk;
   for(int ind=0; ind < sim.ncount; ind++){
      ek = sqrt(sim.Kvecs(ind,0)*sim.Kvecs(ind,0) + 
                sim.Kvecs(ind,1)*sim.Kvecs(ind,1) + 
                sim.Kvecs(ind,2)*sim.Kvecs(ind,2));
      enk = ek*ek/2.0;
      Epsilon(ind) = enk;   
      Dmat(ind) = 2.0/(exp((enk - sim.ef)*sim.beta)+1.0);
      toccup += Dmat(ind);
      if(Dmat(ind) > 1.0e-4) 
         std::cout << sim.Kvecs(ind,0)/sim.pref << " \t " 
                   << sim.Kvecs(ind,1)/sim.pref << " \t " 
                   << sim.Kvecs(ind,2)/sim.pref << " \t " 
                   << enk << " \t " << Dmat(ind)  
                   << " " << ind << std::endl;
            
   }
   std::cout << "----------------------------------------------------------"<<std::endl;

   sim.Fock = Epsilon;
   energy = EnergyFock(Epsilon,Dmat,sim.ncount,1.0);
   ennel = energy/toccup;
   std::cout << " Zero-order energy:  " << std::endl;
   std::cout << "---------------------" << std::endl;
   std::cout << " E_0/N             = " << ennel << " a.u. " << std::endl;
   double e0tdl = 1.10495/(sim.rs*sim.rs);
   std::cout << " E_0/N from TDL    = " << e0tdl << " a.u. " << std::endl;
   std::cout << " Error wrt TDL     = " << abs(ennel-e0tdl) << " a.u. " << std::endl;
   
}

void ZEROTH::checkNe(UEG sim){

   std::cout << " Total number of electrons = " 
             << toccup << std::endl;
  
   std::cout << " REPLACE  THIS WITH BISECTION " << std::endl; 
   double dff=abs(toccup - sim.Nel);
   if(dff>1.0e-5) {
      std::cout << " Wrong number of electrons please adjust chemical potential: " << std::endl;
      std::cout << " Nel(input) = " << sim.Nel << " Nel = " << toccup << " ABS(dff) " << dff << std::endl;   
   }else{
      std::cout << " ----> Number of electrons is good! " << toccup << std::endl;
   }


}

ZEROTH::~ZEROTH(void){ }

