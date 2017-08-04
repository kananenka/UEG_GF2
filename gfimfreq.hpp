#include <cmath>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <blitz/array.h>


class GfImFreq {

   public:

   int iwmax;
   int ncount;
   double beta;
   double mu;
   Array<std::complex<double>,2> gf;
   Array<double,1> c1;
   Array<double,1> c2;
   Array<double,1> c3;
   Array<double,1> omegas;
   Array<double,1> Dmat;
   Array<double,1> Fock;

   GfImFreq(int,int,double);
   void GfImFreqHF(UEG);
   void GfImFreqPrintGamma();
   void SetupHFT(int);
   void density_matrix();
   double density();
   void GenFock(UEG,ZEROTH);
   void clear();
   ~GfImFreq();

};

GfImFreq :: GfImFreq(int IWMAX,int NCOUNT,double BETA){
//
// Initialize things here
//
   iwmax = IWMAX;
   ncount = NCOUNT;
   beta = BETA;
   gf.resize(shape(ncount,iwmax));
   c1.resize(shape(ncount));
   c2.resize(shape(ncount));
   c3.resize(shape(ncount));
   Dmat.resize(shape(ncount));
   Fock.resize(shape(ncount));
   omegas.resize(shape(iwmax));
   gf = 0.0;
   c1 = 0.0;
   c2 = 0.0;
   c3 = 0.0;
   omegas = 0.0;
   Fock = 0.0;
   for(int w=0;w<iwmax;++w) omegas(w) = (2.0*w + 1.0)*M_PI/beta;
}

void GfImFreq :: GfImFreqHF(UEG sim){

   std::complex<double> val=0.0;
   std::complex<double> iw=0.0;
   std::complex<double> II=-1.0;
   II = sqrt(II);
   mu = sim.ef;
   for(int w=0;w<iwmax;++w){
      iw = II*omegas(w);
      for(int v1=0;v1<sim.ncount;++v1){
         val = iw + mu - sim.Fock(v1);
         gf(v1,w) = 1.0/val;
      }
   }
   SetupHFT(sim.ncount);   
}

void GfImFreq :: GfImFreqPrintGamma(){

   std::cout << " Writing Greens function for (0,0,0) state into Gfw000.dat" << std::endl;
   std::ofstream gfile("Gfw000.dat");
   for(int w=0;w<iwmax;++w){
     gfile << w << " " << 0 << " " << 0 << " " << gf(0,w).real() << " " << gf(0,w).imag() << std::endl;
   }
   gfile.close();
}

void GfImFreq::SetupHFT(int ncount){

  std::complex<double> II=-1.0;
  II = sqrt(II);
  std::complex<double> wnmax = II*omegas(iwmax-1);
   for(int v1=0;v1<ncount;++v1){
      c1(v1) = -gf(v1,iwmax-1).imag() * wnmax.imag();
      c2(v1) =  gf(v1,iwmax-1).real() * (wnmax * wnmax).real();
      c3(v1) = -(gf(v1,iwmax-1).imag() - (c1(v1)/wnmax).imag())*(wnmax*wnmax*wnmax).imag();
   }

}

void GfImFreq::density_matrix(){
//
// Perform Fourier Transform to calculate density matrix:
//
   Array<std::complex<double>,2> gftemp;
   gftemp.resize(shape(ncount,iwmax));
   std::complex<double> iws;
   std::complex<double> II=-1.0;
   II = sqrt(II);     
   gftemp=0.0;
   for(int w=0;w<iwmax;++w){
      iws = II*omegas(w);
      for(int i=0;i<ncount;++i){
         gftemp(i,w) = gf(i,w) - (c1(i)/iws + c2(i)/(iws*iws) + c3(i)/(iws*iws*iws));
      }
   }

   Dmat = 0.0;
   double cs, sn;
   double arg;
   for(int i=0;i<ncount;++i){
      for(int w=0;w<iwmax;++w){
         arg = -(2*w+1)*M_PI;
         cs = cos(arg);
         sn = sin(arg);
         Dmat(i) += (2.0/beta)*(cs*gftemp(i,w).real() - sn*gftemp(i,w).imag());
      }   
   }

   double tau=beta;
   for(int i=0;i<ncount;++i){
      Dmat(i) += (-0.5*c1(i) + 0.25*c2(i)*(2.0*tau - beta) + 0.25*c3(i)*(beta*tau - tau*tau));
   }

   Dmat = -2.0*Dmat; 
   
   double nel=0.0;
   for(int i=0;i<ncount;++i) nel += Dmat(i);
}

double GfImFreq::density(){
//
// Perform Fourier Transform to calculate density matrix:
//
   Array<std::complex<double>,2> gftemp;
   gftemp.resize(shape(ncount,iwmax));
   std::complex<double> iws;
   std::complex<double> II=-1.0;
   II = sqrt(II);     
   gftemp=0.0;
   for(int w=0;w<iwmax;++w){
      iws = II*omegas(w);
      for(int i=0;i<ncount;++i){
         gftemp(i,w) = gf(i,w) - (c1(i)/iws + c2(i)/(iws*iws) + c3(i)/(iws*iws*iws));
      }
   }

   Array<double,1> dmat;
   dmat.resize(shape(ncount));
   dmat = 0.0;
   double cs, sn;
   double arg;
   for(int i=0;i<ncount;++i){
      for(int w=0;w<iwmax;++w){
         arg = -(2*w+1)*M_PI;
         cs = cos(arg);
         sn = sin(arg);
         dmat(i) += (2.0/beta)*(cs*gftemp(i,w).real() - sn*gftemp(i,w).imag());
      }   
   }

   double tau=beta;
   for(int i=0;i<ncount;++i){
      dmat(i) += (-0.5*c1(i) + 0.25*c2(i)*(2.0*tau - beta) + 0.25*c3(i)*(beta*tau - tau*tau));
   }

   dmat = -2.0*dmat; 
   
   double nel=0.0;
   for(int i=0;i<ncount;++i) nel += dmat(i);
   return nel;
}

void GfImFreq::clear(){
   gf=0.0;
   c1=0.0;
   c2=0.0;
   c3=0.0;
   mu=0.0;
}

void GfImFreq::GenFock(UEG sim,ZEROTH h0){

   double um = sim.alpha/sim.length;
   int ncount=sim.ncount;
   Array<double,1> Sigma;
   Sigma.resize(shape(ncount));
   Sigma=0.0;
   Fock=0.0;

   for(int v1=0;v1<ncount;++v1){
      for(int v2=0;v2<ncount;++v2){
         if(v1 != v2){
            Sigma(v1) -= Vint(sim.Kvecs,v1,v2,-1.0)*Dmat(v2)/2.0;
         }
      }
      Sigma(v1) = Sigma(v1)/sim.volume;
   }

   for(int v1=0;v1<ncount;++v1)
      Fock(v1) = h0.Epsilon(v1) + Sigma(v1) - Dmat(v1)*um/2.0;
 
}

GfImFreq::~GfImFreq(void){ }



