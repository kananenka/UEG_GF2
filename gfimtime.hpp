#include <cmath>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <blitz/array.h>

class GfImTime {

   public:

   int itmax;
   int ncount;
   double beta;
   int power;
   int uniform;
   Array<double,2> gf;
   Array<double,1> c1;
   Array<double,1> c2;
   Array<double,1> c3;
   Array<double,1> itimes;

   GfImTime(int,int,int,double);
   void GfImTimePrintGamma();
   void clear();
   ~GfImTime();

};

GfImTime :: GfImTime(int POWER,int UNIFORM,int NCOUNT,double BETA){
//
// Initialize things here
//
   power = POWER;
   uniform = UNIFORM;
   int nd = log2(uniform);
   if(uniform != pow(2,nd)) std::cerr << " Sorry, uniform must be an integer power of two. " << std::endl;
   itmax = 2*(power + 1)*uniform + 1;
   ncount = NCOUNT;
   beta = BETA;
   gf.resize(shape(ncount,itmax));
   c1.resize(shape(ncount));
   c2.resize(shape(ncount));
   c3.resize(shape(ncount));
   itimes.resize(shape(itmax));
   gf = 0.0;
   c1 = 0.0;
   c2 = 0.0;
   c3 = 0.0;
   itimes = 0.0;
// create power-law grid   
   itimes(0) = 0.0;
   itimes(itmax-1) = beta;
   int pinc = uniform;
   int ind = pinc;
// create a mesh of power points:
   for(int i=power;i>=0;--i){
      itimes(ind) = 0.5*beta*pow(2,-i);
      ind += pinc; 
   }
   ind = itmax - 1 - uniform;
   for(int i=power;i>0;--i){
      itimes(ind) = beta*(1.0 - 0.5*pow(2,-i));
      ind -= pinc;
   }
// create a mesh of uniform points
   int npp = 3 + 2*power;
   double ppc, ppn;
   double dtau;
   int istart;
   for(int i=0;i<(npp-1);++i){
      ppc  = itimes(i*pinc);
      ppn  = itimes((i+1)*pinc);  
      dtau = (ppn - ppc)/((double)uniform);
      istart = i*pinc;
      for(int j=1;j<uniform;++j){
         itimes(istart + j) = ppc + j*dtau;
      }
   }

}

void GfImTime :: GfImTimePrintGamma(){

   std::cout << " Writing Greens for (0,0,0) state into Gft000.dat" << std::endl;
   std::ofstream gfile("Gft000.dat");
   for(int t=0;t<itmax;++t){
     gfile << t << " " << 0 << " " << 0 << " " << gf(0,t) << std::endl;
   }
   gfile.close();
}

void GfImTime::clear(){
   gf=0.0;
   c1=0.0;
   c2=0.0;
   c3=0.0;
}

GfImTime::~GfImTime(void){ }

