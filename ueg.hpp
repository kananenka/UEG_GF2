#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <blitz/array.h>

using namespace blitz;

class UEG {

//   friend class ZEROTH;
//   friend class FIRST;

   public:

   double rs;
   double kf;
   double tf;
   double kc;
   double length;
   double volume;
   double Nel;
   double pref;
   double beta;
   double ef;
   static const double alpha = 2.837297;
   Array<double,2> Kvecs;
   Array<double,1> Fock;
   int ncount;

   UEG(double,int,double,double);
   ~UEG();
   void kmesh();

};

UEG :: UEG(double RS,int NEL,double BETA,double kci){
   rs = RS;
   Nel = NEL;
   beta = BETA;
   std::cout << "=========================================================="<<std::endl;
   std::cout << "|                 CALCULATION PARAMETERS:                | "<<std::endl;        
   std::cout << "=========================================================="<<std::endl;
   std::cout << " Number of electrons = " << Nel << std::endl;
   std::cout << " Inverse temperature = " << beta 
             << " a.u. " << std::endl;
   std::cout << " Physical temperature = " << 1.0/beta << " a.u. " << std::endl;
   std::cout << " Wigner-Seitz radius = " << rs   
             << " a.u. " << std::endl;
//
// calculate radius of Fermi vector, kc vector and Fermi energy:
//
   kf = pow(9.0*M_PI*0.25,1.0/3.0)/rs;
   kc = sqrt(kci)*kf;
   ef = kf*kf/2.0;
   tf = 0.5*pow(9.0*M_PI/4.0,2.0/3.0)/pow(rs,2.0);
   std::cout << " Fermi vector        = " << kf << " a.u. " << std::endl;
   std::cout << " Fermi energy        = " << ef << " a.u. " << std::endl;
   std::cout << " Fermi temparature   = " << tf << " a.u. " << std::endl;
   std::cout << " Dimensionless T     = " << (1.0/beta)/tf << std::endl;
   std::cout << " Cut-off vector      = " << kc << " a.u. " << std::endl;
//
// calculate volume and length of the box
//
   volume = 4.0*M_PI*pow(rs,3.0)*Nel/3.0;
   length = pow(volume,1.0/3.0);
   std::cout << " Volume              = " << volume  << " a.u. " << std::endl;
   std::cout << " Length              = " << length << " a.u. " << std::endl;
   pref = 2.0*M_PI/length;
   
}


void UEG::kmesh()
{
// 
// this is the upper bound on the number of states within kcut
//
   int nmax = kc*length/(2.0*M_PI) + 1;
   nmax = pow(2*nmax+1,3);
   Kvecs.resize(shape(nmax,3));

   Kvecs = 0.0;
   double ks=0.0;
   ncount=0;

   for(int ix=0;ix<(nmax+1);ix++){
      for(int iy=0;iy<(nmax+1);iy++){
         for(int iz=0;iz<(nmax+1);iz++){
            ks = pref*sqrt(ix*ix + iy*iy + iz*iz);
            if(ks <= kc){
               // (x,y,z)
               Kvecs(ncount,0) = pref*ix;
               Kvecs(ncount,1) = pref*iy;
               Kvecs(ncount,2) = pref*iz;
               ncount++;
               // (-x,y,z)
               if(ix!=0){
                  Kvecs(ncount,0) = -pref*ix;
                  Kvecs(ncount,1) = pref*iy;
                  Kvecs(ncount,2) = pref*iz;
                  ncount++;
               }
               // (x,-y,z)
               if(iy!=0){
                  Kvecs(ncount,0) = pref*ix;
                  Kvecs(ncount,1) = -pref*iy;
                  Kvecs(ncount,2) = pref*iz;
                  ncount++;
               }
               // (x,y,-z)
               if(iz!=0){
                  Kvecs(ncount,0) = pref*ix;
                  Kvecs(ncount,1) = pref*iy;
                  Kvecs(ncount,2) = -pref*iz;
                  ncount++;
               }
               // (-x,-y,z)
               if(iy!=0 && ix!=0){
                  Kvecs(ncount,0) = -pref*ix;
                  Kvecs(ncount,1) = -pref*iy;
                  Kvecs(ncount,2) = pref*iz;
                  ncount++;
               }
               // (-x,y,-z)
               if(iz!=0 && ix!=0){
                  Kvecs(ncount,0) = -pref*ix;
                  Kvecs(ncount,1) = pref*iy;
                  Kvecs(ncount,2) = -pref*iz;
                  ncount++;
               }
               //(x,-y,-z)
               if(iy!=0 && iz!=0){ 
                  Kvecs(ncount,0) = pref*ix;
                  Kvecs(ncount,1) = -pref*iy;
                  Kvecs(ncount,2) = -pref*iz;
                  ncount++;
               }
               //(-x,-y,-z)
               if(iy!=0 && iz!=0 && ix!=0){
                  Kvecs(ncount,0) = -pref*ix;
                  Kvecs(ncount,1) = -pref*iy;
                  Kvecs(ncount,2) = -pref*iz;
                  ncount++;
               }               
            }
         }
      }
   }
   std::cout << " Number of spatial orbitals = " << ncount << std::endl;
   Fock.resize(shape(ncount));
   Fock=0.0;
}

UEG::~UEG(void) {}
