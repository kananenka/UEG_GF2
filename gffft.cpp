#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <complex>
#include <blitz/array.h>
#include<boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/factorials.hpp>
//#include <fftw3.h>
//#include "nfft3util.h"
//#include "nfft3.h"
#include "omp.h"

using namespace blitz;

void frequency_to_time(GfImFreq gfin,GfImTime gfout){

   double beta = gfin.beta;
   int itmax = gfout.itmax;
   int ncount = gfin.ncount;
   int iwmax = gfin.iwmax;

   Array<std::complex<double>,2> gftemp;
   gftemp.resize(shape(ncount,iwmax));
   std::complex<double> iws;
   std::complex<double> II=-1.0;
   II = sqrt(II);
   double taup=0.0;  
   double cs, sn;
   double arg;

   for(int w=0;w<iwmax;++w){
       iws = II*gfin.omegas(w);
       for(int i=0;i<ncount;++i){
           gftemp(i,w) = gfin.gf(i,w) - (gfin.c1(i)/iws + gfin.c2(i)/(iws*iws) + gfin.c3(i)/(iws*iws*iws));
       }
   }
#pragma omp parallel for default(none) shared(ncount,itmax,gfout,beta,gftemp,gfin,iwmax) private(taup,cs,sn,arg)
   for(int i=0;i<ncount;++i){
      for(int t=0;t<itmax;++t){
         taup = gfout.itimes(t);
         for(int w=0;w<iwmax;++w){
            arg = -(2*w+1)*M_PI*taup/beta;
            cs = cos(arg);
            sn = sin(arg);
            gfout.gf(i,t) += (2.0/beta)*(cs*gftemp(i,w).real() - sn*gftemp(i,w).imag());
         }
         gfout.gf(i,t) += (-0.5*gfin.c1(i) + 0.25*gfin.c2(i)*(2.0*taup - beta) + 0.25*gfin.c3(i)*(beta*taup - taup*taup));
      }
   }

   gfout.c1 = gfin.c1;
   gfout.c2 = gfin.c2;
   gfout.c3 = gfin.c3;

}

void time_to_frequency_legendre(GfImFreq &gfout,const GfImTime gfin,Array<std::complex<double>,2> Tnl,
                       int power,int uniform,int nleg){
   gfout.clear();
   int norb = gfin.ncount;
   int iwmax = gfout.iwmax;
   int itmax = gfin.itmax;
   double beta=gfin.beta;
   int ntaur,ntaul;
   int n_int = 2*power + 2;
   int ntau1; int ntau2; int ntau3;
   double h;
   double sqrtl;
   Array<double,2> Gl;
   Gl.resize(shape(norb,nleg));
   Gl = 0.0;
#pragma omp parallel for default(none) shared(Gl,beta,uniform,n_int,nleg,norb) private(sqrtl,ntaul,ntaur,h,ntau1,ntau2,ntau3)
  for(int l=0;l<nleg;++l){
    sqrtl=std::sqrt(2.*l+1.);
    for(int k=0;k<norb;++k){
        Gl(k,l) = 0.0;
        for(int p=0;p<n_int;++p){
          ntaul = uniform*p;
          ntaur = uniform*(p+1);
          h = gfin.itimes(ntaul+1) - gfin.itimes(ntaul);
          for(int t=1;t<(uniform/2+1);++t){
            ntau1 = ntaul + 2*t-2;
            ntau2 = ntaul + 2*t-1;
            ntau3 = ntaul + 2*t;
            Gl(k,l) += h/3.*sqrtl*(gfin.gf(k,ntau1)*boost::math::legendre_p(l,(gfin.itimes(ntau1)/beta-0.5)*2.) +
                                       4.0*gfin.gf(k,ntau2)*boost::math::legendre_p(l,(gfin.itimes(ntau2)/beta-0.5)*2.) +
                                       gfin.gf(k,ntau3)*boost::math::legendre_p(l,(gfin.itimes(ntau3)/beta-0.5)*2.));
          }
        }
      }
  } 

  std::cout << " Writing Legendre coefficients for 000 sector into: Leg.dat" << std::endl;
  std::ofstream gfile("Leg.dat");
  for(int l=0;l<nleg-1;l=l+2)
     gfile << l << " " << Gl(0,l) << " " << l+1 << "  " << Gl(0,l+1) << std::endl;
  gfile.close();

  std::complex<double> conTnlGl;
#pragma omp parallel for default(none) shared(gfout,Gl,Tnl,nleg,norb,iwmax) private(conTnlGl)
  for(int w=0;w<iwmax;++w){
     for(int i=0;i<norb;++i){
        conTnlGl=0;
        for(int l=0;l<nleg;++l){
            conTnlGl += Tnl(w,l)*Gl(i,l);
          }
          gfout.gf(i,w) += conTnlGl;
      }
   }

   gfout.SetupHFT(norb);

}

void ut_Tnl(int iwmax, const int nleg,Array<std::complex<double>,2> &Tnl,int genindex) {
  Tnl.resize(shape(iwmax,nleg));
  std::complex<double> tnlbessel;
  if(genindex==1){
#pragma omp parallel for default(none) shared(Tnl,iwmax,nleg) private(tnlbessel)
    for(int l=0;l<nleg;++l){
       for(int n=0;n<iwmax;++n){
      //tnlbessel=pow(-1,n)*pow(std::complex<double>(0,1),l+1)*std::sqrt(2*l+1.)*sp_bessel::sph_besselJ(l,(2*n+1)*M_PI/2.0);
          tnlbessel=pow(-1,n)*pow(std::complex<double>(0,1),l+1)*std::sqrt(2*l+1.)*boost::math::sph_bessel(l,(2*n+1)*M_PI/2.0);
          Tnl(n,l)=tnlbessel;
       }
    }
  }else{
     std::cout << " Reading Tnl matrix from file... " << std::endl;
     std::ifstream tnlfile("tnl.dat");
     int nl,nw;
     for(int l=0;l<nleg;++l)
        for(int w=0;w<iwmax;++w)
           tnlfile >> nl >> nw >> Tnl(w,l).real() >> Tnl(w,l).imag();
     tnlfile.close(); 
  }

//  if(nl != (nleg-1) || nw != (iwmax-1)) 
}


//void frequency_to_time2(GfImFreq gfin,GfImTime &gfout){

 //  double beta = gfin.beta;
 //  int itmax = gfout.itmax;
 //  int ncount = gfin.ncount;
 //  int iwmax = gfin.iwmax;

 //  ncount = 1;
 //  std::cout << " Warning using NCOUNT = 1 " << std::endl;
 //  std::cout << " itmax = " << itmax << std::endl;
 //  std::cout << " iwmax = " << iwmax << std::endl;

 //  Array<std::complex<double>,2> gftemp;
 //  gftemp.resize(shape(ncount,iwmax));

 //  std::complex<double> iws;
 //  std::complex<double> II=-1.0;
 //  II = sqrt(II);

 //  for(int i=0;i<ncount;++i){
 //     for(int w=0;w<iwmax;++w){
 //        iws = II*gfin.omegas(w);
 //        gftemp(i,w) = gfin.gf(i,w) - (gfin.c1(i)/iws + gfin.c2(i)/(iws*iws) + gfin.c3(i)/(iws*iws*iws));
 //     }
 //  } 

 //  nfft_plan p;
 //  nfft_init_1d(&p,iwmax,itmax);

 //  for(int i=0;i<itmax;++i)   
 //     p.x[i]=gfout.itimes(i)/beta;

 //  if(p.nfft_flags & PRE_ONE_PSI)
 //     nfft_precompute_one_psi(&p); 

 //  for(int k=0;k<ncount;++k){
 //     for(int w=0;w<iwmax;++w){
 //        std::complex<double> coeff = gftemp(k,w);
 //        p.f_hat[k][0] = 1.0;//coeff.real();
 //        p.f_hat[k][1] = 0.0;//coeff.imag(); 
 //        std::cout << p.f_hat[k][0] << "  " << p.f_hat[k][1] << std::endl;
 //     }
 //     nfft_trafo(&p);
 //     for(int t=0;t<itmax;++t){
 //         double tau=gfout.itimes(t);
 //         std::complex<double> gt = std::complex<double>(p.f[t][0], p.f[t][1]);
 //         std::cout << gt << std::endl;
 //         gt*=std::exp(-2.0*M_PI*II*(double)(iwmax/2.0)*p.x[t]);
 //         gt*=std::exp(-M_PI*II*(p.x[t]));
 //         gt*=2.0/beta;
 //         //gfout.gf(k,t)=gt.real();//+f_tau(tau, beta, G_omega.c1(s1,s2,f), G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
 //         gfout.gf(k,t) = gt.real();// - 0.5*gfin.c1(k) + (gfin.c2(k)*0.25)*(-beta + 2.0*tau) + (gfin.c3(k)*0.25)*(beta*tau-tau*tau);
 //     }
 //  }

 //  nfft_finalize(&p);

 //  gfout.c1 = gfin.c1;
 //  gfout.c2 = gfin.c2;
 //  gfout.c3 = gfin.c3;

 //  for(int t=0;t<itmax;++t)
 //    std::cout << gfout.itimes(t) << "  " << gfout.gf(0,t) << std::endl;
//}

void time_to_frequency3(GfImFreq &gfout,const GfImTime gfin,int power,int uniform){
   int    norb = gfin.ncount;
   int   iwmax = gfout.iwmax;
   int   itmax = gfin.itmax;
   double beta = gfin.beta;
   int ntaur,ntaul;
   int n_int = 2*power + 2;
   int ntau1; int ntau2; int ntau3;
   double h;
   double sqrtl;
   std::complex<double> II=-1.0;
   II = sqrt(II);
   std::complex<double> expwt1,expwt2,expwt3;
   std::complex<double> omega;
#pragma omp parallel for default(none) shared(II,n_int,iwmax,gfout,beta,uniform,norb) private(sqrtl,ntaul,ntaur,h,ntau1,ntau2,ntau3,expwt1,expwt2,expwt3,omega)
   for(int k=0;k<norb;++k){
      for(int w=0;w<iwmax;++w){
        gfout.gf(k,w) = 0.0;
        omega = gfout.omegas(w);
        for(int p=0;p<n_int;++p){
          ntaul = uniform*p;
          ntaur = uniform*(p+1);
          h = gfin.itimes(ntaul+1) - gfin.itimes(ntaul);
          for(int t=1;t<(uniform/2+1);++t){
            ntau1 = ntaul + 2*t-2;
            ntau2 = ntaul + 2*t-1;
            ntau3 = ntaul + 2*t;
            expwt1 = exp(II * omega * gfin.itimes(ntau1));
            expwt2 = exp(II * omega * gfin.itimes(ntau2));
            expwt3 = exp(II * omega * gfin.itimes(ntau3));
            gfout.gf(k,w) += h/3.0*(gfin.gf(k,ntau1)*expwt1 + 4.0*gfin.gf(k,ntau2)*expwt2 + gfin.gf(k,ntau3)*expwt3);
          }
        }
      }
  } 
 
  gfout.SetupHFT(norb);

  std::ofstream gfile("FFTout.dat");
  for(int w=0;w<iwmax;++w)
     gfile << w << " " << gfout.gf(0,w).real() << " " << gfout.gf(0,w).imag() << std::endl;
  gfile.close();

}
