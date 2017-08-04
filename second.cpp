#include <cmath>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <omp.h>
#include <blitz/array.h>

using namespace blitz;

void Direct2(GfImTime &Sigmad2,GfImTime GF,UEG sim,Array<int,2> F2vD,Array<int,2> F2vS)
{

   double beta=sim.beta;
   double volume=sim.volume;
   int ncount=sim.ncount;
   int itmax=GF.itmax;
   Sigmad2.clear();

   int ind1,ind2,ind3;
   double vq,k1x,k1y,k1z;
#pragma omp parallel for default(none) shared(Sigmad2,itmax,ncount,GF,volume,sim,F2vD,F2vS) private(ind1,ind2,vq,k1x,k1y,k1z)
   for(int t=0;t<itmax;++t){
      for(int k=0;k<ncount;++k){
         for(int p=0;p<ncount;++p){
            for(int q=1;q<ncount;++q){
               ind1 = F2vD(k,q);
               if(ind1>-1){          //meaning that k-q vector is within the kc sphere, here case k=q is ignored
                 ind2 = F2vS(p,q); 
                 if(ind2>-1){        //meaning that p+q vector is within the kc sphere
                     k1x = sim.Kvecs(q,0);
                     k1y = sim.Kvecs(q,1);
                     k1z = sim.Kvecs(q,2);
                     vq = k1x*k1x + k1y*k1y + k1z*k1z;
                     vq = 4.0*M_PI/vq;
                     Sigmad2.gf(k,t) += 2.0*GF.gf(ind1,t)*GF.gf(ind2,t)*GF.gf(p,itmax-t)*vq*vq/(volume*volume);            
                 }
               }
            }
         }
      }
   }
}

void Exchange2(GfImTime &Sigmax2,GfImTime GF,UEG sim,Array<int,3> F3vD,Array<int,2> F2vD)
{
   double beta=sim.beta;
   double volume=sim.volume;
   int ncount=sim.ncount;
   int itmax=GF.itmax;
   Sigmax2.clear();

   int ind1,ind2,ind3;
   double vq,vp,k1xq,k1yq,k1zq,k1xp,k1yp,k1zp;
#pragma omp parallel for default(none) shared(Sigmax2,itmax,ncount,GF,volume,sim,F2vD,F3vD) private(ind1,ind2,vq,k1xq,k1yq,k1zq,k1xp,k1yp,k1zp,vp,ind3)
   for(int t=0;t<itmax;++t){
      for(int k=0;k<ncount;++k){
         for(int p=1;p<ncount;++p){
            for(int q=1;q<ncount;++q){
               ind1 = F2vD(k,q);
               if(ind1>-1){        //ignore k=q case for which this is =0
                 ind2 = F2vD(k,p);
                 if(ind2>-1){     
                   ind3 = F3vD(k,q,p);
                   if(ind3>-1){       //again ignore k-q-p=0 case
                      k1xq = sim.Kvecs(q,0);
                      k1yq = sim.Kvecs(q,1);
                      k1zq = sim.Kvecs(q,2);
                      vq = k1xq*k1xq + k1yq*k1yq + k1zq*k1zq;
                      vq = 4.0*M_PI/vq;
                      k1xp = sim.Kvecs(p,0);
                      k1yp = sim.Kvecs(p,1);
                      k1zp = sim.Kvecs(p,2);
                      vp = k1xp*k1xp + k1yp*k1yp + k1zp*k1zp;
                      vp = 4.0*M_PI/vp;
                      Sigmax2.gf(k,t) -= GF.gf(ind1,t)*GF.gf(ind2,t)*GF.gf(ind3,itmax-t)*vq*vp/(volume*volume);
                   }
                 }
               }
            }
         }
      }
   }
}


void Sigma2All(GfImTime &Sigma2,GfImTime Sx,GfImTime Sd){

   Sigma2.clear();
   int itmax = Sx.itmax;
   int norb = Sx.ncount;

   for(int t=0;t<itmax;++t)
      for(int i=0;i<norb;++i)
         Sigma2.gf(i,t) = Sx.gf(i,t) + Sd.gf(i,t);

   double sigma_der_zero;
   double sigma_der_beta;
   double sigma_dder_zero;
   double sigma_dder_beta;
   double temp;
//set up high frequency tails
   for(int k=0;k<norb;++k){
      Sigma2.c1(k) = -Sigma2.gf(k,0) - Sigma2.gf(k,itmax-1);
      sigma_der_zero = (Sigma2.gf(k,1)-Sigma2.gf(k,0))/(Sigma2.itimes(1) - Sigma2.itimes(0));
      sigma_der_beta = (Sigma2.gf(k,itmax-1)-Sigma2.gf(k,itmax-2))/(Sigma2.itimes(itmax-1) - Sigma2.itimes(itmax-2));
      Sigma2.c2(k) = sigma_der_zero + sigma_der_beta;
      temp = Sigma2.itimes(1) - Sigma2.itimes(0);
      sigma_dder_zero = (Sigma2.gf(k,2) - 2.0*Sigma2.gf(k,1) + Sigma2.gf(k,0))/(temp*temp);
      temp = Sigma2.itimes(itmax-1) - Sigma2.itimes(itmax-2);
      sigma_dder_beta = (Sigma2.gf(k,itmax-1) - 2.0*Sigma2.gf(k,itmax-2) + Sigma2.gf(k,itmax-3))/(temp*temp);
      Sigma2.c3(k) = -1.0*(sigma_dder_zero + sigma_dder_beta);
   }
}
