#include <cmath>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <blitz/array.h>


double Vint(Array<double,2> Kvecs,int one,int two,double sign){
   double k1x = Kvecs(one,0);
   double k1y = Kvecs(one,1);
   double k1z = Kvecs(one,2);
   double k2x = Kvecs(two,0);
   double k2y = Kvecs(two,1);
   double k2z = Kvecs(two,2);
   double v = (k1x + sign*k2x)*(k1x + sign*k2x);
   v += (k1y + sign*k2y)*(k1y + sign*k2y);
   v += (k1z + sign*k2z)*(k1z + sign*k2z);
   if(abs(v)>1.0e-15){
      v = 4.0*M_PI/v;
   }
   return v;
}

void vectorListDN(Array<double,2> Kvecs,Array<int,2> &Flist1,int ncount){
//
// this subroutine will create a vector list corresponding to k-q i.e.
// given vectors k and q, and their location in Kvecs where vector
// k-q will be located in Kvecs?
//
   Flist1.resize(shape(ncount,ncount));
   Flist1=-1;
  
   double tol=1.0e-10;
   double v1x,v1y,v1z,v2x,v2y,v2z;
   double v3x,v3y,v3z;
   double vdx,vdy,vdz;
#pragma omp parallel for default(none) shared(Kvecs,ncount,Flist1,tol) private(v1x,v1y,v1z,v2x,v2y,v2z,vdx,vdy,vdz,v3x,v3y,v3z)
   for(int ind1=0;ind1<ncount;++ind1){
      v1x = Kvecs(ind1,0);
      v1y = Kvecs(ind1,1);
      v1z = Kvecs(ind1,2);
      for(int ind2=0;ind2<ncount;++ind2){
         v2x = Kvecs(ind2,0);
         v2y = Kvecs(ind2,1);
         v2z = Kvecs(ind2,2);
         vdx = v1x - v2x;
         vdy = v1y - v2y;
         vdz = v1z - v2z;
         for(int ind3=0;ind3<ncount;++ind3){
            v3x = Kvecs(ind3,0);
            v3y = Kvecs(ind3,1);
            v3z = Kvecs(ind3,2);
            if(abs(vdx-v3x)<tol && abs(vdy-v3y)<tol && abs(vdz-v3z)<tol){
               Flist1(ind1,ind2)=ind3;
               
            }
         }
      }
   }   

}

void vectorListSN(Array<double,2> Kvecs,Array<int,2> &Flist,int ncount){
//
// this subroutine will create a vector list corresponding to k-q i.e.
// given vectors k and q, and their location in Kvecs where vector
// k-q will be located in Kvecs?
//
   Flist.resize(shape(ncount,ncount));
   Flist=-1;
  
   double tol=1.0e-10;
   double v1x,v1y,v1z,v2x,v2y,v2z;
   double v3x,v3y,v3z;
   double vdx,vdy,vdz;
#pragma omp parallel for default(none) shared(Kvecs,ncount,Flist,tol) private(v1x,v1y,v1z,v2x,v2y,v2z,vdx,vdy,vdz,v3x,v3y,v3z)
   for(int ind1=0;ind1<ncount;++ind1){
      v1x = Kvecs(ind1,0);
      v1y = Kvecs(ind1,1);
      v1z = Kvecs(ind1,2);
      for(int ind2=0;ind2<ncount;++ind2){
         v2x = Kvecs(ind2,0);
         v2y = Kvecs(ind2,1);
         v2z = Kvecs(ind2,2);
         vdx = v1x + v2x;
         vdy = v1y + v2y;
         vdz = v1z + v2z;
         for(int ind3=0;ind3<ncount;++ind3){
            v3x = Kvecs(ind3,0);
            v3y = Kvecs(ind3,1);
            v3z = Kvecs(ind3,2);
            if(abs(vdx-v3x)<tol && abs(vdy-v3y)<tol && abs(vdz-v3z)<tol){
               Flist(ind1,ind2)=ind3;
            }
         }
      }
   }   
}

void vectorListDD(Array<double,2> Kvecs,Array<int,3> &Flist,int ncount){
//
// this subroutine will create a vector list corresponding to k-q i.e.
// given vectors k and q, and their location in Kvecs where vector
// k-q will be located in Kvecs?
//
   Flist.resize(shape(ncount,ncount,ncount));
   Flist=-1;
  
   double tol=1.0e-10;
   double v1x,v1y,v1z,v2x,v2y,v2z;
   double v3x,v3y,v3z;
   double v4x,v4y,v4z;
   double vdx,vdy,vdz;
#pragma omp parallel for default(none) shared(Kvecs,ncount,Flist,tol) private(v1x,v1y,v1z,v2x,v2y,v2z,v3x,v3y,v3z,vdx,vdy,vdz,v4x,v4y,v4z)
   for(int ind1=0;ind1<ncount;++ind1){
      v1x = Kvecs(ind1,0);
      v1y = Kvecs(ind1,1);
      v1z = Kvecs(ind1,2);
      for(int ind2=0;ind2<ncount;++ind2){
         v2x = Kvecs(ind2,0);
         v2y = Kvecs(ind2,1);
         v2z = Kvecs(ind2,2);
         for(int ind3=0;ind3<ncount;++ind3){
            v3x = Kvecs(ind3,0);
            v3y = Kvecs(ind3,1);
            v3z = Kvecs(ind3,2);
            for(int ind4=0;ind4<ncount;++ind4){
               v4x = Kvecs(ind4,0);
               v4y = Kvecs(ind4,1);
               v4z = Kvecs(ind4,2);
               vdx = v1x - v2x - v3x;
               vdy = v1y - v2y - v3y;
               vdz = v1z - v2z - v3z;
               if(abs(vdx-v4x)<tol && abs(vdy-v4y)<tol && abs(vdz-v4z)<tol){
                  Flist(ind1,ind2,ind3)=ind4;
               }
            }
         }
      }
   }   
}
