/*****************************************************************************
 *
 * ALPS DMFT Project
 *
 * Copyright (C) 2005 - 2009 by Emanuel Gull <gull@phys.columbia.edu>
 *                              Philipp Werner <werner@itp.phys.ethz.ch>,
 *                              Sebastian Fuchs <fuchs@theorie.physik.uni-goettingen.de>
 *                              Matthias Troyer <troyer@comp-phys.org>
 *
 *
 * This software is part of the ALPS Applications, published under the ALPS
 * Application License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 *
 * You should have received a copy of the ALPS Application License along with
 * the ALPS Applications; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include <complex>
#include "gfpower.hpp"
#include "gffour.hpp"
#include <fftw3.h>
#include "nfft3util.h"
#include "nfft3.h"
#include "omp.h"

std::vector<double> mmult(const std::vector<double> &m1, const std::vector<double> &m2, int nmo);

inline std::complex<ft_float_type> f_omega(std::complex<ft_float_type> iw, ft_float_type c1, ft_float_type c2, ft_float_type c3) {
  std::complex<ft_float_type> iwsq=iw*iw;
  return c1/iw + c2/(iwsq) + c3/(iw*iwsq);
}


inline ft_float_type f_tau(ft_float_type tau, ft_float_type beta, ft_float_type c1, ft_float_type c2, ft_float_type c3) {
  return -0.5*c1 + (c2*0.25)*(-beta+2.*tau) + (c3*0.25)*(beta*tau-tau*tau);
}

void frequency_to_time_ft(itime_green_function_t &G_tau, const matsubara_green_function_t &G_omega, ft_float_type beta) {
  if(G_tau.nflavor()!=G_omega.nflavor() || G_tau.nsite()!=G_omega.nsite()) throw std::logic_error("GF in tau and omega have different shape.");
  int N_tau = G_tau.ntime()-1;
  int N_omega = G_omega.nfreq();
  int N_site = G_omega.nsite();
  int N_flavor = G_omega.nflavor();
  std::complex<double> I(0., 1.);
  std::cout<<"Ntau is: "<<N_tau<<" Nomega is: "<<N_omega<<std::endl;//jordan
  green_function<std::complex<ft_float_type> > G_omega_no_model_2(G_omega.nfreq(), G_omega.nsite(), G_omega.nflavor());
  green_function<std::complex<ft_float_type> > G_omega_model(G_omega.nfreq(), G_omega.nsite(), G_omega.nflavor());
  //green_function<std::complex< double> > G_omega_no_model(G_omega);
  G_tau.c1()=G_omega.c1();
  G_tau.c2()=G_omega.c2();
//std::cout<<"G_tau.c2() frequency_to_time_ft"<<std::endl;//jordan
//      for(std::size_t p=0;p<G_tau.nsite();++p){
//        for(std::size_t q=0;q<G_tau.nsite();++q){
//               std::cout<<p<<" "<<q<<" "<<G_tau.c2(p,q,0)<<std::endl;
//        }
//      } //jordan ends
  G_tau.c3()=G_omega.c3();
  for(int f=0;f<G_omega.nflavor();++f){
    for (int s1=0; s1<N_site; ++s1){
      for (int s2=0; s2<N_site; ++s2) {
        if(G_omega.c1(s1,s2,f)==0 && G_omega.c2(s1,s2,f) == 0 && G_omega.c3(s1,s2,f)){  //nothing happening in this gf.
          for (int i=0; i<=N_tau; i++) {
            G_tau(i,s1,s2,f)=0.;
          }
        }
        else {
          for (std::size_t k=0; k<N_omega; k++) {
            std::complex<ft_float_type> iw(0,(2*k+1)*M_PI/beta);
            //G_omega_no_model(k,s1,s2,f) -= f_omega(iw, G_omega.c1(s1,s2,f),G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
            G_omega_model(k,s1,s2,f) = f_omega(iw, G_omega.c1(s1,s2,f),G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
            G_omega_no_model_2(k,s1,s2,f) = std::complex<ft_float_type>(G_omega(k,s1,s2,f)) - f_omega(iw, G_omega.c1(s1,s2,f),G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
          }
        }
      }
    }
  }
  /** run non-equidistant fast Fourier transform */
  nfft_plan p;
  /** init an one dimensional plan */
  nfft_init_1d(&p,N_omega,N_tau);
  /** copy the times into the nfft container*/
  for(int i=0;i<N_tau;++i){
    p.x[i]=G_tau.tau(i)/beta;//std::copy(G_tau_nfft.tau().begin(), G_tau_nfft.tau().end(), p.x);
  }
  /** precompute psi, the entries of the matrix B */
  if(p.nfft_flags & PRE_ONE_PSI)
    nfft_precompute_one_psi(&p);
  
  for(std::size_t f=0;f<N_flavor;++f){
    for (std::size_t s1=0; s1<N_site; ++s1){
      for (std::size_t s2=0; s2<N_site; ++s2) {
        for(int k=0;k<N_omega;++k){
          std::complex<double> coeff=G_omega_no_model_2(k,s1,s2,f);
          p.f_hat[k][0]=coeff.real();
          p.f_hat[k][1]=coeff.imag();
        }
        nfft_trafo(&p);
        for(int i=0;i<N_tau;++i){
          double tau=G_tau.tau(i);
          std::complex<double> gt=std::complex<double>(p.f[i][0], p.f[i][1]);
          gt*=std::exp(-2.*M_PI*I*(double)(N_omega/2.)*p.x[i]);
          gt*=std::exp(-M_PI*I*(p.x[i]));
          gt*=2./beta;
          G_tau(i,s1,s2,f)=gt.real()+f_tau(tau, beta, G_omega.c1(s1,s2,f), G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
        }
      }
    }
  }
  
  
  for(std::size_t f=0;f<N_flavor;++f){
    for (std::size_t s1=0; s1<N_site; ++s1){
      for (std::size_t s2=0; s2<N_site; ++s2) {
        
        G_tau(N_tau,s1,s2,f)= -1.0*G_omega.c1(s1,s2,f);//s1==s2 ? -1. : 0.;
        G_tau(N_tau,s1,s2,f)-=G_tau(0,s1,s2,f);
      }
    }
  }
  
}

void density(const itime_green_function_t &gf, std::vector<double> &density,const std::vector<double> &s_ao){
  
  density.resize(gf.nsite()*gf.nflavor());
  std::vector<double>  aux_mat(gf.nsite()*gf.nsite());
  aux_mat.resize(gf.nsite()*gf.nsite());
  for(std::size_t i=0;i<gf.nsite();++i){
    for(std::size_t j=0;j<gf.nsite();++j){
      aux_mat[i*gf.nsite()+j]=-gf(gf.ntime()-1, i,j,0)*2.;
    }
  }
  aux_mat=mmult(aux_mat,s_ao,gf.nsite());
  for(std::size_t i=0;i<gf.nsite();++i){
    density[i]=aux_mat[i*gf.nsite()+i];
  }
}
void compute_density_matrix(const itime_green_function_t &gf, std::vector<double> &density){
  density.resize(gf.nsite()*gf.nsite()*gf.nflavor());
  double trace=0.;
  for(std::size_t i=0;i<gf.nsite();++i){
    for(std::size_t j=0;j<gf.nsite();++j){
      for(std::size_t f=0;f<gf.nflavor();++f){
        density[i*gf.nsite()*gf.nflavor()+j*gf.nflavor()+f]=-gf(gf.ntime()-1, i,j,f)*2.;
      }
    }
    trace+=density[i*gf.nsite()*gf.nflavor()+i*gf.nflavor()];
  }
  //std::cout<<"density is: "<<trace<<std::endl;
}
void density(const matsubara_green_function_t &gf, std::vector<double> &density, double beta,const std::vector<double> &s_ao){
  density.resize(gf.nsite()*gf.nflavor());
  std::vector<double>  aux_mat(gf.nsite()*gf.nsite());
  aux_mat.resize(gf.nsite()*gf.nsite());
  for(std::size_t s1=0;s1<gf.nsite();++s1){
    for(std::size_t s2=0;s2<gf.nsite();++s2){
      for(std::size_t f=0;f<gf.nflavor();++f){
        double n=-f_tau(beta, beta, gf.c1(s1,s2,f),gf.c2(s1,s2,f), gf.c3(s1,s2,f));
        for(std::size_t w=0;w<gf.nfreq();++w){
          std::complex<double> iw(0,(2*w+1)*M_PI/beta);
          n+=2./beta*(gf(w,s1,s2,f).real()-f_omega(iw,gf.c1(s1,s2,f),gf.c2(s1,s2,f), gf.c3(s1,s2,f)).real());
        }
        aux_mat[s1*gf.nsite()+s2]=2.*n;
        //density[s1*gf.nflavor()+f]=2.*n;
      }
    }
  }
  aux_mat=mmult(aux_mat,s_ao,gf.nsite());
  for(std::size_t s1=0;s1<gf.nsite();++s1){
    density[s1]=aux_mat[s1*gf.nsite()+s1];
  }
}


void compute_density_matrix1(const matsubara_green_function_t &G_omega, std::vector<double> &density, ft_float_type beta) {
 
  int N_omega = G_omega.nfreq();
  int N_site = G_omega.nsite();
  int N_flavor = G_omega.nflavor();
  std::complex<double> I(0., 1.);
  density.resize(N_site*N_site);
  green_function<std::complex<ft_float_type> > G_omega_no_model_2(G_omega.nfreq(), G_omega.nsite(), G_omega.nflavor());
  green_function<std::complex<ft_float_type> > G_omega_model(G_omega.nfreq(), G_omega.nsite(), G_omega.nflavor());


  for(int f=0;f<G_omega.nflavor();++f){
    for (int s1=0; s1<N_site; ++s1){
      for (int s2=0; s2<N_site; ++s2) {
          for (std::size_t k=0; k<N_omega; k++) {
            std::complex<ft_float_type> iw(0,(2*k+1)*M_PI/beta);
            //G_omega_no_model(k,s1,s2,f) -= f_omega(iw, G_omega.c1(s1,s2,f),G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
            G_omega_model(k,s1,s2,f) = f_omega(iw, G_omega.c1(s1,s2,f),G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
            G_omega_no_model_2(k,s1,s2,f) = std::complex<ft_float_type>(G_omega(k,s1,s2,f)) - f_omega(iw, G_omega.c1(s1,s2,f),G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
	    //	    std::cout<<"LLL "<<f_omega(iw,G_omega.c1(s1,s2,f),G_omega.c2(s1,s2,f),G_omega.c3(s1,s2,f)).real()<<std::endl;
          }
      }
      }
  }
  
  /** run non-equidistant fast Fourier transform */
  int s_t=1;
  nfft_plan p;
  /** init an one dimensional plan */
  nfft_init_1d(&p,N_omega,s_t);
  /** copy the times into the nfft container*/
  //for(int i=0;i<N_tau;++i){
  p.x[0]=beta/beta;//std::copy(G_tau_nfft.tau().begin(), G_tau_nfft.tau().end(), p.x);
//}
  
  /** precompute psi, the entries of the matrix B */
  if(p.nfft_flags & PRE_ONE_PSI)
    nfft_precompute_one_psi(&p);
  
  //  for(std::size_t f=0;f<N_flavor;++f){
  int f=0;
    for (std::size_t s1=0; s1<N_site; ++s1){
      for (std::size_t s2=0; s2<N_site; ++s2) {
        for(int k=0;k<N_omega;++k){
          std::complex<double> coeff=G_omega_no_model_2(k,s1,s2,f);
          p.f_hat[k][0]=coeff.real();
	  //          p.f_hat[k][1]=coeff.imag();
        }
        nfft_trafo(&p);
       
	//        for(int i=0;i<N_tau;++i){
          double tau=beta;
          std::complex<double> gt=std::complex<double>(p.f[0][0],0.0);
          gt*=std::exp(-2.*M_PI*I*(double)(N_omega/2.)*p.x[0]);
          gt*=std::exp(-M_PI*I*(p.x[0]));
          gt*=2./beta;
	  
          density[s1*N_site+s2]=2.0*(gt.real()+f_tau(tau, beta, G_omega.c1(s1,s2,f), G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f)));
	  //}
	  
      }
    }
    //}
  
  
  
  //std::ofstream g_omega_no_model_file("/tmp/g_omega_no_model_w2t"); g_omega_no_model_file<<std::make_pair(G_omega_no_model_2, beta)<<std::endl;
  //std::ofstream g_omega_model_file("/tmp/g_omega_model_w2t"); g_omega_model_file<<std::make_pair(G_omega_model, beta)<<std::endl;
  //std::ofstream g_omega_file("/tmp/g_omega_w2t"); g_omega_file<<std::make_pair(G_omega, beta)<<std::endl;
}


void compute_density_matrix(const matsubara_green_function_t &gf, std::vector<double> &density, double beta){
  density.resize(gf.nsite()*gf.nsite()*gf.nflavor());
  
  for(std::size_t f=0;f<gf.nflavor();++f){
    for(std::size_t i=0;i<gf.nsite();++i){
      for(std::size_t j=0;j<gf.nsite();++j){
        double n=-f_tau(beta, beta, gf.c1(i,j,f),gf.c2(i,j,f), gf.c3(i,j,f));
        for(std::size_t w=0;w<gf.nfreq();++w){
          std::complex<double> iw(0,(2*w+1)*M_PI/beta);
          n+=2./beta*(gf(w,i,j,f).real()-f_omega(iw,gf.c1(i,j,f),gf.c2(i,j,f), gf.c3(i,j,f)).real());
          //std::cout<<"LLL "<<f_omega(iw,gf.c1(i,j,f),gf.c2(i,j,f),gf.c3(i,j,f)).real()<<std::endl;
        }
        //	std::cout<<"LLL******** "<<i<<" "<<j<<" "<<n<<std::endl;
        density[i*gf.nsite()*gf.nflavor()+j*gf.nflavor()+f]=2.*n;
      }
      //trace+=density[i*gf.nsite()*gf.nflavor()+i*gf.nflavor()+f];
    }
  }

//for(int i=0;i<84;i=i+14){
//  for(int k=0;k<14;++k){
//    for(int l=0;l<14;++l){
//      density[(k+i)*84+i+l]=density[k*84+l];
//
//    }
//  }
//}
//for(int i=0;i<18;i=i+3){
//  for(int k=0;k<3;++k){
//    for(int l=0;l<3;++l){
//      density[(k+i)*18+i+l]=density[k*18+l];
//    }
//  }
//}
//for(int i=0;i<nmo;i=i+3){
//  for(int k=0;k<3;++k){
//    for(int l=0;l<3;++l){
//      density[(k+i)*nmo+i+l]=density[k*nmo+l];
//
//    }
//  }
//}
// Note: this only works for H64 with minimal basis!!
//use only with finite cubic H lattice
// for(int i=0;i<gf.nsite();++i){
//   for(int j=0;j<gf.nsite();++j){
//     density[i*gf.nsite()+j]=density[(gf.nsite()-(j+1))*gf.nsite()+gf.nsite()-(i+1)];
//   }
// }
// H32 square lattice only 
// density[3*32+3]=density[0*32+0];//first the four corners
// density[12*32+12]=density[0*32+0];
// density[15*32+15]=density[0*32+0];
// density[2*32+2]=density[1*32+1];//now the middle two on sides
// density[4*32+4]=density[1*32+1];
// density[7*32+7]=density[1*32+1];
// density[8*32+8]=density[1*32+1];
// density[11*32+11]=density[1*32+1];
// density[13*32+13]=density[1*32+1];
// density[14*32+14]=density[1*32+1];
// density[6*32+6]=density[5*32+5];//now the center four 
// density[9*32+9]=density[5*32+5];
// density[10*32+10]=density[5*32+5];
// for(int i=0;i<gf.nsite();++i){
//   for(int j=0;j<gf.nsite();++j){
//     density[(gf.nsite()-(j+1))*gf.nsite()+gf.nsite()-(i+1)]=density[i*gf.nsite()+j];
//   }
// }
//
// H64 8x8 lattice only 
// for(int i=0;i<gf.nsite();++i){
//   for(int j=0;j<gf.nsite();++j){
//     density[(gf.nsite()-(j+1))*gf.nsite()+gf.nsite()-(i+1)]=density[i*gf.nsite()+j];
//   }
// }
// symm for reverse ordering of Hydrogens
// for(int i=0;i<64;++i){
//   density[(63-i)*64+63-i]=density[i*64+i];
// }
// 4 symms for mirror bisecting sides
// for(int j=0;j<4;j++){
//   for(int i=0;i<8;i++){
//     density[(8*(7-j)+i)*64+(8*(7-j)+i)]=density[(i+8*j)*64+(i+8*j)];
//   } 
// } 
// for(int i=0;i<8;++i){
//   density[(56+i)*64+56+i]=density[i*64+i];
// }
// for(int i=0;i<8;++i){
//   density[(48+i)*64+48+i]=density[(i+8)*64+i+8];
// }
// for(int i=0;i<8;++i){
//   density[(40+i)*64+40+i]=density[(i+16)*64+i+16];
// }
// for(int i=0;i<8;++i){
//   density[(32+i)*64+32+i]=density[(i+24)*64+i+24];
// }
// 8 symms for mirror bisecting corners
// for(int j=0;j<8;j++){
//   for(int i=0;i<8;i++){
//     density[(i*8+j)*64+(i*8+j)]=density[(i+8*j)*64+(i+8*j)];
//   } 
// } 
//
//
// for H12 rect 4x3 lattice only 
// density[3*12+3]=density[0*12+0];//first the four corners
// density[8*12+8]=density[0*12+0];
// density[11*12+11]=density[0*12+0];
// density[2*12+2]=density[1*12+1];//now the middle two on sides
// density[9*12+9]=density[1*12+1];
// density[10*12+10]=density[1*12+1];
// density[6*12+6]=density[5*12+5];//now the center two 
// density[7*12+7]=density[4*12+4];//now the side two
//

// for H16 square lattice only 
// density[3*16+3]=density[0*16+0];//first the four corners
// density[12*16+12]=density[0*16+0];
// density[15*16+15]=density[0*16+0];
// density[2*16+2]=density[1*16+1];//now the middle two on sides
// density[4*16+4]=density[1*16+1];
// density[7*16+7]=density[1*16+1];
// density[8*16+8]=density[1*16+1];
// density[11*16+11]=density[1*16+1];
// density[13*16+13]=density[1*16+1];
// density[14*16+14]=density[1*16+1];
// density[6*16+6]=density[5*16+5];//now the center four 
// density[9*16+9]=density[5*16+5];
// density[10*16+10]=density[5*16+5];
// for(int i=0;i<gf.nsite();++i){
//   for(int j=0;j<gf.nsite();++j){
//     density[(gf.nsite()-(j+1))*gf.nsite()+gf.nsite()-(i+1)]=density[i*gf.nsite()+j];
//   }
// }
//
// for(int i=1;i<8;++i){//for H8 cube only! every H is equivalent
//   density[0*8+0]+=density[i*8+i];
// }
// density[0*8+0]=density[0*8+0]/8.0;
// for(int i=0;i<8;++i){
//   density[i*8+i]=density[0*8+0];
// }
// for(int i=1;i<4;++i){//for H4 square only! every H is equivalent
//   density[0*4+0]+=density[i*4+i];
// }
// density[0*4+0]=density[0*4+0]/4.0;
// for(int i=0;i<4;++i){
//   density[i*4+i]=density[0*4+0];
// }
//
// 2x4 H8 rectangle only!
// density[1*8+1]=density[0*8+0];//4 corners
// density[6*8+6]=density[0*8+0];
// density[7*8+7]=density[0*8+0];
// density[3*8+3]=density[2*8+2];//4 middle
// density[4*8+4]=density[2*8+2];
// density[5*8+5]=density[2*8+2];
//
// h6 ring only!
// for(int i=0;i<3;i++){
//   density[(3+i)*18+(3+i)]=density[(0+i)*18+(0+i)];
//   density[(6+i)*18+(6+i)]=density[(0+i)*18+(0+i)];
//   density[(9+i)*18+(9+i)]=density[(0+i)*18+(0+i)];
//   density[(12+i)*18+(12+i)]=density[(0+i)*18+(0+i)];
//   density[(15+i)*18+(15+i)]=density[(0+i)*18+(0+i)];
// }
// for(int j=3;j<18;j++){
//   for(int i=j;i<18;i++){
//     density[i*18+j]=density[(i-3)*18+(j-3)];
//   }
// }
//  
//
//
// LiH only!
// density[7*11+7]=density[5*11+5];
// density[8*11+8]=density[10*11+10];
//
// LiLi only!
// density[5*18+5]=density[3*18+3];
// density[8*18+8]=density[6*18+6];
// for(int i=0;i<9;i++){
// density[(i+9)*18+(i+9)]=density[i*18+i];
// }
//
// He-He with aug-cc-pvdz
// account for the p orb symmetr
// density[5*18+5]=density[3*18+3];
// density[8*18+8]=density[6*18+6];
// He-He mirror symmetry
// for(int i=0;i<9;++i){
//   density[(9+i)*18+(9+i)]=density[i*18+i];
// }
//  
//
// He-ghostHe with aug-cc-pvdz
// account for the p orb symmetr
// density[5*18+5]=density[3*18+3];
// density[8*18+8]=density[6*18+6];
// density[14*18+14]=density[12*18+12];
// density[17*18+17]=density[15*18+15];
// 
//
// He-He with aug-cc-pvtz
// account for the 3p orb symmetr
// density[6*46+6]=density[4*46+4];
// density[9*46+9]=density[7*46+7];
// density[12*46+12]=density[10*46+10];
// account for the 2d orb symmetr
// density[14*46+14]=density[13*46+13];
// density[19*46+19]=density[18*46+18];
// He-He mirror symmetry
// for(int i=0;i<23;++i){
//   density[(23+i)*46+(23+i)]=density[i*46+i];
// }
//  
//
// He-ghostHe with aug-cc-pvtz
// account for the 3p orb symmetr
// density[6*46+6]=density[4*46+4];
// density[9*46+9]=density[7*46+7];
// density[12*46+12]=density[10*46+10];
// account for the 2d orb symmetr
// density[14*46+14]=density[13*46+13];
// density[19*46+19]=density[18*46+18];
// now do the same for the non symmetric ghost
// account for the 3p orb symmetr
// density[29*46+29]=density[27*46+27];
// density[32*46+32]=density[30*46+30];
// density[35*46+35]=density[33*46+33];
// account for the 2d orb symmetr
// density[37*46+37]=density[36*46+36];
// density[42*46+42]=density[41*46+41];
//
// Ar-Ar with aug-cc-pvtz   6s5p3d2f
// density[8*100+8]=density[6*100+6];
// density[11*100+11]=density[9*100+9];
// density[14*100+14]=density[12*100+12];
// density[17*100+17]=density[15*100+15];
// density[20*100+20]=density[18*100+18];
// density[22*100+22]=density[21*100+21];
// density[27*100+27]=density[26*100+26];
// density[32*100+32]=density[31*100+31];
// Ar-Ar mirror symmetry
// for(int i=0;i<50;++i){
//   density[(50+i)*100+(50+i)]=density[i*100+i];
// }
//
// Ar-ghostAr with aug-cc-pvtz
// density[8*100+8]=density[6*100+6];
// density[11*100+11]=density[9*100+9];
// density[14*100+14]=density[12*100+12];
// density[17*100+17]=density[15*100+15];
// density[20*100+20]=density[18*100+18];
// density[22*100+22]=density[21*100+21];
// density[27*100+27]=density[26*100+26];
// density[32*100+32]=density[31*100+31];
//    now the same for the ghost Argon
// density[58*100+58]=density[56*100+56];
// density[61*100+61]=density[59*100+59];
// density[64*100+64]=density[62*100+62];
// density[67*100+67]=density[65*100+65];
// density[70*100+70]=density[68*100+68];
// density[72*100+72]=density[71*100+71];
// density[77*100+77]=density[76*100+76];
// density[82*100+82]=density[81*100+81];
//
//
// N2 with 18 basis functions
// account for the p orb symmetr
// density[5*18+5]=density[3*18+3];
// density[8*18+8]=density[6*18+6];
// now account for N-N mirror symmetry
// for(int i=0;i<9;++i){
//   density[i*18+i]+=density[(9+i)*18+(9+i)];
//   density[i*18+i]=0.5*density[i*18+i];
// }
// for(int i=0;i<9;++i){
//   density[(9+i)*18+(9+i)]=density[i*18+i];
// }
//
//
}

extern "C" void dgesv_(int *N, int *NRHS,double *A, int *LDA,int *IPIV,double *B,int * LDB,int * INFO );
void generate_spline_matrix(std::vector<double> & spline_matrix, const std::vector<double> &tau_grid, int Np1) {
  spline_matrix.assign(Np1*Np1, 0.);
  std::vector<double> A(Np1*Np1, 0.);
  //numerical recipes eq. 3.3.7
  for(int i=1;i<Np1-1;++i){
    A[i*Np1+i]=(tau_grid[i+1]-tau_grid[i-1])/3.;
  }
  
  for (int i=1; i<Np1-1; i++) {
    A[i*Np1+i-1] = (tau_grid[i]-tau_grid[i-1])/6.;
    A[i*Np1+i+1] = (tau_grid[i+1]-tau_grid[i])/6.;
  }
  
  //this is for the c3 coefficient
  A[0] = 1.;
  A[Np1-1] = 1.;
  
  //this is for the c2 coefficient
  A[Np1*(Np1-1)      ] = -2.*(tau_grid[1    ]-tau_grid[0    ])/6.;
  A[Np1*(Np1-1)+ 1   ] = -1.*(tau_grid[1    ]-tau_grid[0    ])/6.;
  A[Np1*(Np1-1)+Np1-2] =  1.*(tau_grid[Np1-1]-tau_grid[Np1-2])/6.;
  A[Np1*(Np1-1)+Np1-1] =  2.*(tau_grid[Np1-1]-tau_grid[Np1-2])/6.;
  
  /*//this is for the first derivative, at zero:
   A[0] = -2./6.*(tau_grid[1    ]-tau_grid[0    ]);
   A[1] = -1./6.*(tau_grid[1    ]-tau_grid[0    ]);
   //this is for the second derivative, at beta:
   A[Np1*(Np1-1)+Np1-2] =  1./6.*(tau_grid[Np1-1]-tau_grid[Np1-2]);
   A[Np1*(Np1-1)+Np1-1] =  2./6.*(tau_grid[Np1-1]-tau_grid[Np1-2]);*/
  
  for(int i=0;i<Np1;++i){spline_matrix[i*Np1+i]=1.;}
  std::vector<int> ipivot(Np1);
  int info;
  /*std::cout<<"A matrix before: "<<std::endl;
   for(int i=0;i<Np1;++i){
   for(int j=0;j<Np1;++j){
   std::cout<<A[i*Np1+j]<<" ";
   }
   std::cout<<std::endl;
   }*/
  /*std::cout<<"spline matrix before: "<<std::endl;
   for(int i=0;i<Np1;++i){
   for(int j=0;j<Np1;++j){
   std::cout<<spline_matrix[i*Np1+j]<<" ";
   }
   std::cout<<std::endl;
   }*/
  dgesv_(&Np1,&Np1,&(A[0]),&Np1,&(ipivot[0]),&(spline_matrix[0]),&(Np1),&info);
  /*std::cout<<"spline matrix after: "<<std::endl;
   for(int i=0;i<Np1;++i){
   for(int j=0;j<Np1;++j){
   std::cout<<spline_matrix[i*Np1+j]<<" ";
   }
   std::cout<<std::endl;
   }
   exit(1);*/
  if(info !=0) throw std::runtime_error("matrix solver failed in spline matrix routine.");
}



void evaluate_second_derivatives(const std::vector<double> &tau_grid, std::vector<double> & spline_matrix, std::vector<double> & g, std::vector<double> & second_derivatives, const double c1g, const double c2g, const double c3g, int Np1) {
  std::vector<double> rhs(Np1, 0);
  //G''(0)+G''(beta)=-c3
  rhs[0] = -c3g;
  
  //rhs[0    ] = gprime_zero-(g[1] - g[0])/(tau_grid[1]-tau_grid[0]);
  //rhs[Np1-1] = gprime_beta-(g[Np1-1] - g[Np1-2])/(tau_grid[Np1-1]-tau_grid[Np1-2]);
  //this is according to Eq. B.12 in my thesis, or Eq. 3.3.7 in numerical recipes.
  for (int i=1; i<Np1-1; i++) {
    rhs[i] = (g[i+1]-g[i])/(tau_grid[i+1]-tau_grid[i])-(g[i]-g[i-1])/(tau_grid[i]-tau_grid[i-1]);
  }
  
  //this is the derivative coefficient c2g
  rhs[Np1-1] = c2g -(g[1] - g[0])/(tau_grid[1]-tau_grid[0]) -(g[Np1-1] - g[Np1-2])/(tau_grid[Np1-1]-tau_grid[Np1-2]);
  
  for (int i=0; i<Np1; i++) {
    second_derivatives[i]=0;
    for (int j=0; j<Np1; j++) {
      second_derivatives[i] += spline_matrix[i*Np1+j]*rhs[j];
    }
  }
  //std::cout<<"second derivative at zero: "<<second_derivatives[0]<<" beta: "<<second_derivatives[Np1-1]<<" sum: "<<second_derivatives[0]+second_derivatives[Np1-1]<<std::endl;
  //double vprime_zero=(g[1] - g[0])/(tau_grid[1]-tau_grid[0])
  //-2./6.*(tau_grid[1    ]-tau_grid[0    ])*second_derivatives[0    ]-1./6.*(tau_grid[1    ]-tau_grid[0    ])*second_derivatives[1    ];
  //double vprime_beta=(g[Np1-1] - g[Np1-2])/(tau_grid[Np1-1]-tau_grid[Np1-2])
  //+1./6.*(tau_grid[Np1-1]-tau_grid[Np1-2])*second_derivatives[Np1-2]+2./6.*(tau_grid[Np1-1]-tau_grid[Np1-2])*second_derivatives[Np1-1];
  //std::cout<<"first derivative at zero: "<<vprime_zero<<std::endl;
  //std::cout<<"first derivative at beta: "<<vprime_beta<<std::endl;
  //std::cout<<"first derivative sum: "<<vprime_zero+vprime_beta<<" should be: "<<gprime_zero<<" "<<gprime_beta<<" "<<gprime_zero+gprime_beta<<std::endl;
}

void time_to_frequency_fft(const itime_green_function_t & G_tau, matsubara_green_function_t & gomega, double beta){
  std::vector<double> v,v2;
  int Np1 = G_tau.ntime();
  std::vector<double> tau_vector=G_tau.tau();
  std::vector<double> uniform_tau_grid;
  std::vector<double> uniform_tau_values;
  std::vector<std::complex<double> > fft_prefactor;
  std::vector<std::complex<double> > fft_output;
  
  int N_omega = gomega.nfreq();
  int N_tau=G_tau.ntime();
  
  gomega.c1()=G_tau.c1();
  gomega.c2()=G_tau.c2();
  gomega.c3()=G_tau.c3();
//std::cout<<"G_tau.c2() time_to_frequency_fft"<<std::endl;//jordan
//      for(std::size_t p=0;p<G_tau.nsite();++p){
//        for(std::size_t q=0;q<G_tau.nsite();++q){
//               std::cout<<p<<" "<<q<<" "<<G_tau.c2(p,q,0)<<std::endl;
//        }
//      } //jordan ends
  /*std::vector<double> spline_matrix(Np1* Np1, 0.);
  double time1 = omp_get_wtime();
  std::cout<<"generating spline matrix."<<std::endl;
  generate_spline_matrix(spline_matrix, tau_vector, Np1);
  std::cout<<"generated spline matrix."<<std::endl;
  double time2 = omp_get_wtime();
  std::cout<<"spline time: "<<time2-time1<<std::endl;
  {
    double dtau=G_tau.tau(1)-G_tau.tau(0);
    for(int i=0;i*dtau<beta;++i){
      uniform_tau_grid.push_back(i*dtau);
    }
    uniform_tau_grid.push_back(beta);
    
    for(std::size_t f=0;f<G_tau.nflavor();++f){
      for(std::size_t p=0;p<G_tau.nsite();++p){
        for(std::size_t q=0;q<G_tau.nsite();++q){
          //prepare values
          uniform_tau_values.resize(uniform_tau_grid.size());
          fft_prefactor.resize(uniform_tau_grid.size());
          fft_output.resize(uniform_tau_grid.size());
          v2.resize(Np1);
          v.resize(Np1);
          //Spline interpolation
          for(int tau=0;tau<Np1;++tau){
            v[tau]=G_tau(tau,p,q,f);
          }
          // matrix containing the second derivatives y'' of interpolated y=v[tau] at points tau_n
          evaluate_second_derivatives(tau_vector, spline_matrix, v, v2, G_tau.c1(p,q,f), G_tau.c2(p,q,f), G_tau.c3(p,q,f),Np1);
          
          //evaluation of these splines on an uniform grid
          int j=0;
          for(std::size_t i=0;i<uniform_tau_grid.size()-1;++i){
            double tau_i=dtau*i;
            if(G_tau.tau(j)>tau_i){
              throw std::logic_error("problem with increment.");
            }else if(G_tau.tau(j)==tau_i){
              uniform_tau_values[i]=G_tau(j,p,q,f);// - f_tau(tau_i, beta, G_tau.c1(p,q,f), G_tau.c2(p,q,f), G_tau.c3(p,q,f));
            }else{
              while(G_tau.tau(j+1)<tau_i) j++;
              //interpolate spline between point j and point j+1, using the second derivatives from the spline interpolation
              double dt=G_tau.tau(j+1)-G_tau.tau(j);
              double A = (G_tau.tau(j+1)-tau_i)/dt;
              double B = 1-A;
              double C = 1./6.*(A*A*A-A)*dt*dt;
              double D = 1./6.*(B*B*B-B)*dt*dt;
              uniform_tau_values[i]=A*v[j]+B*v[j+1]+C*v2[j]+D*v2[j+1];// - f_tau(tau_i, beta, G_tau.c1(p,q,f), G_tau.c2(p,q,f), G_tau.c3(p,q,f));
            }
          }
          uniform_tau_values[uniform_tau_values.size()-1]=G_tau(G_tau.ntime()-1,p,q,f);// - f_tau(beta, beta,G_tau.c1(p,q,f), G_tau.c2(p,q,f), G_tau.c3(p,q,f));
          //initialize data
          for(std::size_t j=0;j<uniform_tau_values.size()-1;++j){
            fft_prefactor[j]=uniform_tau_values[j]*std::exp(std::complex<double>(0., M_PI*j/(double)(uniform_tau_grid.size()-1)));
          }
          fftw_plan A_plan=fftw_plan_dft_1d(uniform_tau_grid.size()-1, (fftw_complex*)(&fft_prefactor[0]), (fftw_complex*)(&fft_output[0]), FFTW_BACKWARD, FFTW_ESTIMATE);
          fftw_execute(A_plan);
          fftw_destroy_plan(A_plan);
          
          for (int k=0; k<N_omega; k++) {
            gomega(k,p,q,f)=fft_output[k]*beta/(double)(uniform_tau_grid.size()-1);
            //trapezoidal rule: subtract 1/2 of first element, add 1/2 of last element
            gomega(k,p,q,f)-=0.5*beta/(double)(uniform_tau_grid.size()-1)*G_tau(0,p,q,f);
            gomega(k,p,q,f)-=0.5*beta/(double)(uniform_tau_grid.size()-1)*G_tau(G_tau.ntime()-1,p,q,f);
          }
        }
      }
    }
    double time3=omp_get_wtime();
    std::cout<<"ft time: "<<time3-time2<<std::endl;
  }
  {
    double time4=omp_get_wtime();
    matsubara_green_function_t gf_slow(gomega);
    std::complex<double> I(0.,1.);
    
    for(std::size_t f=0;f<G_tau.nflavor();++f){
      for(std::size_t p=0;p<G_tau.nsite();++p){
        for(std::size_t q=0;q<G_tau.nsite();++q){
          for(int w=0;w<50;++w){
           gf_slow(w,p,q,f)=0.;
           for(int i=1;i<N_tau-1;++i){
            gf_slow(w,p,q,f)+=std::exp(I*(2.*w+1)*M_PI*G_tau.tau(i)/beta)*G_tau(i,p,q,f)*0.5*(G_tau.tau(i+1)-G_tau.tau(i-1));
           }
            gf_slow(w,p,q,f)+=std::exp(I*(2.*w+1)*M_PI*G_tau.tau(0)/beta)*G_tau(0,p,q,f)*0.5*(G_tau.tau(1)-G_tau.tau(0));
            gf_slow(w,p,q,f)+=std::exp(I*(2.*w+1)*M_PI*G_tau.tau(N_tau-1)/beta)*G_tau(N_tau-1,p,q,f)*0.5*(G_tau.tau(N_tau-1)-G_tau.tau(N_tau-2));
           std::cout<<w<<" "<<gomega(w,0,0,0).real()<<" "<<gomega(w,0,0,0).imag()<<" "<<gf_slow(w,0,0,0).real()<<" "<<gf_slow(w,0,0,0).imag()<<std::endl;
          }       
        }       
      }
    }
    double time3=omp_get_wtime();
    std::cout<<"ft time slow: "<<time3-time4<<std::endl;
  }*/
  {
    double time4=omp_get_wtime();
    std::complex<double> I(0.,1.);
    
    /** run non-equidistant fast Fourier transform */
    nfft_plan plan;
    /** init an one dimensional plan */
    nfft_init_1d(&plan,N_omega,G_tau.tau().size());
    /** copy the times into the nfft container*/
    for(int i=0;i<N_tau;++i){
      plan.x[i]=G_tau.tau(i)/beta;
    }
    /** precompute psi, the entries of the matrix B */
    if(plan.nfft_flags & PRE_ONE_PSI)
      nfft_precompute_one_psi(&plan);
    for(std::size_t f=0;f<G_tau.nflavor();++f){
      for(std::size_t p=0;p<G_tau.nsite();++p){
        for(std::size_t q=0;q<G_tau.nsite();++q){
          for(int i=0;i<N_tau;++i){
            std::complex<double> coeff=G_tau(i,p,q,f)*std::exp(I*M_PI*G_tau.tau(i)/beta)*(
             (i==0)?0.5*(G_tau.tau(1)-G_tau.tau(0))
            :(i==N_tau-1)?0.5*(G_tau.tau(N_tau-1)-G_tau.tau(N_tau-2))
            :0.5*(G_tau.tau(i+1)-G_tau.tau(i-1))
            )*std::exp(N_omega*M_PI*I*G_tau.tau(i)/beta);
            plan.f[i][0]=coeff.real();
            plan.f[i][1]=coeff.imag();
          }
          nfft_adjoint(&plan);
          for(int k=0;k<N_omega;++k){
            std::complex<double> fhat(plan.f_hat[k][0],plan.f_hat[k][1]);
            gomega(k,p,q,f).real()=fhat.real();
            gomega(k,p,q,f).imag()=fhat.imag();
          }
          //std::ofstream gf_nfft_file("/tmp/gf_nfft.dat");
          //for(int k=0;k<N_omega;++k){
          //  gf_nfft_file<<k<<" "<<gomega(k,0,0,0).real()<<" "<<gomega(k,0,0,0).imag()<<" "<<gf_nfft(k,0,0,0).real()<<" "<<gf_nfft(k,0,0,0).imag()<<std::endl;
          //}
        }
      }
    }
    double time3=omp_get_wtime();
    std::cout<<"ft time nfft: "<<time3-time4<<std::endl;
  }
}
/*
 //this algorithm uses the fourth derivative. This is not a good idea if we do not have an equidistant grid, as we do not know the fourth derivative in between spline points.
 void time_to_frequency_ft(const itime_green_function_t & G_tau, matsubara_green_function_t & gomega, double beta){
 std::vector<double> v(G_tau.ntime());
 std::vector<std::complex<double> > v_omega(gomega.nfreq());
 int Np1 = v.size();
 std::vector<double> v2(Np1, 0);
 std::vector<double> tau_vector=G_tau.tau();
 int N = Np1-1;
 int N_omega = v_omega.size();
 
 gomega.c1()=G_tau.c1();
 gomega.c2()=G_tau.c2();
 gomega.c3()=G_tau.c3();
 
 std::vector<double> spline_matrix(Np1* Np1, 0.);
 generate_spline_matrix(spline_matrix, tau_vector, Np1);
 
 {
 
 std::ofstream spline_file("/tmp/splines.dat");
 std::ofstream model_file("/tmp/model.dat");
 std::ofstream gf_file("/tmp/gf.dat");
 
 for(unsigned f=0;f<G_tau.nflavor();++f){
 for(unsigned p=0;p<G_tau.nsite();++p){
 for(unsigned q=0;q<G_tau.nsite();++q){
 for(int tau=0;tau<Np1;++tau){
 v[tau]=G_tau(tau,p,q,f);
 }
 
 // matrix containing the second derivatives y'' of interpolated y=v[tau] at points tau_n
 evaluate_second_derivatives(tau_vector, spline_matrix, v, v2, G_tau.c1(p,q,f), G_tau.c2(p,q,f), G_tau.c3(p,q,f),Np1);
 v_omega.assign(N_omega, 0);
 
 std::vector<std::complex<double> > v_model(N_omega, 0.);
 std::vector<std::complex<double> > v_nomodel(N_omega, 0.);
 
 for (int k=0; k<N_omega; k++) {
 std::complex<double> iw(0, M_PI*(2*k+1)/beta);
 v_omega[k]=(v2[1] - v2[0])/(G_tau.tau(1)-G_tau.tau(0)) + (v2[N] - v2[N-1])/(G_tau.tau(N)-G_tau.tau(N-1)); //G'''(0)+G'''(beta)
 for (int n=1; n<N; n++) {
 double tau=G_tau.tau(n);
 double h=(G_tau.tau(n+1)-G_tau.tau(n))/2.+(G_tau.tau(n)-G_tau.tau(n-1))/2.;
 double deriv_diff=(v2[n+1]-2*v2[n]+v2[n-1])/(h*h);
 v_omega[k] += exp(iw*tau)*deriv_diff*h;
 }
 v_omega[k] *= 1./(iw*iw*iw*iw);
 v_nomodel[k]=v_omega[k];
 v_omega[k] += f_omega(iw, G_tau.c1(p,q,f), G_tau.c2(p,q,f), G_tau.c3(p,q,f));
 gomega(k,p,q,f)=v_omega[k]; //that's the proper convention for the self consistency loop here.
 
 v_model[k]=f_omega(iw, G_tau.c1(p,q,f), G_tau.c2(p,q,f), G_tau.c3(p,q,f));
 }
 for (int j=0; j<Np1-1; j++) {
 // define parameters of spline fit y(tau) = a*y_j + b*y_{j+1} + c*y_j^'' + d*y_{j+1}^''
 double dt=G_tau.tau(j+1)-G_tau.tau(j);
 for(int i=0;i<10;++i){
 double tau=G_tau.tau(j)+i*dt/10.;
 double A = (G_tau.tau(j+1)-tau)/dt;
 double B = 1-A;
 double C = 1./6.*(A*A*A-A)*dt*dt;
 double D = 1./6.*(B*B*B-B)*dt*dt;
 
 spline_file<<tau<<" "<<A*v[j]+B*v[j+1]<<" "<<A*v[j]+B*v[j+1]+C*v2[j]+D*v2[j+1]<<" "<<(v[j+1]-v[j])/dt-(3*A*A-1)/6.*dt*v2[j]+(3*B*B-1)/6.*dt*v2[j+1]<<" "<<A*v2[j]+B*v2[j+1]<<std::endl;
 }
 }
 spline_file<<std::endl;
 for(int i=0;i<N_omega;++i){
 model_file<<i<<" "<<v_nomodel[i].real()<<" "<<v_nomodel[i].imag()<<" "<<v_model[i].real()<<" "<<v_model[i].imag()<<" "<<v_omega[i].real()<<" "<<v_omega[i].imag()<<std::endl;
 }
 model_file<<std::endl;
 }
 }
 }
 gf_file<<std::make_pair(gomega,beta)<<std::endl;
 }
 exit(0);
 }*/
/*void time_to_frequency_ft(const itime_green_function_t &G_tau,  matsubara_green_function_t &G_omega, ft_float_type beta) {
 if(G_tau.nflavor()!=G_omega.nflavor() || G_tau.nsite()!=G_omega.nsite()) throw std::logic_error("GF in tau and omega have different shape.");
 unsigned int N_tau = G_tau.ntime()-1;
 unsigned int N_omega = G_omega.nfreq();
 unsigned int N_site = G_omega.nsite();
 std::cout<<"Ntau is: "<<N_tau<<" Nomega is: "<<N_omega<<std::endl;
 itime_green_function_t G_tau_no_model(G_tau);
 itime_green_function_t G_tau_model(G_tau);
 matsubara_green_function_t G_omega_no_model(G_omega.nfreq(), G_tau.nsite(), G_tau.nflavor());
 matsubara_green_function_t G_omega_model(G_omega.nfreq(), G_tau.nsite(), G_tau.nflavor());
 G_omega.c1()=G_tau.c1();
 G_omega.c2()=G_tau.c2();
 G_omega.c3()=G_tau.c3();
 ft_float_type dtau = beta/(N_tau);
 unsigned int f,s1,s2,k,i;
 ft_float_type tau;
 std::complex<ft_float_type> iw,I;
 #pragma omp parallel for collapse(3) default(none) shared(G_omega_no_model,G_omega,N_tau,N_site,G_tau_model,G_omega_model,G_tau,N_omega,beta,G_tau_no_model) private(k,i,I,iw,tau,s1,s2,f)
 for( f=0;f<G_omega.nflavor();++f){
 for (s1=0; s1<N_site; ++s1){
 for (s2=0; s2<N_site; ++s2) {
 if(G_omega.c1(s1,s2,f)==0 && G_omega.c2(s1,s2,f) == 0 && G_omega.c3(s1,s2,f)){  //nothing happening in this gf.
 for (unsigned int i=0; i<=N_tau; i++) {
 G_omega(i,s1,s2,f)=0.;
 }
 }
 else {
 for (i=0; i<=N_tau; i++) {
 G_tau_no_model(i,s1,s2,f) = G_tau(i,s1,s2,f)-0.;//f_tau(G_tau.tau(i), beta, G_omega.c1(s1,s2,f), G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
 G_tau_model(i,s1,s2,f) = 0.;//f_tau(G_tau.tau(i), beta, G_omega.c1(s1,s2,f), G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
 }
 for (k=0; k<N_omega; k++) {
 iw=std::complex<ft_float_type>(0,(2*k+1)*M_PI/beta);
 I=0.;
 for(i=0;i<=N_tau;++i){
 tau=G_tau.tau(i);
 //std::cout<<i<<" "<<tau<<" "<<iw<<" "<<G_tau.weight(i)<<" "<<std::endl;
 I+=std::exp(iw*tau)*G_tau_no_model(i,s1,s2,f)*G_tau.weight(i);
 }
 //std::cout<<k<<" "<<I<<std::endl;
 G_omega_no_model(k,s1,s2,f)=I;
 }
 //exit(0);
 for (int k=0; k<N_omega; k++) {
 std::complex<ft_float_type> iw(0,(2*k+1)*M_PI/beta);
 ---> reintroduce comment for f omega: G_omega(k,s1,s2,f)=/f_omega(iw, G_omega.c1(s1,s2,f), G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f))+/G_omega_no_model(k,s1,s2,f);
 G_omega_model(k,s1,s2,f)=f_omega(iw, G_omega.c1(s1,s2,f), G_omega.c2(s1,s2,f), G_omega.c3(s1,s2,f));
 }
 }
 }
 }
 }
 { std::ofstream out_file(("/tmp/g_iomega_nomodel.dat")); out_file<<std::make_pair(G_omega_no_model,beta)<<std::endl;}
 { std::ofstream out_file(("/tmp/g_iomega_model.dat")); out_file<<std::make_pair(G_omega_model,beta)<<std::endl;}
 { std::ofstream out_file(("/tmp/g_tau_no_model.dat")); out_file<<G_tau_no_model<<std::endl;}
 { std::ofstream out_file(("/tmp/g_tau.dat")); out_file<<G_tau<<std::endl;}
 { std::ofstream out_file(("/tmp/g_tau_model.dat")); out_file<<G_tau_model<<std::endl;}
 
 }       */
void dft_by_hand(const std::vector<std::complex<double> > &At, std::vector<std::complex<double> > &Aw, double sgn){
  int N=At.size();
  std::complex<double> prefactor=2.*M_PI*std::complex<double>(0,1.)*sgn/(double)N;
  for(std::size_t k=0;k<Aw.size();++k){
    Aw[k]=0;
    for(std::size_t j=0;j<At.size();++j){
      Aw[k]+=At[j]*std::exp(prefactor*(double)(j*k));
    }
  }
}
