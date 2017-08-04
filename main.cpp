#include <cmath>
#include <ctime>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <blitz/array.h>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include "complex_bessel.h"
#include "omp.h"
#include "main.hpp"
#include "ueg.hpp"
#include "zero.hpp"
#include "gfimfreq.hpp"
#include "gfimtime.hpp"
#include "vint.cpp"
#include "first.hpp"
#include "second.cpp"
#include "energy.cpp"
#include "gffft.cpp"

int main(int argc, char** argv){
/*
 *  Second-order Green's function theory for homogenious
 *  electron gas.
 *
 */
   double rs = 4.0;   
   double beta = 50.0;
   int iwmax=30000;
   int power=10;
   int uniform=32;
   int nleg = 60;
   //int nleg=atoi(argv[2]);
   int Nel= atoi(argv[1]);
   double kci = atof(argv[2]);
   std::cout << " iwmax = " << iwmax << std::endl;
   std::cout << " power = " << power << std::endl;
   std::cout << " uniform = " << uniform << std::endl;
   std::cout << " itmax = " << (2*(power + 1)*uniform + 1) << std::endl;
   std::cout << " nleg = " << nleg << std::endl;
   double e1b,e2b,etot;
   int readtnl = 0;

   UEG sim (rs,Nel,beta,kci);   
   sim.kmesh();
//
// Zero order:
//
   std::cout << "=========================================================="<<std::endl;
   std::cout << "|                     ZERO ORDER PART:                   |"<<std::endl;
   std::cout << "=========================================================="<<std::endl;
   ZEROTH h0 (sim);
//
// First order:
// 
   std::cout << "=========================================================="<<std::endl;
   std::cout << "|                      FIRST ORDER PART:                 |"<<std::endl;
   std::cout << "=========================================================="<<std::endl;
   GfImFreq Sigmaw(iwmax,sim.ncount,beta);
   FIRST h1(sim,h0);
   GfImFreq GF0w(iwmax,sim.ncount,beta);
   GF0w.GfImFreqHF(sim);
   bisecton(sim,GF0w,Sigmaw);
   GF0w.density_matrix();
   h1.Energy(sim);
   GF0w.GfImFreqPrintGamma();
   GF0w.GenFock(sim,h0);
   e1b = onebodyenergy(sim,GF0w.Dmat,GF0w.Fock,h0.Epsilon);
//
// Second order:
//
   std::cout<<"=========================================================="<<std::endl;
   std::cout<<"|                     SECOND ORDER PART:                 |"<<std::endl;
   std::cout<<"=========================================================="<<std::endl;

   GfImFreq GFc(iwmax,sim.ncount,beta);

   Array<int,2> List2vD;
   Array<int,2> List2vS;
   Array<int,3> List3vD;

   std::cout << " Creating vector lists..." << std::endl;
   vectorListDN(sim.Kvecs,List2vD,sim.ncount);
   std::cout << "1/3..." << std::endl;
   vectorListSN(sim.Kvecs,List2vS,sim.ncount);
   std::cout << "2/3..." << std::endl;
   vectorListDD(sim.Kvecs,List3vD,sim.ncount);
   std::cout << "3/3...done." << std::endl;

   Array<std::complex<double>,2> Tnl;
   std::cout << " Generating Tnl..." <<std::endl;
   ut_Tnl(iwmax,nleg,Tnl,readtnl);

   GfImTime GF0t(power,uniform,sim.ncount,beta);
   GfImTime Sigmad2(power,uniform,sim.ncount,beta);
   GfImTime Sigmax2(power,uniform,sim.ncount,beta);
   GfImTime Sigma2(power,uniform,sim.ncount,beta);

   std::cout << " FFT: G(iw) -> G(t)... " << std::endl;
   frequency_to_time(GF0w,GF0t);

   GfImFreq GFw(iwmax,sim.ncount,beta);

   std::cout << " Calculating second-order direct diagram..." << std::endl;
   Direct2(Sigmad2,GF0t,sim,List2vD,List2vS);
   std::cout << " Calculating second-order exchange diagram..." << std::endl;
   Exchange2(Sigmax2,GF0t,sim,List3vD,List2vD);
   Sigma2All(Sigma2,Sigmad2,Sigmax2);
   std::cout << " FFT: Sigma(t) -> Sigma(iw) ..." << std::endl;
   //time_to_frequency3(Sigmaw,Sigma2,power,uniform);
   time_to_frequency_legendre(Sigmaw,Sigma2,Tnl,power,uniform,nleg);
//
// Print sigma
//
   std::ofstream grfile("Sigma2t.dat");
   for(int t=0;t<Sigmad2.itmax;++t)
      grfile << setprecision(8) << Sigmad2.itimes(t) << "  " << Sigmad2.gf(0,t) << "  " << Sigmax2.gf(0,t) << std::endl;
   grfile.close();
//
// Write Sigma(iw) into a file:
//
   std::ofstream s1file("Sigma2w.dat");
   for(int w=0;w<iwmax;++w)
      s1file << Sigmaw.omegas(w) << " " << Sigmaw.gf(0,w).real() << " " << Sigmaw.gf(0,w).imag() << std::endl;  
   s1file.close();

   bisecton(sim,GFc,Sigmaw);
   GFc.density_matrix();
   GFc.GenFock(sim,h0);
//
// Print density matrix:
//
   std::ofstream f1file("Dmat2.dat");
   for(int ind=0;ind<sim.ncount;++ind){
   f1file <<ind <<  " \t " << sim.Kvecs(ind,0)/sim.pref
                <<  " \t " << sim.Kvecs(ind,1)/sim.pref
                <<  " \t " << sim.Kvecs(ind,2)/sim.pref
                <<  " \t " << GFc.Dmat(ind) << std::endl;
   }
   f1file.close();
   std::cout << " GF2 populations are written to Dmat2.dat " << std::endl;
//
// Print Green's function
//
   std::ofstream ggfile("Gf2w.dat");
   for(int w=0;w<iwmax;++w)
       ggfile << GFc.omegas(w) << " " << GFc.gf(0,w).real() << " " << GFc.gf(0,w).imag() << std::endl;
   ggfile.close();
//
// Calculate energies:
//
   std::cout << " MP2 energy: " << std::endl;
   e2b = twobodyenergy(Sigmaw,GF0w,sim,1.0);

   std::cout << " GF2 energy: " << std::endl;
   e1b = onebodyenergy(sim,GFc.Dmat,GFc.Fock,h0.Epsilon);
   e2b = twobodyenergy(Sigmaw,GFc,sim,2.0);
   etot = e1b + e2b;
   std::cout << " Total energy: = " << etot << std::endl;
}

