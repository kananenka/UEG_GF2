#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include"gfpower.hpp"


void generate_mesh(int power, int uniform, double beta, std::vector<double> &power_mesh, std::vector<double> &weights){
  //std::cout<<"generating mesh: p"<<power<<"u"<<uniform<<std::endl;

  std::vector<double> power_points;
  power_points.push_back(0);
  power_mesh.resize(0);
  for(int i=power;i>=0;--i){
    power_points.push_back(beta*0.5*std::pow(2.,-i));
  }
 for(int i=power;i>0;--i){
    power_points.push_back(beta*(1.-0.5*std::pow(2.,-i)));
  }
  power_points.push_back(beta);
 
 
  std::sort(power_points.begin(),power_points.end());
  //for(int i=0;i<power_points.size();++i){
  //  std::cout<<i<<" "<<power_points[i]<<std::endl;
  //}

  if(uniform%2 !=0) throw std::invalid_argument("Simpson weights in power grid: please choose even uniform spacing.");
  for(std::size_t i=0;i<power_points.size()-1;++i){
    for(int j=0;j<uniform;++j){
      double dtau=(power_points[i+1]-power_points[i])/(double)(uniform);
      power_mesh.push_back(power_points[i]+dtau*j);
      if(j==0){
        if(i==0){
          weights.push_back(dtau/3.);
        }else{
          weights.push_back(dtau/3.+(power_points[i]-power_points[i-1])/(3.*(double)(uniform)));
        }
      }else if(j%2==0){
        weights.push_back(2*dtau/3.);
      }else{
        weights.push_back(4*dtau/3.);
      }
    }
  }
  power_mesh.push_back(power_points.back());
  weights.push_back(weights[0]);
  /*std::cout<<"power mesh: "<<std::endl;
  double sum=0.;
  for(int i=0;i<power_mesh.size();++i){
    sum+=weights[i];
    std::cout<<i<<" "<<power_mesh[i]<<" "<<weights[i]<<" "<<sum<<std::endl;
  }*/
  //std::cout<<"mesh size is: "<<power_mesh.size()<<std::endl;
  //exit(1);
}
std::ostream &operator<<(std::ostream &os, const green_power &gf){
  os<<std::setprecision(14);
  for(unsigned int o=0;o<gf.ntime();++o){
    os<<gf.tau_grid_[o]<<" ";
    for(unsigned int i=0;i<gf.nsite();++i)
      for(unsigned int j=0;j<gf.nsite();++j)
        for(unsigned int z=0;z<gf.nflavor();++z)
          os<<gf(o,i,j,z)<<" ";
    os<<std::endl;
  }
  return os;
}

