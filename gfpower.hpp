#ifndef GREEN_POWER_H
#define GREEN_POWER_H
#include"green_function.hpp"
void generate_mesh(int power, int uniform, double beta, std::vector<double> &power_mesh, std::vector<double> &simpson_weight);
class green_power:public green_function<double>{
public:
  green_power(int power, int uniform,unsigned int nsite, unsigned int nflavor, double beta)
  :green_function<double>(mesh_size(power,uniform),nsite,nflavor){
    beta_=beta;
    //std::cout<<"creating a power mesh of size: "<<mesh_size(power,uniform)<<std::endl;
    generate_mesh(power,uniform,beta,tau_grid_,weight_);
    //std::cout<<"mesh size is: "<<tau_grid_.size()<<std::endl;
    derivative_zero_.resize(nsite*nsite*nflavor, 0.);
    derivative_beta_.resize(nsite*nsite*nflavor, 0.);
  }
  green_power(unsigned int ntime, unsigned int nsite, unsigned int nflavor, double beta)
  :green_function<double>(ntime,nsite,nflavor){
    generate_mesh(1, (ntime-1)/4, beta_,tau_grid_,weight_);
    beta_=beta;
  }
// AAK 
  void swit(int i,double in) {tau_grid_[i]=in;}
// AAK END
  double tau(int i) const{return tau_grid_[i];}
  const std::vector<double> &tau() const{ return tau_grid_;}
  const double &derivative_zero(int i, int j, int f) const{ return derivative_zero_[i*nsite()*nflavor()+j*nflavor()+f];}
  const double &derivative_beta(int i, int j, int f) const{ return derivative_beta_[i*nsite()*nflavor()+j*nflavor()+f];}
  double &derivative_zero(int i, int j, int f) { return derivative_zero_[i*nsite()*nflavor()+j*nflavor()+f];}
  double &derivative_beta(int i, int j, int f) { return derivative_beta_[i*nsite()*nflavor()+j*nflavor()+f];}
  double weight(int i) const{return weight_[i];}
  static int mesh_size(int power, int uniform){
    return 2*(power+1)*uniform+1;
  }
#ifdef USE_MPI
  void broadcast() const{
    //note the 'const' here which really is just so it can be broadcast as const from the master.
    (const_cast<green_power*>(this))->broadcast();
  }
  void broadcast(){
    green_function<double>::broadcast();
    int grid_size=tau_grid_.size(), weight_size=weight_.size(), derivative_zero_size=derivative_zero_.size(), derivative_beta_size=derivative_beta_.size();
    MPI_Bcast( const_cast<double *>(&(beta_)), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); //we'll need to change the const members...
    MPI_Bcast( &grid_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &weight_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &derivative_zero_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &derivative_beta_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    tau_grid_.resize(grid_size);
    weight_.resize(weight_size);
    derivative_zero_.resize(derivative_zero_size);
    derivative_beta_.resize(derivative_beta_size);
    MPI_Bcast( &(tau_grid_[0]), tau_grid_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast( &(weight_[0]), weight_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast( &(derivative_zero_[0]), derivative_zero_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast( &(derivative_beta_[0]), derivative_beta_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
  friend std::ostream &operator<<(std::ostream &os, const green_power &gf);
  
private:
  std::vector<double> tau_grid_;
  std::vector<double> weight_;
  std::vector<double> derivative_zero_;
  std::vector<double> derivative_beta_;
  double beta_;
};
std::ostream &operator<<(std::ostream &os, const green_power &gf);
typedef green_power itime_green_function_t;

#endif
