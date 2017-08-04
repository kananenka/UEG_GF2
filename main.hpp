
using namespace blitz;

double Vint(Array<double,2> Kvecs,int one,int two,double sign);
void kmesh(double kc,double length,Array<double,2> &Kvecs,
           int ncount);
double EnergyFock(Array<double,1> Fock,Array<double,1> Dmat,int ncount,double pref);

