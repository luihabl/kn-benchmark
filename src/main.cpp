#include <cstdio>
#include "kn/collisions/mcc.h"


int main() {

	double f = 13.56e6;
	double dt = 1.0 / (400.0 * f);
	double l = 6.7e-2;
	double dx = l / 128.0;
	double ng = 9.64e20;
	double tg = 300.0;
	

	kn::collisions::MonteCarloCollisions::DomainConfig coll_config;
	coll_config.m_dt = dt;
	coll_config.m_m_dx = dx;
	coll_config.m_n_neutral = ng;
	coll_config.m_t_neutral = tg;


	


	
	printf("hello!\\n");
	return 0;
}
