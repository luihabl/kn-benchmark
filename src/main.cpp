#include <iostream>
#include "kn/collisions/mcc.h"
#include "kn/constants/constants.h"
#include "kn/particle/species.h"
#include "kn/random/random.h"
#include "rapidcsv.h"

kn::collisions::MonteCarloCollisions::CollisionReaction load_reaction(const char* path, double energy_threshold) {
	
	kn::collisions::MonteCarloCollisions::CollisionReaction coll;

	rapidcsv::Document doc(path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));
	coll.energy = doc.GetColumn<double>(0);
	coll.cross_section = doc.GetColumn<double>(1);
	coll.energy_threshold = energy_threshold;

	return coll;
}

auto maxwellian_emitter(double t, double l, double m) {
	return [l, t, m](kn::particle::ChargedSpecies1D3V::Vec3& v, double& x){
		x = l * kn::random::uniform();
		double vth = std::sqrt(kn::constants::e * t / m);
		v = { 
			kn::random::normal(0.0, vth),
			kn::random::normal(0.0, vth),
			kn::random::normal(0.0, vth)
		};
	};
}

int main() {

	size_t nx = 129;
	double f = 13.56e6;
	double dt = 1.0 / (400.0 * f);
	double l = 6.7e-2;
	double dx = l / static_cast<double>(nx - 1);
	double ng = 9.64e20;
	double tg = 300.0;
	double te = 30000.0;
	double ti = 300.0;
	double n0 = 2.56e14;
	double m_he = 6.67e-27;
	size_t ppc = 512;
	

	kn::collisions::MonteCarloCollisions::DomainConfig coll_config;
	coll_config.m_dt = dt;
	coll_config.m_m_dx = dx;
	coll_config.m_n_neutral = ng;
	coll_config.m_t_neutral = tg;
	coll_config.m_m_ion = m_he;

	auto el_cs = load_reaction("../data/Elastic_He.csv", 0.0);
	auto exc_cs = std::vector<kn::collisions::MonteCarloCollisions::CollisionReaction>{
		load_reaction("../data/Excitation1_He.csv", 19.82),
		load_reaction("../data/Excitation2_He.csv", 20.61)
	};
	auto iz_cs = load_reaction("../data/Ionization_He.csv", 24.59);
	auto iso_cs = load_reaction("../data/Isotropic_He.csv", 0.0);
	auto bs_cs = load_reaction("../data/Backscattering_He.csv", 0.0);

	auto coll = kn::collisions::MonteCarloCollisions(
		coll_config,
		std::move(el_cs),
		std::move(exc_cs),
		std::move(iz_cs),
		std::move(iso_cs),
		std::move(bs_cs)
	);

	size_t n_initial = (nx - 1) * ppc;
	
	auto electrons = kn::particle::ChargedSpecies1D3V(-kn::constants::e, kn::constants::m_e);
	electrons.add(n_initial, maxwellian_emitter(te, l, kn::constants::m_e));
	
	auto ions = kn::particle::ChargedSpecies1D3V(kn::constants::e, m_he);
	ions.add(n_initial, maxwellian_emitter(ti, l, m_he));


	





	return 0;
}
