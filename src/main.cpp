#include <iostream>
#include "kn/collisions/mcc.h"
#include "rapidcsv.h"

kn::collisions::MonteCarloCollisions::CollisionReaction load_reaction(const char* path, double energy_threshold) {
	
	kn::collisions::MonteCarloCollisions::CollisionReaction coll;

	rapidcsv::Document doc(path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));
	coll.energy = doc.GetColumn<double>(0);
	coll.cross_section = doc.GetColumn<double>(1);
	coll.energy_threshold = energy_threshold;

	return coll;
}

int main() {

	double f = 13.56e6;
	double dt = 1.0 / (400.0 * f);
	double l = 6.7e-2;
	double dx = l / 128.0;
	double ng = 9.64e20;
	double tg = 300.0;
	double m_he = 6.67e-27;
	

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

	return 0;
}
