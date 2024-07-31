#include <iostream>
#include "kn/collisions/mcc.h"
#include "kn/constants/constants.h"
#include "kn/electromagnetics/poisson.h"
#include "kn/interpolate/field.h"
#include "kn/interpolate/weight.h"
#include "kn/particle/boundary.h"
#include "kn/particle/pusher.h"
#include "kn/particle/species.h"
#include "kn/random/random.h"
#include "kn/spatial/grid.h"
#include "rapidcsv.h"
#include <fstream>

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
		double vth = std::sqrt(kn::constants::kb * t / m);
		v = { 
			kn::random::normal(0.0, vth),
			kn::random::normal(0.0, vth),
			kn::random::normal(0.0, vth)
		};
	};
}

int main() {

	kn::random::initialize(42);

	size_t nx = 129;
	double f = 13.56e6;
	double dt = 1.0 / (400.0 * f);
	double l = 6.7e-2;
	double dx = l / static_cast<double>(nx - 1);
	double ng = 9.64e20;
	double tg = 300.0;
	double te = 30'000.0;
	double ti = 300.0;
	double n0 = 2.56e14;
	double m_he = 6.67e-27;
	double volt = 450.0;
	size_t ppc = 512;
	size_t n_steps = 512'000;
	double particle_weight = n0 * l / (double)(ppc * nx);

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

	auto electron_density = kn::spatial::UniformGrid(l, nx);
	auto ion_density = kn::spatial::UniformGrid(l, nx);
	auto rho = kn::spatial::UniformGrid(l, nx);

	auto poisson_solver = kn::electromagnetics::DirichletPoissonSolver(nx, dx);
	auto phi = kn::spatial::UniformGrid(l, nx);
	auto efield = kn::spatial::UniformGrid(l, nx);

	std::cout << "starting" << std::endl;

	double time = 0.0;
	for(size_t i = 0; i < n_steps / 100; i++) {
		
		kn::interpolate::weight_to_grid(electrons, electron_density);
		kn::interpolate::weight_to_grid(ions, ion_density);
		
		kn::electromagnetics::charge_density(particle_weight, ion_density, electron_density, rho);

		double vc = volt * std::sin(2.0 * kn::constants::pi * f * time);

		poisson_solver.solve(rho.data(), phi.data(), 0.0, vc);
		poisson_solver.efield(phi.data(), efield.data());

		kn::interpolate::field_at_particles(efield, electrons);
		kn::interpolate::field_at_particles(efield, ions);

		kn::particle::move_particles(electrons, dt);
		kn::particle::move_particles(ions, dt);

		kn::particle::apply_absorbing_boundary(electrons, 0, l);
		kn::particle::apply_absorbing_boundary(ions, 0, l);

		coll.collide_electrons(electrons, ions);
		coll.collide_ions(ions);

		if(i % 100 == 0) {
			std::cout << "--" << std::endl;
			std::cout << "i: " << i << std::endl;
			std::cout << "electrons: " << electrons.n() << std::endl;
			std::cout << "ions: " << ions.n() << std::endl << std::endl;
		}

		time += dt;
	}


	// auto solver = kn::electromagnetics::DirichletPoissonSolver(nx, dx);

	// auto rho = std::vector<double>(nx, n0 * kn::constants::e);
	// auto phi = std::vector<double>(nx);

	// solver.solve(rho, phi, 0.0, 0.0);

	// auto efield = std::vector<double>(nx);
	// solver.efield(phi, efield);

	// std::ofstream outf ("output.txt");

	// for (size_t i = 0; i < phi.size(); i++) 
	// { 
	// 	if (i!=0) { outf << "\n"; } 
	// 	outf << efield[i]; 
	// }

	return 0;
}
