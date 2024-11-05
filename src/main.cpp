#include <fstream>
#include <iostream>
#include <span>
#include <vector>

#include "kn/collisions/mcc.h"
#include "kn/collisions/reaction.h"
#include "kn/collisions/reactions/he_reactions.h"
#include "kn/collisions/target.h"
#include "kn/constants/constants.h"
#include "kn/core/vec.h"
#include "kn/electromagnetics/poisson.h"
#include "kn/interpolate/field.h"
#include "kn/interpolate/weight.h"
#include "kn/particle/boundary.h"
#include "kn/particle/pusher.h"
#include "kn/particle/species.h"
#include "kn/random/random.h"
#include "kn/spatial/grid.h"
#include "rapidcsv.h"

void save_vec(const char* filename, const std::span<double>& v) {
    std::ofstream outf(filename);

    for (size_t i = 0; i < v.size(); i++) {
        if (i != 0) {
            outf << "\n";
        }
        outf << v[i];
    }
}

void save_vec(const char* filename, const std::span<kn::core::Vec<3>>& v) {
    std::ofstream outf(filename);

    for (size_t i = 0; i < v.size(); i++) {
        if (i != 0) {
            outf << "\n";
        }
        outf << v[i].x << "," << v[i].y << "," << v[i].z;
    }
}

void save_vec(const char* filename, const std::span<kn::core::Vec<1>>& v) {
    std::ofstream outf(filename);

    for (size_t i = 0; i < v.size(); i++) {
        if (i != 0) {
            outf << "\n";
        }
        outf << v[i].x;
    }
}

// kn::collisions::CollisionReaction load_reaction(const char* path,
//                                                 double energy_threshold,
//                                                 kn::collisions::CollisionType ctype,
//                                                 kn::collisions::CollisionProjectile projectile) {
//     kn::collisions::CollisionReaction coll;
//
//     rapidcsv::Document doc(path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));
//     coll.energy = doc.GetColumn<double>(0);
//     coll.cross_section = doc.GetColumn<double>(1);
//     coll.energy_threshold = energy_threshold;
//     coll.projectile = projectile;
//     coll.type = ctype;
//
//     return coll;
// }

kn::collisions::CrossSection load_cross_section(const char* path, double energy_threshold) {
    kn::collisions::CrossSection cs;
    rapidcsv::Document doc(path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(';'));
    cs.energy = doc.GetColumn<double>(0);
    cs.cross_section = doc.GetColumn<double>(1);
    cs.threshold = energy_threshold;

    return cs;
}

auto maxwellian_emitter(double t, double l, double m) {
    return [l, t, m](kn::core::Vec<3>& v, kn::core::Vec<1>& x) {
        x.x = l * kn::random::uniform();
        double vth = std::sqrt(kn::constants::kb * t / m);
        v = {kn::random::normal(0.0, vth), kn::random::normal(0.0, vth),
             kn::random::normal(0.0, vth)};
    };
}

int main() {
    kn::random::initialize(500);

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
    double particle_weight = n0 * l / static_cast<double>(ppc * (nx - 1));

    size_t n_initial = (nx - 1) * ppc;

    auto electrons = kn::particle::ChargedSpecies<1, 3>(-kn::constants::e, kn::constants::m_e);
    electrons.add(n_initial, maxwellian_emitter(te, l, kn::constants::m_e));
    auto ions = kn::particle::ChargedSpecies<1, 3>(kn::constants::e, m_he);
    ions.add(n_initial, maxwellian_emitter(ti, l, m_he));

    auto electron_density = kn::spatial::UniformGrid(l, nx);
    auto av_electron_density = kn::spatial::AverageGrid(electron_density);
    auto ion_density = kn::spatial::UniformGrid(l, nx);
    auto av_ion_density = kn::spatial::AverageGrid(ion_density);
    auto rho = kn::spatial::UniformGrid(l, nx);

    auto poisson_solver = kn::electromagnetics::DirichletPoissonSolver(nx, dx);
    auto phi = kn::spatial::UniformGrid(l, nx);
    auto efield = kn::spatial::UniformGrid(l, nx);

    // Electron-neutral collisions
    kn::collisions::Reactions<1, 3> electron_reactions;
    electron_reactions.push_back(
        std::make_unique<kn::collisions::reactions::HeElectronElasticCollision<1, 3>>(
            kn::collisions::reactions::HeCollisionConfig{m_he},
            load_cross_section("../data/Elastic_He.csv", 0.0)));

    electron_reactions.push_back(
        std::make_unique<kn::collisions::reactions::HeExcitationCollision<1, 3>>(
            kn::collisions::reactions::HeCollisionConfig{m_he},
            load_cross_section("../data/Excitation1_He.csv", 19.82)));

    electron_reactions.push_back(
        std::make_unique<kn::collisions::reactions::HeExcitationCollision<1, 3>>(
            kn::collisions::reactions::HeCollisionConfig{m_he},
            load_cross_section("../data/Excitation2_He.csv", 20.61)));

    electron_reactions.push_back(
        std::make_unique<kn::collisions::reactions::HeIonizationCollision<1, 3>>(
            ions, tg, kn::collisions::reactions::HeCollisionConfig{m_he},
            load_cross_section("../data/Ionization_He.csv", 24.59)));

    kn::collisions::ReactionConfig<1, 3> electron_reaction_config{
        dt, dx, std::make_unique<kn::collisions::StaticUniformTarget<1, 3>>(ng, tg),
        std::move(electron_reactions), kn::collisions::RelativeDynamics::FastProjectile};

    auto electron_collisions =
        kn::collisions::MCCReactionSet(electrons, std::move(electron_reaction_config));

    // Ion-neutral collisions
    kn::collisions::Reactions<1, 3> ion_reactions;
    ion_reactions.push_back(
        std::make_unique<kn::collisions::reactions::HeIonElasticCollision<1, 3>>(
            kn::collisions::reactions::HeCollisionConfig{m_he},
            load_cross_section("../data/Isotropic_He.csv", 0.0)));

    ion_reactions.push_back(
        std::make_unique<kn::collisions::reactions::HeIonChargeExchangeCollision<1, 3>>(
            kn::collisions::reactions::HeCollisionConfig{m_he},
            load_cross_section("../data/Backscattering_He.csv", 0.0)));

    kn::collisions::ReactionConfig<1, 3> ion_reaction_config{
        dt, dx, std::make_unique<kn::collisions::StaticUniformTarget<1, 3>>(ng, tg),
        std::move(ion_reactions), kn::collisions::RelativeDynamics::SlowProjectile};

    auto ion_collisions = kn::collisions::MCCReactionSet(ions, std::move(ion_reaction_config));

    // Main loop
    std::cout << "starting" << std::endl;

    for (size_t i = 0; i < n_steps; i++) {
        kn::interpolate::weight_to_grid(electrons, electron_density);
        kn::interpolate::weight_to_grid(ions, ion_density);

        kn::electromagnetics::charge_density(particle_weight, ion_density, electron_density, rho);

        double vc = volt * std::sin(2.0 * kn::constants::pi * f * dt * static_cast<double>(i));

        poisson_solver.solve(rho.data(), phi.data(), 0.0, vc);
        poisson_solver.efield(phi.data(), efield.data());

        kn::interpolate::field_at_particles(efield, electrons);
        kn::interpolate::field_at_particles(efield, ions);

        kn::particle::move_particles(electrons, dt);
        kn::particle::move_particles(ions, dt);

        kn::particle::apply_absorbing_boundary(electrons, 0, l);
        kn::particle::apply_absorbing_boundary(ions, 0, l);

        electron_collisions.react_all();
        ion_collisions.react_all();

        if (i > (n_steps - 12'800)) {
            av_electron_density.add(electron_density);
            av_ion_density.add(ion_density);
        }

        // if(i > 50) {
        //     save_vec("pos_e.txt", std::vector<double>(electrons.x(), electrons.x() +
        //     electrons.n())); save_vec("v_e.txt", std::vector<kn::core::Vec3>(electrons.v(),
        //     electrons.v() + electrons.n())); save_vec("v_i.txt",
        //     std::vector<kn::core::Vec3>(ions.v(), ions.v() + ions.n())); save_vec("pos_i.txt",
        //     std::vector<double>(ions.x(), ions.x() + ions.n())); save_vec("field_e.txt",
        //     std::vector<double>(electrons.f(), electrons.f() + electrons.n()));
        //     save_vec("field_i.txt", std::vector<double>(ions.f(), ions.f() + ions.n()));
        //     save_vec("density_e.txt", electron_density.data());
        //     save_vec("density_i.txt", ion_density.data());
        //     save_vec("rho.txt", rho.data());
        //     save_vec("phi.txt", phi.data());
        //     save_vec("efield.txt", efield.data());
        // return 0;

        // }

        if (i % 1000 == 0) {
            std::cout << "--" << std::endl;
            std::cout << "i: " << i << std::endl;
            // std::cout << "electrons: " << electrons.n() << std::endl;
            // std::cout << "ions: " << ions.n() << std::endl << std::endl;

            printf("e: %lu\t i: %lu\n", electrons.n(), ions.n());
        }
    }

    save_vec("pos_e.txt", std::span(electrons.x(), electrons.n()));
    save_vec("v_e.txt", std::span(electrons.v(), +electrons.n()));
    save_vec("v_i.txt", std::span(ions.v(), ions.n()));
    save_vec("pos_i.txt", std::span(ions.x(), ions.n()));
    save_vec("field_e.txt", std::span(electrons.f(), electrons.n()));
    save_vec("field_i.txt", std::span(ions.f(), ions.n()));
    save_vec("density_e.txt", av_electron_density.get());
    save_vec("density_i.txt", av_ion_density.get());
    save_vec("rho.txt", rho.data());
    save_vec("phi.txt", phi.data());
    save_vec("efield.txt", efield.data());

    // auto solver = kn::electromagnetics::DirichletPoissonSolver(nx, dx);

    // auto rho = std::vector<double>(nx, n0 * kn::constants::e);
    // auto phi = std::vector<double>(nx);

    // solver.solve(rho, phi, 0.0, 0.0);

    // auto efield = std::vector<double>(nx);
    // solver.efield(phi, efield);

    // std::ofstream outf ("output.txt");

    // for (size_t i = 0; i < phi.n(); i++)
    // {
    //     if (i!=0) { outf << "\n"; }
    //     outf << phi.data()[i];
    // }

    return 0;
}
