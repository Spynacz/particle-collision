#include <omp.h>

#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/Window/Window.hpp>
#include <cmath>
#include <map>
#include <vector>

#include "Particle.hpp"

using namespace std;

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1200

// >= 2x max radius
#define GRID_CELL_SIZE 50

typedef std::vector<size_t> GridCell;
typedef std::map<int64_t, GridCell> ParticleGrid;

int64_t getCellKey(int x, int y) {
    return (static_cast<int64_t>(x) << 32) | static_cast<int64_t>(y);
}

void handle_wall_collisions(const sf::Window& window,
                            std::vector<Particle>& particles) {
    int window_width = window.getSize().x;
    int window_height = window.getSize().y;

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& particle = particles[i];
        if (particle.x > window_width - particle.radius) {
            particle.x = window_width - particle.radius;
            particle.vx = -particle.vx;
        } else if (particle.x < particle.radius) {
            particle.x = particle.radius;
            particle.vx = -particle.vx;
        }

        if (particle.y > window_height - particle.radius) {
            particle.y = window_height - particle.radius;
            particle.vy = -particle.vy;
        } else if (particle.y < particle.radius) {
            particle.y = particle.radius;
            particle.vy = -particle.vy;
        }
    }
}

void handle_single_collision(Particle& p1, Particle& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double dist = sqrt(dx * dx + dy * dy);

    if (dist < p1.radius + p2.radius && dist > 0) {
        double nx = dx / dist;
        double ny = dy / dist;

        double dvx = p2.vx - p1.vx;
        double dvy = p2.vy - p1.vy;

        double dvn = dvx * nx + dvy * ny;

        if (dvn > 0) return;

        double impulse = 2 * dvn / (p1.mass + p2.mass);
        p1.vx += impulse * p2.mass * nx;
        p1.vy += impulse * p2.mass * ny;
        p2.vx -= impulse * p1.mass * nx;
        p2.vy -= impulse * p1.mass * ny;

        // Separate overlapping particles
        double totalMass = p1.mass + p2.mass;
        double overlap = 0.5 * (p1.radius + p2.radius - dist);
        p1.x -= overlap * (p2.mass / totalMass) * nx;
        p1.y -= overlap * (p2.mass / totalMass) * ny;
        p2.x += overlap * (p1.mass / totalMass) * nx;
        p2.y += overlap * (p1.mass / totalMass) * ny;
    }
}

void handle_ball_collisions_omp(std::vector<Particle>& particles,
                                ParticleGrid& grid,
                                std::vector<omp_lock_t>& particle_locks) {
    // grid population not parallelized
    grid.clear();
    for (size_t i = 0; i < particles.size(); ++i) {
        int cellX = static_cast<int>(particles[i].x / GRID_CELL_SIZE);
        int cellY = static_cast<int>(particles[i].y / GRID_CELL_SIZE);
        grid[getCellKey(cellX, cellY)].push_back(i);
    }

    // collisions parallel
    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p1 = particles[i];
        int cellX = static_cast<int>(p1.x / GRID_CELL_SIZE);
        int cellY = static_cast<int>(p1.y / GRID_CELL_SIZE);

        // iterate over p1's cell and its 8 neighbors
        for (int y = cellY - 1; y <= cellY + 1; ++y) {
            for (int x = cellX - 1; x <= cellX + 1; ++x) {
                // find the neighboring cell
                auto it = grid.find(getCellKey(x, y));
                if (it == grid.end()) {
                    continue;  // no particles in this cell
                }

                // check against all particles (p2) in this neighboring cell
                GridCell& cell = it->second;
                for (size_t j_idx : cell) {
                    if (i >= j_idx) {
                        continue;
                    }

                    Particle& p2 = particles[j_idx];

                    // lock in a consistent order to prevent deadlock
                    omp_set_lock(&particle_locks[i]);
                    omp_set_lock(&particle_locks[j_idx]);

                    handle_single_collision(p1, p2);

                    // unlock in reverse order
                    omp_unset_lock(&particle_locks[j_idx]);
                    omp_unset_lock(&particle_locks[i]);
                }
            }
        }
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
                            "Particle collisions");

    vector<Particle> particles;
    vector<omp_lock_t> particle_locks;

    for (int i = 0; i < 300; i++) {
        Particle particle = Particle();
        particle.randomize(window.getSize().x, window.getSize().y, 5);
        particle.id = i;
        particles.push_back(particle);

        omp_lock_t lock;
        omp_init_lock(&lock);
        particle_locks.push_back(lock);
    }

    ParticleGrid grid;

    // Start the game loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::Resized) {
                sf::FloatRect visibleArea(0, 0, event.size.width,
                                          event.size.height);
                window.setView(sf::View(visibleArea));
                handle_wall_collisions(window, particles);
            }
        }
        // clang-format off
        #pragma omp parallel for
        // clang-format on 
        for (size_t i = 0; i < particles.size(); ++i) {
            particles[i].update(0.1);
        }

        handle_wall_collisions(window, particles);
        handle_ball_collisions_omp(particles, grid, particle_locks);

        window.clear();

        for (auto& particle : particles) {
            sf::CircleShape shape(particle.radius);
            shape.setOrigin(shape.getRadius(), shape.getRadius());
            shape.move(particle.x, particle.y);
            shape.setFillColor(particle.color);
            window.draw(shape);
        }

        // Update the window
        window.display();
    }

    // OMP: Clean up locks
    for (auto& lock : particle_locks) {
        omp_destroy_lock(&lock);
    }

    return EXIT_SUCCESS;
}
