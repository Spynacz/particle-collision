#include <omp.h>

#include <SFML/Graphics.hpp>
#include <chrono>  // For high-precision timing
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

#include "CudaParticle.h"
#include "Particle.hpp"

using namespace std;

#define INITIAL_WIDTH 1600
#define INITIAL_HEIGHT 900
#define GRID_CELL_SIZE 50

typedef std::vector<size_t> GridCell;
typedef std::map<int64_t, GridCell> ParticleGrid;

struct Config {
    int num_particles = 500;
    int num_frames = 1000;
    bool use_naive = false;
    bool use_cuda = false;
    bool render = false;
    int particle_size = 5;
    bool velocity_color = true;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -n, --count <int>    Number of particles (default: 500)\n"
              << "  -f, --frames <int>   Number of frames (default: 1000)\n"
              << "  -s, --size <int>     Particle size (default: 5)\n"
              << "  --naive              Use Naive O(N^2) algorithm\n"
              << "  --cuda               Use CUDA implementation\n"
              << "  --render             Enable rendering\n"
              << "  --no-color           Disable velocity coloring\n"
              << "  -h, --help           Show this help message\n"
              << std::endl;
}

Config parse_args(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "-n" || arg == "--count") {
            if (i + 1 < argc) {
                config.num_particles = std::stoi(argv[++i]);
            } else {
                std::cerr << "--count requires an argument\n";
                std::exit(1);
            }
        } else if (arg == "-f" || arg == "--frames") {
            if (i + 1 < argc) {
                config.num_frames = std::stoi(argv[++i]);
            } else {
                std::cerr << "--frames requires an argument\n";
                std::exit(1);
            }
        } else if (arg == "-s" || arg == "--size") {
            if (i + 1 < argc) {
                config.particle_size = std::stoi(argv[++i]);
            }
        } else if (arg == "--naive") {
            config.use_naive = true;
        } else if (arg == "--cuda") {
            config.use_cuda = true;
        } else if (arg == "--render") {
            config.render = true;
        } else if (arg == "--no-color") {
            config.velocity_color = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return config;
}

extern "C" void run_cuda_simulation(CudaParticle* host_particles,
                                    int num_particles, float dt, int width,
                                    int height);

int64_t getCellKey(int x, int y) {
    return (static_cast<int64_t>(x) << 32) | static_cast<int64_t>(y);
}

void handle_wall_collisions(int width, int height,
                            std::vector<Particle>& particles) {
    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& particle = particles[i];
        if (particle.x > width - particle.radius) {
            particle.x = width - particle.radius;
            particle.vx = -particle.vx;
        } else if (particle.x < particle.radius) {
            particle.x = particle.radius;
            particle.vx = -particle.vx;
        }
        if (particle.y > height - particle.radius) {
            particle.y = height - particle.radius;
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

        double totalMass = p1.mass + p2.mass;
        double overlap = 0.5 * (p1.radius + p2.radius - dist);
        p1.x -= overlap * (p2.mass / totalMass) * nx;
        p1.y -= overlap * (p2.mass / totalMass) * ny;
        p2.x += overlap * (p1.mass / totalMass) * nx;
        p2.y += overlap * (p1.mass / totalMass) * ny;
    }
}

// NAIVE ALGORITHM (O(N^2)) - Serial Only for comparison
void handle_collisions_naive(std::vector<Particle>& particles) {
    for (size_t i = 0; i < particles.size(); i++) {
        for (size_t j = i + 1; j < particles.size(); j++) {
            handle_single_collision(particles[i], particles[j]);
        }
    }
}

// GRID ALGORITHM (O(N)) - Parallelizable
void handle_collisions_grid_omp(std::vector<Particle>& particles,
                                ParticleGrid& grid,
                                std::vector<omp_lock_t>& locks) {
    // 1. Clear and Populate Grid (Serial overhead)
    grid.clear();
    for (size_t i = 0; i < particles.size(); ++i) {
        int cellX = static_cast<int>(particles[i].x / GRID_CELL_SIZE);
        int cellY = static_cast<int>(particles[i].y / GRID_CELL_SIZE);
        grid[getCellKey(cellX, cellY)].push_back(i);
    }

// 2. Parallel Collision Checks
// clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p1 = particles[i];
        int cellX = static_cast<int>(p1.x / GRID_CELL_SIZE);
        int cellY = static_cast<int>(p1.y / GRID_CELL_SIZE);

        for (int y = cellY - 1; y <= cellY + 1; ++y) {
            for (int x = cellX - 1; x <= cellX + 1; ++x) {
                auto it = grid.find(getCellKey(x, y));
                if (it == grid.end()) continue;

                for (size_t j_idx : it->second) {
                    if (i >= j_idx)
                        continue;  // Avoid double check and self-check

                    omp_set_lock(&locks[i]);
                    omp_set_lock(&locks[j_idx]);
                    handle_single_collision(p1, particles[j_idx]);
                    omp_unset_lock(&locks[j_idx]);
                    omp_unset_lock(&locks[i]);
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.use_naive && cfg.use_cuda) {
        std::cerr
            << "Warning: Both Naive and CUDA selected. Defaulting to CUDA.\n";
        cfg.use_naive = false;
    }

    std::cout << "--- Simulation Configuration ---\n"
              << "Particles: " << cfg.num_particles << "\n"
              << "Frames:    " << cfg.num_frames << "\n"
              << "Algorithm: "
              << (cfg.use_cuda ? "CUDA"
                               : (cfg.use_naive ? "Naive O(N^2)" : "Grid O(N)"))
              << "\n"
              << "Render:    " << (cfg.render ? "ON" : "OFF") << "\n"
              << "Ball Size: " << cfg.particle_size << "\n"
              << "--------------------------------" << std::endl;

    // --- Init ---
    sf::RenderWindow window;
    if (cfg.render) {
        window.create(sf::VideoMode(INITIAL_WIDTH, INITIAL_HEIGHT),
                      "Benchmarks");
        window.setFramerateLimit(60);
    }

    srand(42);  // Fixed seed for consistent benchmarks
    vector<Particle> particles;
    vector<omp_lock_t> locks;
    ParticleGrid grid;

    for (int i = 0; i < cfg.num_particles; i++) {
        Particle p;
        p.randomize(window.getSize().x, window.getSize().y, cfg.particle_size);
        p.id = i;
        particles.push_back(p);

        omp_lock_t lock;
        omp_init_lock(&lock);
        locks.push_back(lock);
    }

    // --- Benchmark Loop ---
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (frame_count < cfg.num_frames) {
        // SFML Event Handling (Keep window responsive if rendering)
        if (cfg.render) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window.close();
                    return 0;
                }
                if (event.type == sf::Event::Resized) {
                    sf::FloatRect visibleArea(0, 0, event.size.width,
                                              event.size.height);
                    window.setView(sf::View(visibleArea));
                    handle_wall_collisions(window.getSize().x,
                                           window.getSize().y, particles);
                }
            }
        }

        // --- PHYSICS ---
        if (cfg.use_naive) {
            for (auto& p : particles) {
                p.update(0.1);
                if (cfg.velocity_color) {
                    p.updateColor((p.vx * p.vx + p.vy * p.vy) / 100.0);
                }
            }
            handle_wall_collisions(window.getSize().x, window.getSize().y,
                                   particles);
            handle_collisions_naive(particles);
        } else if (cfg.use_cuda) {
            // 1. Convert Particle -> CudaParticle
            std::vector<CudaParticle> c_particles(particles.size());
            for (size_t i = 0; i < particles.size(); ++i) {
                c_particles[i].x = (float)particles[i].x;
                c_particles[i].y = (float)particles[i].y;
                c_particles[i].vx = (float)particles[i].vx;
                c_particles[i].vy = (float)particles[i].vy;
                c_particles[i].radius = (float)particles[i].radius;
                c_particles[i].mass = (float)particles[i].mass;
            }

            // 2. Run GPU Simulation
            run_cuda_simulation(c_particles.data(), c_particles.size(), 0.1f,
                                window.getSize().x, window.getSize().y);

            ColorGradient gradient;
            gradient.viridisHeatMap();
            // 3. Convert CudaParticle -> Particle
            for (size_t i = 0; i < particles.size(); ++i) {
                particles[i].x = c_particles[i].x;
                particles[i].y = c_particles[i].y;
                particles[i].vx = c_particles[i].vx;
                particles[i].vy = c_particles[i].vy;

                if (cfg.velocity_color) {
                    float r, g, b;
                    float velocity_magnitude =
                        (particles[i].vx * particles[i].vx +
                         particles[i].vy * particles[i].vy);
                    gradient.getColorAtValue(velocity_magnitude / 100, r, g, b);
                    particles[i].color =
                        sf::Color(static_cast<sf::Uint8>(r * 255),
                                  static_cast<sf::Uint8>(g * 255),
                                  static_cast<sf::Uint8>(b * 255));
                }
            }
        } else {
// Grid/OMP version
// clang-format off
            #pragma omp parallel for
            // clang-format on
            for (size_t i = 0; i < particles.size(); ++i) {
                particles[i].update(0.1);
                if (cfg.velocity_color) {
                    particles[i].updateColor(
                        (particles[i].vx * particles[i].vx +
                         particles[i].vy * particles[i].vy) /
                        100.0);
                }
            }

            handle_wall_collisions(window.getSize().x, window.getSize().y,
                                   particles);
            handle_collisions_grid_omp(particles, grid, locks);
        }

        // --- RENDER ---
        if (cfg.render && window.isOpen()) {
            window.clear();
            for (auto& p : particles) {
                sf::CircleShape shape(p.radius);
                shape.setOrigin(p.radius, p.radius);
                shape.setPosition(p.x, p.y);
                shape.setFillColor(p.color);
                window.draw(shape);
            }
            window.display();
        }

        frame_count++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    // --- Results ---
    double fps = cfg.num_frames / diff.count();
    std::cout << "------------------------------------------------"
              << std::endl;
    std::cout << "Time: " << diff.count() << " s" << std::endl;
    std::cout << "Average UPS (Updates Per Second): " << fps << std::endl;
    std::cout << "------------------------------------------------"
              << std::endl;

    // Cleanup
    for (auto& lock : locks) omp_destroy_lock(&lock);

    return 0;
}
