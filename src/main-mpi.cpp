#include <mpi.h>
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <map>
#include <algorithm>
#include <cstring>

#include "Particle.hpp"

struct MpiParticleData {
    double x, y;
    double vx, vy;
    double radius;
    double mass;
    size_t id;
};

struct Config {
    int num_particles = 500;
    int num_frames = 1000;
    bool render = false;
    int particle_size = 5;
    int world_width = 1600;
    int world_height = 900;
};

#define GRID_CELL_SIZE 50
typedef std::vector<size_t> GridCell;
typedef std::map<int64_t, GridCell> ParticleGrid;

int64_t getCellKey(int x, int y) {
    return (static_cast<int64_t>(x) << 32) | static_cast<int64_t>(y);
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

void handle_collisions_grid_local(std::vector<Particle>& particles, ParticleGrid& grid) {
    grid.clear();
    
    for (size_t i = 0; i < particles.size(); ++i) {
        if(particles[i].x < 0 || particles[i].y < 0) continue; 
        
        int cellX = static_cast<int>(particles[i].x / GRID_CELL_SIZE);
        int cellY = static_cast<int>(particles[i].y / GRID_CELL_SIZE);
        grid[getCellKey(cellX, cellY)].push_back(i);
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p1 = particles[i];
        int cellX = static_cast<int>(p1.x / GRID_CELL_SIZE);
        int cellY = static_cast<int>(p1.y / GRID_CELL_SIZE);

        for (int y = cellY - 1; y <= cellY + 1; ++y) {
            for (int x = cellX - 1; x <= cellX + 1; ++x) {
                auto it = grid.find(getCellKey(x, y));
                if (it == grid.end()) continue;

                for (size_t j_idx : it->second) {
                    if (i >= j_idx) continue;
                    handle_single_collision(p1, particles[j_idx]);
                }
            }
        }
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: mpirun -np <N> " << program_name << " [options]\n"
              << "Options:\n"
              << "  -n, --count <int>    Total particles (default: 500)\n"
              << "  -f, --frames <int>   Number of frames (default: 1000)\n"
              << "  --render             Enable rendering (Rank 0 gathers data - SLOW)\n"
              << std::endl;
}

Config parse_args(int argc, char* argv[]) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--count") config.num_particles = std::stoi(argv[++i]);
        else if (arg == "-f" || arg == "--frames") config.num_frames = std::stoi(argv[++i]);
        else if (arg == "--render") config.render = true;
        else if (arg == "-h") { print_usage(argv[0]); exit(0); }
    }
    return config;
}

MpiParticleData pack_particle(const Particle& p) {
    return {p.x, p.y, p.vx, p.vy, p.radius, p.mass, p.id};
}

Particle unpack_particle(const MpiParticleData& data) {
    Particle p(data.x, data.y, data.vx, data.vy, data.radius, data.mass);
    p.id = data.id;
    return p;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Datatype mpi_particle_type;
    MPI_Type_contiguous(7, MPI_DOUBLE, &mpi_particle_type);
    MPI_Type_commit(&mpi_particle_type);

    Config cfg = parse_args(argc, argv);
    srand(42 + world_rank);

    // split the screen vertically into columns.
    double domain_width = (double)cfg.world_width / world_size;
    double min_x = world_rank * domain_width;
    double max_x = (world_rank + 1) * domain_width;

    int particles_per_rank = cfg.num_particles / world_size;
    if (world_rank == world_size - 1) {
        particles_per_rank += cfg.num_particles % world_size; // Last rank takes remainder
    }

    std::vector<Particle> particles;
    ParticleGrid grid;

    for (int i = 0; i < particles_per_rank; i++) {
        Particle p;
        p.randomize(cfg.world_width, cfg.world_height, cfg.particle_size);
        
        p.x = min_x + (static_cast<double>(rand()) / RAND_MAX) * (domain_width - p.radius * 2) + p.radius;
        
        p.id = world_rank * 1000000 + i;
        particles.push_back(p);
    }

    sf::RenderWindow window;
    if (world_rank == 0 && cfg.render) {
        window.create(sf::VideoMode(cfg.world_width, cfg.world_height), "MPI Particle Simulation");
        window.setFramerateLimit(60);
    }

    double dt = 0.1;
    double start_time = MPI_Wtime();

    for (int frame = 0; frame < cfg.num_frames; ++frame) {
        if (world_rank == 0 && cfg.render) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) window.close();
            }
        }

        for (auto& p : particles) {
            p.update(dt);
        }

        for (auto& p : particles) {
            if (p.y > cfg.world_height - p.radius) { p.y = cfg.world_height - p.radius; p.vy = -p.vy; }
            else if (p.y < p.radius) { p.y = p.radius; p.vy = -p.vy; }
            
            if (world_rank == 0 && p.x < p.radius) { p.x = p.radius; p.vx = -p.vx; }
            if (world_rank == world_size - 1 && p.x > cfg.world_width - p.radius) { 
                p.x = cfg.world_width - p.radius; p.vx = -p.vx; 
            }
        }

        std::vector<MpiParticleData> send_left, send_right;
        
        auto it = std::remove_if(particles.begin(), particles.end(), [&](const Particle& p) {
            bool remove = false;
            if (p.x < min_x && world_rank > 0) {
                send_left.push_back(pack_particle(p));
                remove = true;
            } else if (p.x >= max_x && world_rank < world_size - 1) {
                send_right.push_back(pack_particle(p));
                remove = true;
            }
            return remove;
        });
        particles.erase(it, particles.end());

        int send_cnt_l = send_left.size();
        int send_cnt_r = send_right.size();
        int recv_cnt_l = 0;
        int recv_cnt_r = 0;

        MPI_Request reqs[4];
        MPI_Status stats[4];
        
        int left_neighbor = (world_rank > 0) ? world_rank - 1 : MPI_PROC_NULL;
        int right_neighbor = (world_rank < world_size - 1) ? world_rank + 1 : MPI_PROC_NULL;

        MPI_Sendrecv(&send_cnt_l, 1, MPI_INT, left_neighbor, 0,
                     &recv_cnt_r, 1, MPI_INT, right_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&send_cnt_r, 1, MPI_INT, right_neighbor, 1,
                     &recv_cnt_l, 1, MPI_INT, left_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<MpiParticleData> recv_left_data(recv_cnt_l);
        std::vector<MpiParticleData> recv_right_data(recv_cnt_r);

        MPI_Sendrecv(send_left.data(), send_cnt_l, mpi_particle_type, left_neighbor, 2,
                     recv_right_data.data(), recv_cnt_r, mpi_particle_type, right_neighbor, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     
        MPI_Sendrecv(send_right.data(), send_cnt_r, mpi_particle_type, right_neighbor, 3,
                     recv_left_data.data(), recv_cnt_l, mpi_particle_type, left_neighbor, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(const auto& data : recv_left_data) particles.push_back(unpack_particle(data));
        for(const auto& data : recv_right_data) particles.push_back(unpack_particle(data));

        handle_collisions_grid_local(particles, grid);

        for(auto& p : particles) {
             float mag = (p.vx * p.vx + p.vy * p.vy) / 100.0;
             p.updateColor(mag);
        }

        if (cfg.render) {
            std::vector<MpiParticleData> local_data;
            local_data.reserve(particles.size());
            for(const auto& p : particles) local_data.push_back(pack_particle(p));

            int local_count = local_data.size();
            std::vector<int> all_counts(world_size);
            MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

            std::vector<int> displacements(world_size, 0);
            int total_particles = 0;
            if (world_rank == 0) {
                for (int i = 0; i < world_size; ++i) {
                    displacements[i] = total_particles;
                    total_particles += all_counts[i];
                }
            }

            std::vector<MpiParticleData> global_data;
            if (world_rank == 0) global_data.resize(total_particles);

            MPI_Gatherv(local_data.data(), local_count, mpi_particle_type,
                        global_data.data(), all_counts.data(), displacements.data(), mpi_particle_type,
                        0, MPI_COMM_WORLD);

            if (world_rank == 0 && window.isOpen()) {
                window.clear();
                ColorGradient gradient; 
                gradient.viridisHeatMap();

                for (const auto& data : global_data) {
                    sf::CircleShape shape(data.radius);
                    shape.setOrigin(data.radius, data.radius);
                    shape.setPosition(data.x, data.y);
                    
                    // Re-calculate color for rendering
                    float val = (data.vx * data.vx + data.vy * data.vy) / 100.0;
                    float r, g, b;
                    gradient.getColorAtValue(val, r, g, b);
                    shape.setFillColor(sf::Color(r*255, g*255, b*255));
                    
                    window.draw(shape);
                }
                window.display();
            }
        }
    }

    double end_time = MPI_Wtime();
    
    if (world_rank == 0) {
        std::cout << "--- MPI Simulation Results ---" << std::endl;
        std::cout << "Processes: " << world_size << std::endl;
        std::cout << "Time: " << (end_time - start_time) << " s" << std::endl;
        std::cout << "UPS: " << cfg.num_frames / (end_time - start_time) << std::endl;
        std::cout << "------------------------------" << std::endl;
    }

    MPI_Type_free(&mpi_particle_type);
    MPI_Finalize();
    return 0;
}
