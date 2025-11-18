#include "Particle.hpp"

Particle::Particle(double x, double y, double vx, double vy, double radius,
                   double mass)
    : x(x), y(y), vx(vx), vy(vy), radius(radius), mass(mass) {}

void Particle::update(double dt) {
    x += vx * dt;
    y += vy * dt;
}

Particle Particle::randomize(int s_width, int s_height) {
    radius = static_cast<double>((rand() % 20) + 5);

    double min_x = radius;
    double min_y = radius;
    double max_x = s_width - radius;
    double max_y = s_height - radius;

    x = std::clamp(static_cast<double>(rand() % s_width), min_x, max_x);
    y = std::clamp(static_cast<double>(rand() % s_height), min_y, max_y);
    vx = static_cast<double>((rand() % 200) - 100) / 100.0;
    vy = static_cast<double>((rand() % 200) - 100) / 100.0;
    mass = radius * radius * 3.14;  // Proporcjonalna do powierzchni
    color = sf::Color(rand() % 255, rand() % 255, rand() % 255);

    return *this;
}
