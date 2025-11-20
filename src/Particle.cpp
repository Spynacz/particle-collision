#include "Particle.hpp"

Particle::Particle(double x, double y, double vx, double vy, double radius,
                   double mass)
    : x(x), y(y), vx(vx), vy(vy), radius(radius), mass(mass) {
    this->colorGradient = ColorGradient();
    this->colorGradient.viridisHeatMap();
}

void Particle::update(double dt) {
    x += vx * dt;
    y += vy * dt;

    updateColor((vx * vx + vy * vy) / 100.0);
}

void Particle::updateColor(double value) {
    float r, g, b;
    colorGradient.getColorAtValue(value, r, g, b);
    color = sf::Color(static_cast<sf::Uint8>(r * 255),
                      static_cast<sf::Uint8>(g * 255),
                      static_cast<sf::Uint8>(b * 255));
}

Particle Particle::randomize(int s_width, int s_height, int particle_size) {
    radius = static_cast<double>((rand() % particle_size) + 1);

    double min_x = radius;
    double min_y = radius;
    double max_x = s_width - radius;
    double max_y = s_height - radius;

    x = std::clamp(static_cast<double>(rand() % s_width), min_x, max_x);
    y = std::clamp(static_cast<double>(rand() % s_height), min_y, max_y);
    vx = static_cast<double>((rand() % 200) - 100) / 5.0;
    vy = static_cast<double>((rand() % 200) - 100) / 5.0;
    mass = radius * radius * 3.14;  // Proporcjonalna do powierzchni
    color = sf::Color(rand() % 255, rand() % 255, rand() % 255);

    return *this;
}
