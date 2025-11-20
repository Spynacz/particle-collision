#pragma once
#include <SFML/Graphics.hpp>

#include "ColorGradient.cpp"

class Particle {
   public:
    double x, y;
    double vx, vy;
    double radius;
    double mass;
    sf::Color color;
    size_t id;
    ColorGradient colorGradient;

    // Konstruktor: pozycja (x,y), prędkość (vx,vy), promień r, masa m
    Particle(double x = 0.0, double y = 0.0, double vx = 0.0, double vy = 0.0,
             double radius = 5.0, double mass = 1.0);

    Particle randomize(int s_width, int s_height);

    // Aktualizuje pozycję na podstawie prędkości i kroku czasowego dt
    void update(double dt);
    void updateColor(double value);
};
