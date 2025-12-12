#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#include "CudaParticle.h"

#define BLOCK_SIZE 256

__global__ void update_physics_kernel(CudaParticle* particles,
                                      int num_particles, float dt, int width,
                                      int height) {
    // global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_particles) return;

    CudaParticle& p1 = particles[i];

    p1.x += p1.vx * dt;
    p1.y += p1.vy * dt;

    if (p1.x > width - p1.radius) {
        p1.x = width - p1.radius;
        p1.vx = -p1.vx;
    } else if (p1.x < p1.radius) {
        p1.x = p1.radius;
        p1.vx = -p1.vx;
    }

    if (p1.y > height - p1.radius) {
        p1.y = height - p1.radius;
        p1.vy = -p1.vy;
    } else if (p1.y < p1.radius) {
        p1.y = p1.radius;
        p1.vy = -p1.vy;
    }

    for (int j = 0; j < num_particles; j++) {
        if (i == j) continue;

        CudaParticle p2 = particles[j];

        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist < p1.radius + p2.radius && dist > 0) {
            float nx = dx / dist;
            float ny = dy / dist;

            float dvx = p2.vx - p1.vx;
            float dvy = p2.vy - p1.vy;
            float dvn = dvx * nx + dvy * ny;

            if (dvn > 0) continue;

            float impulse = 2.0f * dvn / (p1.mass + p2.mass);

            p1.vx += impulse * p2.mass * nx;
            p1.vy += impulse * p2.mass * ny;

            float overlap = 0.5f * (p1.radius + p2.radius - dist);
            float totalMass = p1.mass + p2.mass;
            p1.x -= overlap * (p2.mass / totalMass) * nx;
            p1.y -= overlap * (p2.mass / totalMass) * ny;
        }
    }
}

extern "C" void run_cuda_simulation(CudaParticle* host_particles,
                                    int num_particles, float dt, int width,
                                    int height) {
    CudaParticle* d_particles;
    size_t size = num_particles * sizeof(CudaParticle);

    cudaMalloc((void**)&d_particles, size);

    cudaMemcpy(d_particles, host_particles, size, cudaMemcpyHostToDevice);

    int blocks = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch Kernel
    update_physics_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, num_particles,
                                                  dt, width, height);

    cudaDeviceSynchronize();

    // GPU -> CPU
    cudaMemcpy(host_particles, d_particles, size, cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
}
