#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <algorithm> 

// CUDA runtime
#include <cuda_runtime.h>

//  compute b
__global__ void compute_b_kernel(const float* u, const float* v, float* b, int nx, int ny, double dx, double dy, double dt, double rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = j * nx + i;
        
        int idx_ip1 = j * nx + (i + 1);
        int idx_im1 = j * nx + (i - 1);
        int idx_jp1 = (j + 1) * nx + i;
        int idx_jm1 = (j - 1) * nx + i;

        float u_x = (u[idx_ip1] - u[idx_im1]) / (2.0f * dx);
        float v_y = (v[idx_jp1] - v[idx_jm1]) / (2.0f * dy);
        float u_y = (u[idx_jp1] - u[idx_jm1]) / (2.0f * dy);
        float v_x = (v[idx_ip1] - v[idx_im1]) / (2.0f * dx);

        b[idx] = rho * ((1.0f / dt) *
                   (u_x + v_y) -
                   u_x * u_x - 2.0f * u_y 
		   * v_x - v_y * v_y
                  );
    }
}

// copy 
__global__ void copy_kernel(const float* src, float* dest, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        int idx = j * nx + i;
        dest[idx] = src[idx];
    }
}

// compute p
__global__ void compute_p_kernel(float* p, const float* pn, const float* b, int nx, int ny, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = j * nx + i;
        int idx_ip1 = j * nx + (i + 1);
        int idx_im1 = j * nx + (i - 1);
        int idx_jp1 = (j + 1) * nx + i;
        int idx_jm1 = (j - 1) * nx + i;
        p[idx] = (dy*dy * (pn[idx_ip1] + pn[idx_im1]) +
                  dx*dx * (pn[idx_jp1] + pn[idx_jm1]) -
                  b[idx] * dx*dx * dy*dy)
                 / (2.0f * (dx*dx + dy*dy));
    }
}

// compute p[j][0] p[j][nx-1]
__global__ void p_lr_kernel(float* p, int nx, int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ny) {
      p[j * nx + (nx - 1)] = p[j * nx + (nx - 2)];
      p[j * nx] = p[j * nx + 1];
    }
}

// compute p[0][i] p[ny-1][i] 
__global__ void p_tb_kernel(float* p, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
	p[i] = p[nx + i];
	p[(ny - 1) * nx + i] = 0;
    }
}

// compute u, v 
__global__ void update_uv_kernel(float* u, float* v, const float* un, const float* vn, const float* p,
                                         int nx, int ny, double dx, double dy, double dt, double rho, double nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = j * nx + i;
        int idx_im1 = j * nx + (i - 1);
        int idx_ip1 = j * nx + (i + 1);
        int idx_jm1 = (j - 1) * nx + i;
        int idx_jp1 = (j + 1) * nx + i;

        u[idx] = un[idx] - un[idx] * dt / dx * (un[idx] - un[idx_im1])
                 - un[idx] * dt / dy * (un[idx] - un[idx_jm1]) 
                 - dt / (2.0f * rho * dx) * (p[idx_ip1] - p[idx_im1])
                 + nu * dt / (dx*dx) * (un[idx_ip1] - 2.0f * un[idx] + un[idx_im1])
                 + nu * dt / (dy*dy) * (un[idx_jp1] - 2.0f * un[idx] + un[idx_jm1]);

        v[idx] = vn[idx] - vn[idx] * dt / dx * (vn[idx] - vn[idx_im1])
                 - vn[idx] * dt / dy * (vn[idx] - vn[idx_jm1])
                 - dt / (2.0f * rho * dy) * (p[idx_jp1] - p[idx_jm1])
                 + nu * dt / (dx*dx) * (vn[idx_ip1] - 2.0f * vn[idx] + vn[idx_im1])
                 + nu * dt / (dy*dy) * (vn[idx_jp1] - 2.0f * vn[idx] + vn[idx_jm1]);
    }
}

// Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1] 
__global__ void uv_lr_kernel(float* u, float* v, int nx, int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < ny) {
        u[j * nx] = 0;
        u[j * nx + (nx - 1)] = 0;
        v[j] = 0;
        v[j * nx + (nx - 1)] = 0; 
    }
}

// Kernel for v-velocity boundary conditions (left, right, top walls)
__global__ void uv_tb_kernel(float* u,float* v, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nx) {
       u[i] = 0;
       u[(ny - 1) * nx + i] = 1;
       v[i] = 0;
       v[(ny - 1) * nx + i] = 0;
    }
}


int main() {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx_val = 2. / (nx - 1);
    double dy_val = 2. / (ny - 1);
    double dt_val = .01;
    double rho_val = 1.;
    double nu_val = .02;

    size_t N_elements = (size_t)nx * ny;
    size_t N_bytes = N_elements * sizeof(float);

    std::vector<float> h_u(N_elements);
    std::vector<float> h_v(N_elements);
    std::vector<float> h_p(N_elements);

    float *d_u, *d_v, *d_p, *d_b, *d_un, *d_vn, *d_pn;

    cudaMalloc(&d_u, N_bytes);
    cudaMalloc(&d_v, N_bytes);
    cudaMalloc(&d_p, N_bytes);
    cudaMalloc(&d_b, N_bytes);
    cudaMalloc(&d_un, N_bytes);
    cudaMalloc(&d_vn, N_bytes);
    cudaMalloc(&d_pn, N_bytes);

    for (size_t i = 0; i < N_elements; ++i) {
        h_u[i] = 0.0f;
        h_v[i] = 0.0f;
        h_p[i] = 0.0f;
    }
    
    cudaMemcpy(d_u, h_u.data(), N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p.data(), N_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, N_bytes); 
    cudaMemset(d_un, 0, N_bytes);
    cudaMemset(d_vn, 0, N_bytes);
    cudaMemset(d_pn, 0, N_bytes);


    std::ofstream ufile("u.dat");
    std::ofstream vfile("v.dat");
    std::ofstream pfile("p.dat");
    ufile << std::fixed << std::setprecision(10);
    vfile << std::fixed << std::setprecision(10);
    pfile << std::fixed << std::setprecision(10);

    dim3 threadsPerBlock2D(16, 16); 
    dim3 numBlocks2D((nx + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                     (ny + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);

    dim3 threadsPerBlock1D(256);
    dim3 numBlocks1D_nx((nx + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
    dim3 numBlocks1D_ny((ny + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
    dim3 numBlocks1D_max_dim((std::max(nx, ny) + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);


    for (int n = 0; n < nt; n++) {
        compute_b_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_u, d_v, d_b, nx, ny, dx_val, dy_val, dt_val, rho_val);

        for (int it = 0; it < nit; it++) {
            copy_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_p, d_pn, nx, ny);
            compute_p_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_p, d_pn, d_b, nx, ny, dx_val, dy_val);

            p_lr_kernel<<<numBlocks1D_ny, threadsPerBlock1D>>>(d_p, nx, ny);
            p_tb_kernel<<<numBlocks1D_nx, threadsPerBlock1D>>>(d_p, nx, ny);
        }

        copy_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_u, d_un, nx, ny);
        copy_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_v, d_vn, nx, ny);

        update_uv_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_u, d_v, d_un, d_vn, d_p, nx, ny, dx_val, dy_val, dt_val, rho_val, nu_val);
        uv_lr_kernel<<<numBlocks1D_max_dim, threadsPerBlock1D>>>(d_u, d_v, nx, ny);
        uv_tb_kernel<<<numBlocks1D_max_dim, threadsPerBlock1D>>>(d_u, d_v, nx, ny);


        if (n % 10 == 0) {
            cudaMemcpy(h_u.data(), d_u, N_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v.data(), d_v, N_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p.data(), d_p, N_bytes, cudaMemcpyDeviceToHost);

            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    ufile << h_u[j * nx + i] << " ";
                }
                ufile << "\n";
            }
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    vfile << h_v[j * nx + i] << " ";
                }
                vfile << "\n";
            }
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    pfile << h_p[j * nx + i] << " ";
                }
                pfile << "\n";
            }
        }
    }

    ufile.close();
    vfile.close();
    pfile.close();

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_un);
    cudaFree(d_vn);
    cudaFree(d_pn);

    std::cout << "Simulation complete." << std::endl;
    return 0;
}
