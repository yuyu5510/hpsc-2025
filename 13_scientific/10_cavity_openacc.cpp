#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
  const int nx = 41;
  const int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  float u[ny][nx];
  float v[ny][nx];
  float p[ny][nx];
  float b[ny][nx];
  float un[ny][nx];
  float vn[ny][nx];
  float pn[ny][nx];
  #pragma acc data create(u, v, p, b, un, vn, pn) copyin(dx, dy, dt, rho, nu, nx, ny)
{
  #pragma acc parallel loop present(u, v, p, b, nx, ny)
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  ufile << fixed << setprecision(10);
  vfile << fixed << setprecision(10);
  pfile << fixed << setprecision(10);

  for (int n=0; n<nt; n++) {
    #pragma acc parallel loop present(b, u, v, rho, dt, dx, dy, nx, ny)
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        // Compute b[j][i]
	float u_x = (u[j][i+1] - u[j][i-1]) / (2.0 * dx);
	float v_y = (v[j+1][i] - v[j-1][i]) / (2.0 * dy);
	float u_y = (u[j+1][i] - u[j-1][i]) / (2.0 * dy);
	float v_x = (v[j][i+1] - v[j][i-1]) / (2.0 * dx);

	b[j][i] = rho * ((1.0 / dt) * 
	   (u_x + v_y) -
	    u_x * u_x - 2.0f * u_y *
	    v_x - v_y * v_y
	);
      }
    }
    for (int it=0; it<nit; it++) {
      #pragma acc parallel loop present(pn, p, nx, ny)
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
	  pn[j][i] = p[j][i];
      #pragma acc parallel loop present(p, pn, b, dx, dy, nx, ny)
      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
	  // Compute p[j][i]
	  p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                     dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                     b[j][i] * dx*dx * dy*dy) 
		      / (2 * (dx*dx + dy*dy));
	}
      }
      #pragma acc parallel loop present(p, nx, ny)
      for (int j=0; j<ny; j++) {
        // Compute p[j][0] and p[j][nx-1]
	p[j][nx-1] = p[j][nx-2];
        p[j][0] = p[j][1];
      }
      #pragma acc parallel loop present(p, nx, ny)
      for (int i=0; i<nx; i++) {
	// Compute p[0][i] and p[ny-1][i]
        p[0][i] = p[1][i];
	p[ny-1][i] = 0;
      }
    }
    #pragma acc parallel loop present(un, vn, u, v, nx, ny)
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
	vn[j][i] = v[j][i];
      }
    }
    #pragma acc parallel loop present(u, v, un, vn, p, dt, dx, dy, rho, nu, nx, ny)
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
	// Compute u[j][i] and v[j][i]
	u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
		  - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
		  - dt / (2.0 * rho * dx) * (p[j][i+1] - p[j][i-1])
		  + nu * dt / (dx*dx) * (un[j][i+1] - 2.0 * un[j][i] + un[j][i-1])
		  + nu * dt / (dy*dy) * (un[j+1][i] - 2.0 * un[j][i] + un[j-1][i]);
	v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
          - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
          - dt / (2.0 * rho * dy) * (p[j+1][i] - p[j-1][i]) 
          + nu * dt / (dx*dx) * (vn[j][i+1] - 2.0 * vn[j][i] + vn[j][i-1])
          + nu * dt / (dy*dy) * (vn[j+1][i] - 2.0 * vn[j][i] + vn[j-1][i]);
      }
    }
    #pragma acc parallel loop present(u, v, nx, ny)
    for (int j=0; j<ny; j++) {
      // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
      u[j][0] = 0;
      u[j][nx-1] = 0;
      v[j][0] = 0;
      v[j][nx-1] = 0;
    }
    #pragma acc parallel loop present(u, v, nx, ny)
    for (int i=0; i<nx; i++) {
      // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
      u[0][i] = 0;
      u[ny-1][i] = 1;
      v[0][i] = 0;
      v[ny-1][i] = 0;
    }
    if (n % 10 == 0) {
      #pragma acc update host(u, v, p)
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}
}
