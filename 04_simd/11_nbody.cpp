#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  int ja[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    ja[i] = i;
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512i ivec = _mm512_set1_epi32(i);
    __m512i jvec = _mm512_load_epi32(ja);
    __mmask16 mask = _mm512_cmpneq_epi32_mask(ivec, jvec);
   
    __m512 xvec = _mm512_load_ps(x);
    __m512 yvec = _mm512_load_ps(y);
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 yivec = _mm512_set1_ps(y[i]);
    __m512 rxvec = _mm512_sub_ps(xivec, xvec);
    __m512 ryvec = _mm512_sub_ps(yivec, yvec);
    
    __m512 rxpow2vec = _mm512_mul_ps(rxvec, rxvec);
    __m512 rypow2vec = _mm512_mul_ps(ryvec, ryvec);
    __m512 distvec = _mm512_add_ps(rxpow2vec, rypow2vec);
    __m512 rvec = _mm512_sqrt_ps(distvec);
    __m512 r2vec = _mm512_mul_ps(rvec, rvec);
    __m512 r3vec = _mm512_mul_ps(r2vec, rvec);
    
    __m512 mjvec = _mm512_load_ps(m);
    __m512 rxmjvec = _mm512_mul_ps(rxvec, mjvec);
    __m512 rymjvec = _mm512_mul_ps(ryvec, mjvec);
    __m512 subfxvec = _mm512_div_ps(rxmjvec, r3vec);
    __m512 subfyvec = _mm512_div_ps(rymjvec, r3vec);
    
    __m512 lastfxsubvec = _mm512_setzero_ps();
    lastfxsubvec = _mm512_mask_blend_ps(mask, lastfxsubvec, subfxvec);
    __m512 lastfysubvec = _mm512_setzero_ps();
    lastfysubvec = _mm512_mask_blend_ps(mask, lastfysubvec, subfyvec);

    float subfx = _mm512_reduce_add_ps(lastfxsubvec);
    float subfy = _mm512_reduce_add_ps(lastfysubvec);
    fx[i] -= subfx;
    fy[i] -= subfy;
//    for(int j=0; j<N; j++) {
//      if(i != j) {
//        float rx = x[i] - x[j];
//        float ry = y[i] - y[j];
//        float r = std::sqrt(rx * rx + ry * ry);
//        fx[i] -= rx * m[j] / (r * r * r);
//        fy[i] -= ry * m[j] / (r * r * r);
//      }   
//    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
