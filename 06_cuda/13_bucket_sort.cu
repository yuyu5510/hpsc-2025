#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>


__global__ void count_keys(int *gbucket, int *gkey, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&gbucket[gkey[i]], 1);
} 

__global__ void sum_keys(int *gsum, int *gbucket, int range){
  if (threadIdx.x==0){
    for(int i = blockIdx.x; i < range;i++){
      atomicAdd(&gsum[i], gbucket[blockIdx.x]);
    }
  }
}

__global__ void set(int *gsum, int *gkey, int range, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n){
    for(int ki = 0; ki < range; ki++){
      if (i < gsum[ki]){
        gkey[i] = ki;
	break;
      }
    }
  }
}


int main() {
  int n = 50;
  int range = 5;
//  std::vector<int> key(n);
  int key[50];
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int blocks = range;
  int kbsize = (n + blocks - 1) / blocks;
  int *gkey, *gbucket, *gsum;

  cudaMallocManaged(&gkey   , n*sizeof(int));
  cudaMallocManaged(&gbucket, range*sizeof(int));
  cudaMallocManaged(&gsum   , range*sizeof(int));
  cudaMemcpy(gkey, key, n*sizeof(int), cudaMemcpyDeviceToHost);

  // initialize
  cudaMemset(gbucket, 0, range*sizeof(int));
  cudaMemset(gsum, 0, range*sizeof(int));

  // counting
  count_keys<<<blocks, kbsize>>>(gbucket, gkey, n);
  cudaDeviceSynchronize();
  
  // sum
  sum_keys<<<blocks, 1>>>(gsum, gbucket, range);
  cudaDeviceSynchronize();

  // set
  set<<<kbsize, blocks>>>(gsum, gkey, range, n);
  cudaDeviceSynchronize();
  
  cudaMemcpy(key, gkey, n*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(gkey);
  cudaFree(gbucket);
  cudaFree(gsum);
}
