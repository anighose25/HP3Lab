#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define LOG_INPUT if(1)
#define LOG_SCAN if(0)
#define LOG_OUTPUT if(1)


void print_array(float *A, int N)
{
    for(int i=0;i<N;i++)
        printf("%.2f ",A[i]);
    printf("\n");
}

#define TILE_SIZE 512

__global__ void Convolution1D(float *N, float *M, float *P, int Mask_Width, int Width){ 

	int i = blockIdx.x*blockDim.x + threadIdx.x; 
	__shared__ float N_ds[TILE_SIZE]; 
    N_ds[threadIdx.x] = N[i]; 
	__syncthreads(); 



	int This_tile_start_point = blockIdx.x * blockDim.x; 
	int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x; 
	int N_start_point = i - (Mask_Width/2); 
	float Pvalue = 0; 

	for (int j = 0; j < Mask_Width; j ++) { 
		int N_index = N_start_point + j; 
		if (N_index >= 0 && N_index < Width) { 
			if ((N_index >= This_tile_start_point) && (N_index < Next_tile_start_point))  
				Pvalue += N_ds[threadIdx.x+j-(Mask_Width/2)]*M[j]; 
			else 
				Pvalue += N[N_index] * M[j]; 

		} 
	} 
	P[i] = Pvalue; 
} 	

__global__ void ConvolutionPoolingFused(float *N, float *M, float *P, float *A, int Mask_Width, int Width){ 

	
	__shared__ float N_ds[TILE_SIZE]; 
	__shared__ float P_ds[TILE_SIZE]; 
	unsigned int tid0 = ( threadIdx .x /16) *16*2 + threadIdx .x %16;
	unsigned int tid1 = tid0 + 16;
	unsigned int data_point0 = blockIdx .x *2* blockDim .x + tid0 ;
	unsigned int data_point1 = blockIdx .x *2* blockDim .x + tid1 ;
	N_ds[tid0] =  N[data_point0] ;
	N_ds[tid1] =  N[data_point1];
	__syncthreads ();
	int cur_tile_start_point = blockIdx.x *2 * blockDim.x; 
	int next_tile_start_point = (blockIdx.x + 1) *2* blockDim.x; 
	int N_start_point0 = data_point0 - (Mask_Width/2); 
	int N_start_point1 = data_point1 - (Mask_Width/2);    
	float PValue0 = 0; 
	float PValue1 = 0; 
	for (int j = 0; j < Mask_Width; j ++) { 
		int N_index0 = N_start_point0 + j; 
		int N_index1 = N_start_point1 + j; 
		if (N_index0 >= 0 && N_index0 < Width) { 
			if ((N_index0 >= cur_tile_start_point) && (N_index0 < next_tile_start_point)  )  
				PValue0 += N_ds[tid0 +j-(Mask_Width/2)]*M[j];
			else 
				PValue0 += N[N_index0] * M[j]; 

		} 
		if (N_index1 >= 0 && N_index1 < Width) { 
			if ((N_index1 >= cur_tile_start_point) && (N_index1 < next_tile_start_point)  )  
					PValue1 += N_ds[tid1 +j-(Mask_Width/2)]*M[j]; 
				else 
					PValue1 += N[N_index1] * M[j]; 
			
		} 

	} 
    __syncthreads ();

	P_ds[tid0] = PValue0;
	P_ds[tid1] = PValue1;
	__syncthreads();


	tid0 = threadIdx.x; 
    int bound0 =  blockDim.x;
    tid1 = blockDim.x + threadIdx.x; 
    int bound1 = blockDim.x + blockDim.x;
    int blockSize = blockDim.x;
    
    for(int s=1;s<blockSize;s*=2)
    { 
       if(tid0%2*s == 0 && tid0+s<bound0)
           P_ds[tid0] = max(P_ds[tid0],P_ds[tid0+s]);
       __syncthreads();
	   if(tid1%2*s == 0 && tid1+s<bound1)
           P_ds[tid1] = max(P_ds[tid1],P_ds[tid1+s]);
       __syncthreads();


    }  

    if(threadIdx.x == 0)
    {
    	A[2*blockIdx.x] = P_ds[tid0];
    	A[2*blockIdx.x+1] = P_ds[tid0 + blockDim.x];
    }

} 	



__global__ void Pool1D(float* A, float* B, int N)
{
     int tid = (blockIdx.x) * blockDim.x + threadIdx.x; 
     int bound = (blockIdx.x) * blockDim.x + blockDim.x;
     int blockSize = blockDim.x;
     for(int s=1;s<blockSize;s*=2)
     { 
        if(tid%2*s == 0 && tid+s<bound)
           A[tid] = max(A[tid],A[tid+s]);
        __syncthreads();
     }  
     if(threadIdx.x == 0)
          B[blockIdx.x] = A[(blockIdx.x) * blockDim.x];
}



int main(void)
{
    cudaError_t err = cudaSuccess;
    int numElements;
    int numMaskElements;
    int pooling_size;
    scanf("%d",&numElements);
    scanf("%d",&numMaskElements);
    scanf("%d",&pooling_size);
    size_t size = numElements * sizeof(float);
    size_t sizeMask = numMaskElements * sizeof(float);

    float *h_input1 = (float *)malloc(size);
    float *h_input2 = (float *)malloc(sizeMask);
    float *h_output1 = (float *)malloc(size);
    float *h_output2 = (float *)malloc(size);

 
    if (h_input1 == NULL || h_input2 == NULL || h_output1 == NULL || h_output2 == NULL )
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    LOG_INPUT
    for (int i = 0; i < numElements; ++i){
        h_input1[i] = 1.0;
    }

    for(int i=0;i<numMaskElements;++i){
    	h_input2[i] = 0.25;
    }

    LOG_SCAN
    {
        for (int i = 0; i < numElements; ++i)
        {
            scanf("%f",&h_input1[i]);
            
        }
        for (int i = 0; i < numMaskElements; ++i)
        {
            scanf("%f",&h_input2[i]);
            
        }
    }

    LOG_INPUT print_array(h_input1,numElements);
    LOG_INPUT print_array(h_input2,numMaskElements);


//H2D Transfers

    float *d_input1 = NULL;
    err = cudaMalloc((void **)&d_input1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_input2 = NULL;
    err = cudaMalloc((void **)&d_input2, sizeMask);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_output1 = NULL;
    err = cudaMalloc((void **)&d_output1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector h_output1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_output2 = NULL;
    err = cudaMalloc((void **)&d_output2, size/pooling_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector h_output1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input1, h_input1, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input2, h_input2, sizeMask, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_input2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int block_size=512;
    int grid_size=numElements/block_size;

//Launch kernel

    Convolution1D<<<grid_size, block_size>>>(d_input1, d_input2, d_output1, numMaskElements,numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch Convolution1D kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
// D2H transfer

    err = cudaMemcpy(h_output1, d_output1, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_output1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    LOG_OUTPUT printf("Output of convolution\n");
    LOG_OUTPUT print_array(h_output1,numElements);

    grid_size=numElements/pooling_size;
    block_size=pooling_size;

    Pool1D<<<grid_size,block_size>>>(d_output1,d_output2,numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch Pooling1D kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(h_output2, d_output2, size/pooling_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_output2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    LOG_OUTPUT printf("Output of pooling\n");
    LOG_OUTPUT print_array(h_output2,numElements/pooling_size);

    grid_size=numElements/512;
    block_size=pooling_size;
    ConvolutionPoolingFused<<<grid_size,block_size>>>(d_input1, d_input2, d_output1, d_output2, numMaskElements,numElements);
    err = cudaMemcpy(h_output2, d_output2, size/pooling_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_output2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    LOG_OUTPUT printf("Output of pooling\n");
    LOG_OUTPUT print_array(h_output2,numElements/pooling_size);


}