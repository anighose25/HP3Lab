#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define LOG_DEBUG if(0)
#define LOG_INPUT if(0)
#define LOG_OUTPUT if(0)


__global__ void sumTriangle(float* M, float* V, int N);
__global__ void sumTriangleWithAtomics(float* M, float* V, int N);
__global__ void sumTriangle(float* M, float* V, int N){

	int j=threadIdx.x;
	float sum=0.0;
	for (int i=0;i<j;i++)
		sum+=M[i*N+j];

	V[j]=sum;
	__syncthreads();

	if(j == N-1)
	{	sum = 0.0;
		for(int i=0;i<N;i++)
			sum =sum + V[i];
		V[N] = sum;
	}

}

__global__ void sumTriangleWithAtomics(float* M, float* V, int N){

	int j=threadIdx.x;
	float sum=0.0;
	__shared__ float totalSum;

	if(j==0)
           totalSum=0.0;
	__syncthreads();

        for (int i=0;i<j;i++)
	  sum+=M[i*N+j];

	V[j]=sum;
        atomicAdd(&totalSum, sum);
	__syncthreads();
        if(j == N-1)
	{
         V[N]=totalSum;
        }
}


void print_matrix(float *A,int m,int n)
{
    for(int i =0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            printf("%.2f ",A[i*n+j]);
        printf("\n");
    }

}

void init_matrix(float *A,int m,int n)
{
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            A[i*n+j]=j;
        
    }

}

int main(void)
{
    cudaError_t err = cudaSuccess;
    int t=1;
    int option;
    LOG_INPUT printf("%d\n",t);
    while(t--)
    {
	    int mat_row;
	    scanf("%d %d",&mat_row,&option);
	   	int mat_dim = mat_row;
	    int num_elems = mat_dim*mat_dim;
	    size_t size_M = num_elems*sizeof(float);
	    size_t size_V = (1+mat_dim)*sizeof(float);
	    

	    float *h_input1 = (float *)malloc(size_M);
	    float *h_output1 = (float *)malloc(size_V);

	    if (h_input1 == NULL || h_output1 == NULL)
	    {
	     	fprintf(stderr, "Failed to allocate host vectors!\n");
	        exit(EXIT_FAILURE);
	    }

	    init_matrix(h_input1,mat_dim,mat_dim);
	    LOG_INPUT print_matrix(h_input1,mat_dim,mat_dim);
       

	   

	    float *d_input1 = NULL;
	    err = cudaMalloc((void **)&d_input1, size_M);
	    if (err != cudaSuccess)
	    {
	     	fprintf(stderr, "Failed to allocate device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    float *d_output1 = NULL;
	    err = cudaMalloc((void **)&d_output1, size_V);
	    if (err != cudaSuccess)
	    {
	     	fprintf(stderr, "Failed to allocate device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    LOG_DEBUG printf("Copy input data from the host memory to the CUDA device\n");

	    err = cudaMemcpy(d_input1, h_input1, size_M, cudaMemcpyHostToDevice);
	    if (err != cudaSuccess)
	    {
	     	fprintf(stderr, "Failed to copy vector h_input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    int grid_dim = 1, block_dim = mat_dim;
 

	 cudaEvent_t seq_start, seq_end;
	 cudaEventCreate(&seq_start);
	 cudaEventCreate(&seq_end);

       
        if(option==0)
	{
    	    cudaEventRecord(seq_start,0);		
	    sumTriangle<<<grid_dim, block_dim>>>(d_input1, d_output1, mat_dim);
    	    cudaEventRecord(seq_end,0);		
	    cudaEventSynchronize(seq_end); 
	}
        else
	{
    	    cudaEventRecord(seq_start,0);		
    	    sumTriangleWithAtomics<<<grid_dim, block_dim>>>(d_input1, d_output1, mat_dim);
    	    cudaEventRecord(seq_end,0);		
	    cudaEventSynchronize(seq_end); 
	}
	    err = cudaGetLastError();
	    if (err != cudaSuccess)
	    {
	     	fprintf(stderr, "Failed to launch process_kernel1 kernel (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }
	
	LOG_DEBUG printf("Copy output data from the CUDA device to the host memory\n");

	    err = cudaMemcpy(h_output1, d_output1, size_V, cudaMemcpyDeviceToHost);
	    if (err != cudaSuccess)
	    {
	     	fprintf(stderr, "Failed to copy vector d_input1 from device to host (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	  	
	   
	    LOG_OUTPUT print_matrix(h_output1,1,mat_dim+1);
	  

	    err = cudaFree(d_input1);
	    if (err != cudaSuccess)
	    {
	     	fprintf(stderr, "Failed to free device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    free(h_input1);


	    err = cudaFree(d_output1);
	    if (err != cudaSuccess)
	    {
	     	fprintf(stderr, "Failed to free device vector d_input1 (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }


	    free(h_output1);
	    float event_recorded_time=0.0;
	    cudaEventElapsedTime(&event_recorded_time, seq_start, seq_end);
    	    printf("Execution Time: %f\n",event_recorded_time);	    


	    err = cudaDeviceReset();
	    if (err != cudaSuccess)
	    {
	     	fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    LOG_DEBUG printf("Done\n");
	}
    return 0;
}
