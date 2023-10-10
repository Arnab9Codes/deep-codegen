#include "kernel.h"
#define block_dim 8

__global__ void normal_dot(float *a, float *b, float*c, const int M, const int P, const int N){
    
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int row = by * blockDim.y + ty; // would map the colum index of threads
    int col = bx * blockDim.x + tx; // would map the row index of thread

    if(row< M && col <N){
        float partialSum = 0.0f;
        for(int i=0;i<P;i++){
            partialSum += a[row * P + i] * b[ col+ i*N];
        }
        //__syncthreads();
        c[row* N + col] = partialSum;
        //print()
    }
}

void invoke_mat_dot(float *da, float* db, float* dc, int M, int P, int N){
    
    cudaError_t cudastatus;

    int g1 = (int)ceil((float)M/(float)block_dim);
    int g2 = (int)ceil((float)N/(float)block_dim);

    dim3 ngrids(g2, g1);
    dim3 nblocks(block_dim, block_dim);

    normal_dot<<<ngrids, nblocks>>>(da, db, dc, M, P, N);
    
    cudastatus=cudaDeviceSynchronize();
    if(cudastatus!=cudaSuccess){
        printf("some error in dot product.\n");
    }
}

__global__ void mat_transpose(float *a, float *o, const int M, const int N ){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    //if(tx ==0 && ty ==0){
       // printf("bx %d by %d tx %d ty %d \n", bx, by, tx, ty);
    //}

   o[tx * M + bx] = a[bx * N + tx] ;//  a[bx * N + tx]

}


void invoke_transpose(float *a, float *o, const int M, const int N){
    
    cudaError_t cudastatus;

    mat_transpose<<<M, N>>>(a, o, M, N);

    cudastatus=cudaDeviceSynchronize();
    if(cudastatus!=cudaSuccess){
        printf("some error in transpose.");
    }
}


void mat_dot(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output){
        //invoke_mat_dot(input1.data_ptr, input2.data_ptr, output1.data_ptr, input1.row_count, input1.col_count, input2.col_count);
    int g1 = (int)ceil((float)input1.row_count/(float)block_dim);
    int g2 = (int)ceil((float)input2.col_count/(float)block_dim);

    dim3 ngrids(g2, g1);
    dim3 nblocks(block_dim, block_dim);

    normal_dot<<<ngrids, nblocks>>>(input1.data_ptr, input2.data_ptr, output.data_ptr, input1.row_count, input1.col_count, input2.col_count);
    cudaDeviceSynchronize(); 
}

void mat_transpose(array2d_t<float>& input1, array2d_t<float>& output){
    
    cudaError_t cudastatus;

    //mat_transpose<<<M, N>>>(a, o, M, N);
    mat_transpose<<<input1.row_count, input1.col_count>>>(input1.data_ptr, output.data_ptr, input1.row_count, input1.col_count);

    cudastatus=cudaDeviceSynchronize();
    if(cudastatus!=cudaSuccess){
        printf("some error in transpose.");
    }
}

void mat_add(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output){;}