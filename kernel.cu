#include "kernel.h"

#define tile_dim 2


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
    }

}

__global__ void tiled_dot(float *a, float *b, float*c, const int M, const int P, const int N){// has some issues

    __shared__ float A[tile_dim][tile_dim];
    __shared__ float B[tile_dim][tile_dim];

    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int row = by * blockDim.y + ty; // would map the colum index of threads
    int col = bx * blockDim.x + tx; // would map the row index of thread

    float partialSum = 0.0f;
    int tile_count = ceil((float)P/(float)tile_dim);

    //if(tx ==0 && ty ==0){
    //    printf("tile count %d ", tile_count);
    //}
    for(int tile=0; tile < tile_count; tile++){

        if(((tile * tile_dim + tx)<P) && (row < M)){
            A[ty][tx] = a[row * P + tile * tile_dim + tx];
        } else{
            A[ty][tx] = 0.0f;
        }

        if(((tile * tile_dim + ty)<P) && (col< N)){
            B[ty][tx] = b[col + (tile * tile_dim + ty) * N];
        } else{
            B[ty][tx] = 0.0f;
        }
        __syncthreads();

        //tile computation
        for(int j = 0; j < tile_dim; j++){
            partialSum += A[ty][j] * B[j][tx];
        }
        __syncthreads();
    }

    if((row<M) && (col<N)){
        c[row*N+col] = partialSum;
        //printf("\n-%d %d %f", row, col, partialSum);
    }
    partialSum =0.0f;

}


void invoke_mat_dot(float *da, float* db, float* dc, int M, int P, int N){
    
    cudaError_t cudastatus;

    int g1 = (int)ceil((float)M/(float)tile_dim);
    int g2 = (int)ceil((float)N/(float)tile_dim);

    dim3 ngrids(g2, g1);// ------------------------------ change here
    dim3 nblocks(tile_dim, tile_dim);

    //printf("g1: %d g2: %d, tile dim: %d\n", g1, g2, tile_dim);

    //tiled_dot<<<ngrids, nblocks>>>(da, db, dc, M, P, N);
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

    //printf("\ninvoked transpose kernel.\n");

    mat_transpose<<<M, N>>>(a, o, M, N);

    cudastatus=cudaDeviceSynchronize();

    if(cudastatus!=cudaSuccess){
        printf("some error in transpose.");
    }
}


__global__ void mat_add(float *a, float *b, float*c, const int M, const int N ){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    //if(tx ==0 && ty ==0){
        //printf("bx %d by %d tx %d ty %d \n", bx, by, tx, ty);
    //}

    c[bx * N + tx] = a[bx * N + tx] + b[bx * N + tx];

}

void invoke_addition(float *da, float *db, float *dc, int M, int N){
    
    cudaError_t cudastatus;
    //printf("\n invoked addition kernel.\n");
    mat_add<<<M, N>>>(da, db, dc, M, N);
    cudastatus=cudaDeviceSynchronize();

    if(cudastatus!=cudaSuccess){
        printf("some error in addition.\n");
    }
}


__global__ void mat_mul(float *a, float *b, float*c, const int M, const int N ){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    //if(tx ==0 && ty ==0){
       // printf("bx %d by %d tx %d ty %d \n", bx, by, tx, ty);
    //}

    c[bx * N + tx] = a[bx * N + tx] * b[bx * N + tx];

}


void invoke_multiplication(float *da, float *db, float *dc, int M, int N){
    
    cudaError_t cudastatus;
    printf("\n invoked pointwise multiplication kernel.\n");
    mat_mul<<<M, N>>>(da, db, dc, M, N);
    cudastatus=cudaDeviceSynchronize();
    if(cudastatus!=cudaSuccess){
        printf("some error in multiplication.\n");
    }
}

void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm){;}
void gspmmve(graph_t& graph, array2d_t<float>& input1, array1d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmme(graph_t& graph, array1d_t<float>& edge_input, array1d_t<float>& output, op_t op, bool reverse){;}
void gspmme2d(graph_t& graph, array2d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmmve2d(graph_t& graph, array3d_t<float>& input1, array2d_t<float>& edge_input, array3d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve(graph_t& graph, array1d_t<float>& input_left, array1d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve2d(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv2d(graph_t& graph, array3d_t<float>& input_left, array3d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void test_2out(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test3(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test4(array3d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, int t){;}
void mat_norm(array2d_t<float>& input1, int M, int N){;}

void mat_dot(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, int M, int P, int N){
    invoke_mat_dot(input1.data_ptr, input2.data_ptr, output1.data_ptr, M, P, N);
}
void mat_transpose(array2d_t<float>& input1, array2d_t<float>& output1, int M, int N){
    invoke_transpose(input1.data_ptr, output1.data_ptr, M, N);
}
void mat_add(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, int M, int N){
    invoke_addition(input1.data_ptr, input2.data_ptr, output1.data_ptr, M, N);
}
void mat_mul(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, int M, int N){
    invoke_multiplication(input1.data_ptr, input2.data_ptr, output1.data_ptr, M, N);
}
