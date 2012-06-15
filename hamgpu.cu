#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "hamgpu.h"

// number of threads in a block (must be multiple of 32)
#define NUMTHREADS 256

#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }


GPUHamiltonian::GPUHamiltonian(int Ns, int Nu, int Nd, double J, double U)
    : SparseHamiltonian(Ns,Nu,Nd,J,U)
{
}

GPUHamiltonian::~GPUHamiltonian()
{
}

__global__ void gpu_mvprod(double *x, double *y, double alpha, int NumUp, int NumDown, int dim, double *Umat, double *Down_data,unsigned int *Down_ind, int size_Down, double *Up_data, unsigned int *Up_ind, int size_Up, int rows_shared)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if(index < dim)
    {
	double result = Umat[index] * x[index];

	int sv = index / NumDown; //__fdividef(index,NumDown);
	int id = index % NumDown; // index - sv*NumDown;

	extern __shared__ double shared[];

	if(threadIdx.x <= rows_shared)
	    for(int i=0;i<size_Up;i++)
		shared[threadIdx.x*size_Up+i] = Up_data[(blockDim.x * blockIdx.x)/NumDown + threadIdx.x + i*NumUp];

	__syncthreads();

	for(int i=0;i<size_Up;i++)
//	    result += Up_data[sv+i*NumUp] * x[id + NumDown*Up_ind[sv+i*NumUp]];
	    result += shared[(sv-(blockDim.x * blockIdx.x)/NumDown)*size_Up+i] * x[id + NumDown*Up_ind[sv+i*NumUp]];

	for(int i=0;i<size_Down;i++)
	    result += Down_data[id+i*NumDown] * x[sv*NumDown + Down_ind[id+i*NumDown]];

	y[index] = alpha * y[index] + result;
    }
}

void GPUHamiltonian::mvprod(double *x, double *y, double alpha)
{
    int NumUp = baseUp.size();
    int NumDown = baseDown.size();
    int dim = NumUp*NumDown;
    int numblocks = ceil(dim*1.0/NUMTHREADS);
    int rows_shared = ceil(NUMTHREADS*1.0/NumDown);
    size_t sharedmem = size_Up * (rows_shared+1) * sizeof(double);

    cudaGetLastError();
    gpu_mvprod<<<numblocks,NUMTHREADS,sharedmem>>>(x,y,alpha,NumUp,NumDown,dim,Umat_gpu,Down_data_gpu,Down_ind_gpu,size_Down,Up_data_gpu,Up_ind_gpu,size_Up,rows_shared);
    CUDA_SAFE_CALL(cudaGetLastError());
}

double GPUHamiltonian::LanczosDiagonalize(int m)
{
    int device;
    cudaGetDevice( &device );

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, device );

    int NumUp = baseUp.size();
    int NumDown = baseDown.size();

    size_t neededmem = getDim()*sizeof(double) +
	2*NumUp*size_Up*sizeof(double) +
	2*NumDown*size_Down*sizeof(double) +
	2*dim*sizeof(double);

    if(neededmem > prop.totalGlobalMem)
    {
	std::cerr << "Houston, we have a memory problem!" << std::endl;
	return 0;
    }

    if( ceil(dim*1.0/NUMTHREADS) > prop.maxGridSize[0] )
    {
	std::cerr << "Houston, we have a grid size problem!" << std::endl;
	return 0;
    }

    if( (ceil(NUMTHREADS/NumDown) + 1) * size_Up * sizeof(double) > prop.sharedMemPerBlock )
    {
	std::cerr << "Houston, we have a shared memory size problem!" << std::endl;
	return 0;
    }

    // alloc Umat and copy to gpu
    double *Umat = Umatrix();
    CUDA_SAFE_CALL(cudaMalloc(&Umat_gpu, getDim()*sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(Umat_gpu,Umat,getDim()*sizeof(double),cudaMemcpyHostToDevice));

    delete [] Umat;


    CUDA_SAFE_CALL(cudaMalloc(&Up_data_gpu,NumUp*size_Up*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&Up_ind_gpu,NumUp*size_Up*sizeof(double)));

    CUDA_SAFE_CALL(cudaMemcpy(Up_data_gpu,Up_data,NumUp*size_Up*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Up_ind_gpu,Up_ind,NumUp*size_Up*sizeof(double),cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(&Down_data_gpu,NumDown*size_Down*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&Down_ind_gpu,NumDown*size_Down*sizeof(double)));

    CUDA_SAFE_CALL(cudaMemcpy(Down_data_gpu,Down_data,NumDown*size_Down*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Down_ind_gpu,Down_ind,NumDown*size_Down*sizeof(double),cudaMemcpyHostToDevice));

    double *a = new double[m];
    double *b = new double[m];

    double *qa = new double [dim];
    double *qb = new double [dim];

    double *qa_gpu;
    double *qb_gpu;
    CUDA_SAFE_CALL(cudaMalloc(&qa_gpu,dim*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&qb_gpu,dim*sizeof(double)));

    int i;

    b[0] = 0;
    // does nothing, just to disable valgrind warnings
    a[m-1] = 0;

    srand(time(0));

    for(i=0;i<dim;i++)
    {
        qa[i] = 0;
        qb[i] = rand()*10.0/RAND_MAX;
    }

    int incx = 1;

    double norm = 1.0/sqrt(ddot_(&dim,qb,&incx,qb,&incx));

    dscal_(&dim,&norm,qb,&incx);

    CUDA_SAFE_CALL(cudaMemcpy(qa_gpu,qa,dim*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(qb_gpu,qb,dim*sizeof(double),cudaMemcpyHostToDevice));

    delete [] qa;
    delete [] qb;

    norm = 1;

    double *f1 = qa_gpu;
    double *f2 = qb_gpu;
    double *tmp;

    double alpha = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    for(i=1;i<m;i++)
    {
        alpha = -b[i-1];
	cublasDscal(handle,dim,&alpha,f1,1);

	mvprod(f2,f1,norm);

	cublasDdot(handle,dim,f1,1,f2,1,&a[i-1]);

        alpha = -a[i-1];
	cublasDaxpy(handle,dim,&alpha,f2,1,f1,1);

	cublasDdot(handle,dim,f1,1,f1,1,&b[i]);
	b[i] = sqrt(b[i]);

        if( fabs(b[i]) < 1e-10 )
            break;

        alpha = 1.0/b[i];

	cublasDscal(handle,dim,&alpha,f1,1);

        tmp = f2;
        f2 = f1;
        f1 = tmp;
    }

    cublasDestroy(handle);

    char jobz = 'N';
    int info;

    dstev_(&jobz,&m,a,&b[1],&alpha,&m,&alpha,&info);

    if(info != 0)
        std::cerr << "Error in Lanczos" << std::endl;

    alpha = a[0];

    delete [] a;
    delete [] b;

    CUDA_SAFE_CALL(cudaFree(qa_gpu));
    CUDA_SAFE_CALL(cudaFree(qb_gpu));

    CUDA_SAFE_CALL(cudaFree(Up_data_gpu));
    CUDA_SAFE_CALL(cudaFree(Up_ind_gpu));
    CUDA_SAFE_CALL(cudaFree(Down_data_gpu));
    CUDA_SAFE_CALL(cudaFree(Down_ind_gpu));

    CUDA_SAFE_CALL(cudaFree(Umat_gpu));

    return alpha;
}


