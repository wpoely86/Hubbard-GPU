#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "hamgpu.h"

// number of threads in a block (must be multiple of 32)
#define NUMTHREADS 128

#define GRIDSIZE 65535

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
    int index = threadIdx.x + blockDim.x * blockIdx.x + blockIdx.y * blockDim.x * gridDim.x;

    if(index < dim)
    {
	double result = Umat[index] * x[index];

	int sv = index / NumDown; //__fdividef(index,NumDown);
	int id = index % NumDown; // index - sv*NumDown;

	extern __shared__ double shared[];

	unsigned int *shared_ind = (unsigned int *) &shared[size_Up * rows_shared];

	int s_sv = (blockDim.x * blockIdx.x + blockIdx.y * blockDim.x * gridDim.x)/NumDown;

	if(threadIdx.x < rows_shared && (s_sv + threadIdx.x) < NumUp)
	    for(int i=0;i<size_Up;i++)
	    {
		shared[i*rows_shared+threadIdx.x] = Up_data[s_sv + threadIdx.x + i*NumUp];

		shared_ind[i*rows_shared+threadIdx.x] = Up_ind[s_sv + threadIdx.x + i*NumUp];
	    }

	__syncthreads();

	for(int i=0;i<size_Up;i++)
	    // result += Up_data[sv+i*NumUp] * x[id + NumDown*Up_ind[sv+i*NumUp]];
	    result += shared[sv-s_sv+i*rows_shared] * x[id + NumDown*shared_ind[sv-s_sv+i*rows_shared]];

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
    dim3 numblocks(ceil(dim*1.0/NUMTHREADS));
    int rows_shared = ceil(NUMTHREADS*1.0/NumDown) + 1;
    size_t sharedmem = size_Up * rows_shared * (sizeof(double) + sizeof(unsigned int));

    if(numblocks.x > GRIDSIZE)
    {
	numblocks.x = GRIDSIZE;
	numblocks.y = ceil(ceil(dim*1.0/NUMTHREADS)*1.0/GRIDSIZE);
    }

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
	NumUp*size_Up*(sizeof(double)+sizeof(unsigned int)) +
	NumDown*size_Down*(sizeof(double)+sizeof(unsigned int)) +
	2*dim*sizeof(double);

    if(neededmem > prop.totalGlobalMem)
    {
	std::cerr << "Houston, we have a memory problem!" << std::endl;
	return 0;
    }

    if( ceil(dim*1.0/NUMTHREADS) > (1.0*prop.maxGridSize[0]*prop.maxGridSize[1]) ) // convert all to doubles to avoid int overflow
    {
	std::cerr << "Houston, we have a grid size problem!" << std::endl;
	return 0;
    }

    if( size_Up * (ceil(NUMTHREADS*1.0/NumDown)+1) * (sizeof(double) + sizeof(unsigned int)) > prop.sharedMemPerBlock )
    {
	std::cerr << "Houston, we have a shared memory size problem!" << std::endl;
	return 0;
    }

    // alloc Umat and copy to gpu
    double *Umat = Umatrix();
    CUDA_SAFE_CALL(cudaMalloc(&Umat_gpu, dim*sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(Umat_gpu,Umat,dim*sizeof(double),cudaMemcpyHostToDevice));

    delete [] Umat;


    CUDA_SAFE_CALL(cudaMalloc(&Up_data_gpu,NumUp*size_Up*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&Up_ind_gpu,NumUp*size_Up*sizeof(unsigned int)));

    CUDA_SAFE_CALL(cudaMemcpy(Up_data_gpu,Up_data,NumUp*size_Up*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Up_ind_gpu,Up_ind,NumUp*size_Up*sizeof(unsigned int),cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(&Down_data_gpu,NumDown*size_Down*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&Down_ind_gpu,NumDown*size_Down*sizeof(unsigned int)));

    CUDA_SAFE_CALL(cudaMemcpy(Down_data_gpu,Down_data,NumDown*size_Down*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Down_ind_gpu,Down_ind,NumDown*size_Down*sizeof(unsigned int),cudaMemcpyHostToDevice));

    std::vector<double> a(m,0);
    std::vector<double> b(m,0);

    double *qa = new double [dim];
    double *qb = new double [dim];

    double *qa_gpu;
    double *qb_gpu;
    CUDA_SAFE_CALL(cudaMalloc(&qa_gpu,dim*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&qb_gpu,dim*sizeof(double)));

    srand(time(0));

    for(int i=0;i<dim;i++)
    {
        qa[i] = 0;
        qb[i] = (rand()*10.0/RAND_MAX);
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

//    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
//    cublasSetPointerMode(handle,mode);

    int i=1;

    std::vector<double> acopy(a);
    std::vector<double> bcopy(b);

    double E = 1;

    cudaEvent_t start, stop;
    float exeTime;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord(start, 0);

    while(fabs(E-acopy[0]) > 1e-4)
    {
	E = acopy[0];

	for(;i<m;i++)
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

	acopy = a;
	bcopy = b;

	char jobz = 'N';
	int info;

	dstev_(&jobz,&m,acopy.data(),&bcopy.data()[1],&alpha,&m,&alpha,&info);

	if(info != 0)
	    std::cerr << "Error in Lanczos" << std::endl;

	m += 10;
	a.resize(m);
	b.resize(m);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &exeTime, start, stop );

    std::cout << "Done in " << m-10 << " Iterations" << std::endl;
    std::cout << "Cuda time: " << exeTime/1000 << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);

    alpha = acopy[0];

    CUDA_SAFE_CALL(cudaFree(qa_gpu));
    CUDA_SAFE_CALL(cudaFree(qb_gpu));

    CUDA_SAFE_CALL(cudaFree(Up_data_gpu));
    CUDA_SAFE_CALL(cudaFree(Up_ind_gpu));
    CUDA_SAFE_CALL(cudaFree(Down_data_gpu));
    CUDA_SAFE_CALL(cudaFree(Down_ind_gpu));

    CUDA_SAFE_CALL(cudaFree(Umat_gpu));

    CUDA_SAFE_CALL(cudaDeviceReset());

    return alpha;
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
