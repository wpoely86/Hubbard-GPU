#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include "hamgpu.h"

GPUHamiltonian::GPUHamiltonian(int Ns, int Nu, int Nd, double J, double U)
    : SparseHamiltonian(Ns,Nu,Nd,J,U)
{
}

GPUHamiltonian::~GPUHamiltonian()
{
}

__global__ void gpu_mvprod(double *x, double *y, double alpha, int NumUp, int NumDown, int dim, double *Umat, double *Down_data,unsigned int *Down_ind, int size_Down, double *Up_data, unsigned int *Up_ind, int size_Up)
{
    y[threadIdx.x] = alpha * y[threadIdx.x] + Umat[threadIdx.x] * x[threadIdx.x];

    int sv = threadIdx.x / NumDown;
    int id = threadIdx.x % NumDown;

    for(int i=0;i<size_Down;i++)
	y[threadIdx.x] += Down_data[id+i*NumDown] * x[sv*NumDown + Down_ind[id+i*NumDown]];

    for(int i=0;i<size_Up;i++)
	y[threadIdx.x] += Up_data[sv+i*NumUp] * x[id + NumDown*Up_ind[sv+i*NumUp]];
}

void GPUHamiltonian::mvprod(double *x, double *y, double alpha)
{
    int NumUp = baseUp.size();
    int NumDown = baseDown.size();
    int dim = NumUp*NumDown;

    gpu_mvprod<<<1,dim>>>(x,y,alpha,NumUp,NumDown,dim,Umat_gpu,Down_data_gpu,Down_ind_gpu,size_Down,Up_data_gpu,Up_ind_gpu,size_Up);
}

double GPUHamiltonian::LanczosDiagonalize(int m)
{
    // alloc Umat and copy to gpu
    double *Umat = Umatrix();
    cudaMalloc(&Umat_gpu, getDim()*sizeof(double));
    cudaMemcpy(Umat_gpu,Umat,getDim()*sizeof(double),cudaMemcpyHostToDevice);

    delete [] Umat;

    int NumUp = baseUp.size();
    int NumDown = baseDown.size();

    cudaMalloc(&Up_data_gpu,NumUp*size_Up*sizeof(double));
    cudaMalloc(&Up_ind_gpu,NumUp*size_Up*sizeof(double));

    cudaMemcpy(Up_data_gpu,Up_data,NumUp*size_Up*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Up_ind_gpu,Up_ind,NumUp*size_Up*sizeof(double),cudaMemcpyHostToDevice);

    cudaMalloc(&Down_data_gpu,NumDown*size_Up*sizeof(double));
    cudaMalloc(&Down_ind_gpu,NumDown*size_Up*sizeof(double));

    cudaMemcpy(Down_data_gpu,Down_data,NumDown*size_Down*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Down_ind_gpu,Down_ind,NumDown*size_Down*sizeof(double),cudaMemcpyHostToDevice);

    double *a = new double[m];
    double *b = new double[m];

    double *qa = new double [dim];
    double *qb = new double [dim];

    double *qa_gpu;
    double *qb_gpu;
    cudaMalloc(&qa_gpu,dim*sizeof(double));
    cudaMalloc(&qb_gpu,dim*sizeof(double));

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

    cudaMemcpy(qa_gpu,qa,dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(qb_gpu,qb,dim*sizeof(double),cudaMemcpyHostToDevice);

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

    cudaFree(qa);
    cudaFree(qb);

    cudaFree(Up_data_gpu);
    cudaFree(Up_ind_gpu);
    cudaFree(Down_data_gpu);
    cudaFree(Down_ind_gpu);

    cudaFree(Umat_gpu);

    return alpha;
}


