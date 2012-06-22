#include <iostream>
#include <cstdlib>
#include <cmath>
#include "hamsparse2D.h"

/**
 * Constructor of the SparseHamiltonian2D class
 * @param L The Length of the 2D grid
 * @param D The depth of the 2D grid
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
SparseHamiltonian2D::SparseHamiltonian2D(int L,int D, int Nu, int Nd, double J, double U)
    : SparseHamiltonian2D_CSR(L,D,Nu,Nd,J,U)
{
    Up_data = 0;
    Down_data = 0;

    Up_ind = 0;
    Down_ind = 0;

    size_Up = 0;
    size_Down = 0;
}

/**
 * Destructor of the SparseHamiltonian2D class
 */
SparseHamiltonian2D::~SparseHamiltonian2D()
{
    if(Up_data)
	delete [] Up_data;

    if(Down_data)
	delete [] Down_data;

    if(Up_ind)
	delete [] Up_ind;

    if(Down_ind)
	delete [] Down_ind;
}

/**
 * Builds and fills the sparse hamiltonian
 */
void SparseHamiltonian2D::BuildSparseHam()
{
    SparseHamiltonian2D_CSR::BuildSparseHam();

    unsigned int NumUp = baseUp.size();
    unsigned int NumDown = baseDown.size();

    int max = 0;
    for(unsigned int i=0;i<NumUp;i++)
        if( (Up_row[i+1]-Up_row[i]) > max )
            max = Up_row[i+1]-Up_row[i];

    size_Up = max;

    max = 0;
    for(unsigned int i=0;i<NumDown;i++)
        if( (Down_row[i+1]-Down_row[i]) > max )
            max = Down_row[i+1]-Down_row[i];

    size_Down = max;

    Up_data = new double [NumUp*size_Up];
    Up_ind = new unsigned int [NumUp*size_Up];

    Down_data = new double [NumDown*size_Down];
    Down_ind = new unsigned int [NumDown*size_Down];

    std::cout << "Sizes: " << size_Up << "\t" << size_Down << std::endl;

    for(unsigned int a=0;a<NumUp;a++)
    {
	int count = 0;

	for(unsigned int b=Up_row[a];b<Up_row[a+1];b++)
	{
            Up_data[a+count*NumUp] = Up_data_CSR[b];
            Up_ind[a+count*NumUp] = Up_col[b];
            count++;
	}

	if(count < size_Up)
	    for(int i=count;i<size_Up;i++)
	    {
		Up_data[a+i*NumUp] = 0;
		Up_ind[a+i*NumUp] = 0;
	    }
	else if(count > size_Up)
	    std::cerr << "Something went terribly wrong!" << std::endl;
    }

    for(unsigned int a=0;a<NumDown;a++)
    {
	int count = 0;

	for(unsigned int b=Down_row[a];b<Down_row[a+1];b++)
	{
            Down_data[a+count*NumDown] = Down_data_CSR[b];
            Down_ind[a+count*NumDown] = Down_col[b];
            count++;
	}

	if(count < size_Down)
	    for(int i=count;i<size_Down;i++)
	    {
		Down_data[a+i*NumDown] = 0;
		Down_ind[a+i*NumDown] = 0;
	    }
	else if(count > size_Down)
	    std::cerr << "Something went terribly wrong!" << std::endl;
    }
}

/**
 * Prints the Sparse Hamiltonian
 */
void SparseHamiltonian2D::PrintSparse() const
{
    unsigned int NumUp = baseUp.size();
    unsigned int NumDown = baseDown.size();

    std::cout << "Up:" << std::endl;

    for(unsigned int i=0;i<NumUp;i++)
    {
	int count = 0;

	for(unsigned int j=0;j<NumUp;j++)
	    if(count < size_Up && Up_ind[i+count*NumUp] == j)
		std::cout << Up_data[i + NumUp*count++] << "\t";
	    else
		std::cout << "0\t";

	std::cout << std::endl;
    }

    std::cout << "Down:" << std::endl;

    for(unsigned int i=0;i<NumDown;i++)
    {
	int count = 0;

	for(unsigned int j=0;j<NumDown;j++)
	    if(count < size_Down && Down_ind[i + count*NumDown] == j)
		std::cout << Down_data[i + NumDown*count++] << "\t";
	    else
		std::cout << "0\t";

	std::cout << std::endl;
    }
}

void SparseHamiltonian2D::PrintRawELL() const
{
    unsigned int NumUp = baseUp.size();
    unsigned int NumDown = baseDown.size();

    std::cout << "Up:" << std::endl;

    std::cout << "Data:" << std::endl;
    for(unsigned int i=0;i<NumUp;i++)
    {
	for(int j=0;j<size_Up;j++)
	    std::cout << Up_data[i+j*NumUp] << "\t";

	std::cout << std::endl;
    }

    std::cout << "Indices:" << std::endl;
    for(unsigned int i=0;i<NumUp;i++)
    {
	for(int j=0;j<size_Up;j++)
	    std::cout << Up_ind[i+j*NumUp] << "\t";

	std::cout << std::endl;
    }

    std::cout << "Down:" << std::endl;

    std::cout << "Data:" << std::endl;
    for(unsigned int i=0;i<NumDown;i++)
    {
	for(int j=0;j<size_Down;j++)
	    std::cout << Down_data[i+j*NumDown] << "\t";

	std::cout << std::endl;
    }

    std::cout << "Indices:" << std::endl;
    for(unsigned int i=0;i<NumDown;i++)
    {
	for(int j=0;j<size_Down;j++)
	    std::cout << Down_ind[i+j*NumDown] << "\t";

	std::cout << std::endl;
    }
}

/**
 * Matrix vector product with (sparse) hamiltonian: y = ham*x + alpha*y
 * @param x the x vector
 * @param y the y vector
 * @param alpha the scaling value
 */
void SparseHamiltonian2D::mvprod(double *x, double *y, double alpha) const
{
    int NumUp = baseUp.size();
    int NumDown = baseDown.size();

    int incx = 1;

    for(int i=0;i<NumUp;i++)
    {
#pragma omp parallel for
        for(int k=0;k<NumDown;k++)
        {
            // the interaction part
            y[i*NumDown+k] = alpha*y[i*NumDown+k] + U * CountBits(baseUp[i] & baseDown[k]) * x[i*NumDown+k];

            // the Hop_down part
            for(int l=0;l<size_Down;l++)
                y[i*NumDown+k] += Down_data[k+l*NumDown] * x[i*NumDown + Down_ind[k+l*NumDown]];
        }

        // the Hop_Up part
        for(int l=0;l<size_Up;l++)
            daxpy_(&NumDown,&Up_data[i+l*NumUp],&x[Up_ind[i+l*NumUp]*NumDown],&incx,&y[i*NumDown],&incx);
    }
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
