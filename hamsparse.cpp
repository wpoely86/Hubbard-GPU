/* Copyright (C) 2012-2014  Ward Poelmans

This file is part of Hubbard-GPU.

Hubbard-GPU is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Hubbard-GPU is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Hubbard-GPU.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "hamsparse.h"
#include "ham.h"
#include "nonp-ham.h"

/**
 * Constructor of the SparseHamiltonian class
 * @param L Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
template<class Ham>
SparseHamiltonian<Ham>::SparseHamiltonian(int L, int Nu, int Nd, double J, double U)
    : Ham(L,Nu,Nd,J,U)
{
    Up_data = 0;
    Down_data = 0;

    Up_ind = 0;
    Down_ind = 0;

    size_Up = 0;
    size_Down = 0;
}

/**
 * Destructor of the SparseHamiltonian class
 */
template<class Ham>
SparseHamiltonian<Ham>::~SparseHamiltonian()
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

template<class Ham>
void SparseHamiltonian<Ham>::BuildHam()
{
    BuildSparseHam();
}

/**
 * Builds and fills the sparse hamiltonian
 */
template<class Ham>
void SparseHamiltonian<Ham>::BuildSparseHam()
{
    if( !Ham::baseUp.size() || !Ham::baseDown.size() )
    {
	std::cerr << "Build base before building Hamiltonian" << std::endl;
	return;
    }

    unsigned int NumUp = Ham::baseUp.size();
    unsigned int NumDown = Ham::baseDown.size();

    int upjumpsign, downjumpsign;

    if( Ham::Nu % 2 == 0)
	upjumpsign = -1;
    else
	upjumpsign = 1;

    if( Ham::Nd % 2 == 0)
	downjumpsign = -1;
    else
	downjumpsign = 1;

    size_Up = ((Ham::L-Ham::Nu)>Ham::Nu) ? 2*Ham::Nu : 2*(Ham::L-Ham::Nu);
    size_Down = ((Ham::L-Ham::Nd)>Ham::Nd) ? 2*Ham::Nd : 2*(Ham::L-Ham::Nd);

    Up_data = new double [NumUp*size_Up];
    Up_ind = new unsigned int [NumUp*size_Up];

    Down_data = new double [NumDown*size_Down];
    Down_ind = new unsigned int [NumDown*size_Down];

    for(unsigned int a=0;a<NumUp;a++)
    {
	int count = 0;

	for(unsigned int b=0;b<NumUp;b++)
	{
	    int result = Ham::hopping(Ham::baseUp[a], Ham::baseUp[b],upjumpsign);

	    if(result != 0)
	    {
		Up_data[a+count*NumUp] = Ham::J*result;
		Up_ind[a+count*NumUp] = b;
		count++;
	    }
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

	for(unsigned int b=0;b<NumDown;b++)
	{
	    int result = Ham::hopping(Ham::baseDown[a], Ham::baseDown[b],downjumpsign);

	    if(result != 0)
	    {
		Down_data[a+count*NumDown] = Ham::J*result;
		Down_ind[a+count*NumDown] = b;
		count++;
	    }
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
template<class Ham>
void SparseHamiltonian<Ham>::PrintSparse() const
{
    unsigned int NumUp = Ham::baseUp.size();
    unsigned int NumDown = Ham::baseDown.size();

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

/**
 * Prints the Raw ELL array's.
 */
template<class Ham>
void SparseHamiltonian<Ham>::PrintRawELL() const
{
    unsigned int NumUp = Ham::baseUp.size();
    unsigned int NumDown = Ham::baseDown.size();

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
 * @param x the input vector
 * @param y the output vector
 * @param alpha the scaling value
 */
template<class Ham>
void SparseHamiltonian<Ham>::mvprod(double *x, double *y, double alpha) const
{
    int NumUp = Ham::baseUp.size();
    int NumDown = Ham::baseDown.size();

    int incx = 1;

    for(int i=0;i<NumUp;i++)
    {
#pragma omp parallel for
        for(int k=0;k<NumDown;k++)
        {
            // the interaction part
            y[i*NumDown+k] = alpha*y[i*NumDown+k] + Ham::U * Ham::CountBits(Ham::baseUp[i] & Ham::baseDown[k]) * x[i*NumDown+k];

            // the Hop_down part
            for(int l=0;l<size_Down;l++)
                y[i*NumDown+k] += Down_data[k+l*NumDown] * x[i*NumDown + Down_ind[k+l*NumDown]];
        }

        // the Hop_Up part
        for(int l=0;l<size_Up;l++)
            daxpy_(&NumDown,&Up_data[i+l*NumUp],&x[Up_ind[i+l*NumUp]*NumDown],&incx,&y[i*NumDown],&incx);
    }
}

/**
 * Builds the interaction diagonal
 * @return pointer to interaction vector. You have to free this yourself
 */
template<class Ham>
double* SparseHamiltonian<Ham>::Umatrix() const
{
    double *Umat = new double[Ham::getDim()];

    int NumDown = Ham::baseDown.size();

    for(int i=0;i<Ham::getDim();i++)
    {
	int a = i / NumDown;
	int b = i % NumDown;
	Umat[i] = Ham::U * Ham::CountBits(Ham::baseUp[a] & Ham::baseDown[b]);
    }

    return Umat;
}

// Expliciet specify the template class with the possible template parameters
template class SparseHamiltonian<Hamiltonian>;
template class SparseHamiltonian<NonPeriodicHamiltonian>;

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
