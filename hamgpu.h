/* Copyright (C) 2012  Ward Poelmans

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

#ifndef HAMGPU_H
#define HAMGPU_H

/**
 * This template class is used to do the Lanczos calculation
 * on the GPU.
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
template<class T>
class GPUHamiltonian: public T
{
    public:
	GPUHamiltonian(int L, int Nu, int Nd, double J, double U);
	GPUHamiltonian(int L, int D, int Nu, int Nd, double J, double U);
	virtual ~GPUHamiltonian();

        virtual void mvprod(double *x, double *y, double alpha) const;

        double LanczosDiagonalize(int m=0);

    protected:

	//! The diagonal of the full hamiltonian (onsite interaction)
        double *Umat_gpu;

	//! The array to hold the data (ELL format) for the up hamiltonian
	double *Up_data_gpu;
	//! The array to hold the data (ELL format) for the down hamiltonian
	double *Down_data_gpu;

	//! The array to hold the indices for the up hamiltonian
	unsigned int *Up_ind_gpu;
	//! The array to hold the indices for the down hamiltonian
	unsigned int *Down_ind_gpu;

};

#endif /* HAMGPU_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
