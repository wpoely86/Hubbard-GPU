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

#ifndef HAM_H
#define HAM_H

#include <vector>
#include <string>
#include "bare-ham.h"

/**
 * This is the main (base) class. It calculates the full Hamiltonian matrix for 1D Hubbard. It can both exact diagonlize
 * or use a Lanczos algorithm to calculate the groundstate energy.
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class Hamiltonian: public BareHamiltonian
{
    public:
	Hamiltonian(int L, int Nu, int Nd, double J, double U);
	virtual ~Hamiltonian();

	void BuildFullHam();

        virtual void mvprod(double *x, double *y, double alpha) const;

    protected:
	virtual int hopping(myint a, myint b, int jumpsign) const;
};

#endif /* HAM_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
