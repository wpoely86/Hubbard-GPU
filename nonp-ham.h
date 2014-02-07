/* Copyright (C) 2014  Ward Poelmans

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

#ifndef NONP_HAM_H
#define NONP_HAM_H

#include "ham.h"

/**
 * This is the class is for the 1D Hubbard without periodic boundary conditions
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class NonPeriodicHamiltonian: public Hamiltonian
{
    public:
	NonPeriodicHamiltonian(int L, int Nu, int Nd, double J, double U);
	virtual ~NonPeriodicHamiltonian();

    protected:
	int hopping(myint a, myint b, int jumpsign=0) const;
};

#endif /* NONP_HAM_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
