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

#ifndef HAM2D_H
#define HAM2D_H

#include "ham.h"

/**
 * This is the main class for 2D Hubbard:
 * It makes a grid of length L
 * and depth D. For example a grid of L=4 and D=2 is: \n
 * x  x  x  x \n
 * x  x  x  x \n
 * The periodic boundary condition is used.
 *
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class HubHam2D : public Hamiltonian
{
    public:
	HubHam2D(int L, int D, int Nu, int Nd, double J, double U);
	virtual ~HubHam2D();

	void BuildFullHam();

    protected:
	int hopping(myint a, myint b, int jump=0) const;

	//! The length of the 2D grid
        int L;
	//! The depth of the 2D grid
        int D;
};

#endif /* HAM2D_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
