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

#ifndef HAM_SPIN_H
#define HAM_SPIN_H

#include <vector>
#include <memory>
#include <tuple>
#include <cstring>

#include "bare-ham.h"
#include "helpers.h"


/**
 * This is the main class for spin symmetry. It calculates the full SpinHamiltonian matrix for 1D Hubbard. It can both exact diagonlize
 * or use a Lanczos algorithm to calculate the groundstate energy.
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class SpinHamiltonian: public BareHamiltonian
{
    public:
        SpinHamiltonian(int L, int Nu, int Nd, double J, double U);
        virtual ~SpinHamiltonian();

        void BuildBase();
        void BuildFullHam();

        void BuildPartFullHam();

        void BuildHamWithS(int myS);

        virtual void mvprod(double *x, double *y, double alpha) const;

        std::vector<double> ExactDiagonalizeFull(bool calc_eigenvectors);

        virtual std::vector< std::tuple<int,int,double> > ExactSpinDiagonalizeFull(bool calc_eigenvectors);

        std::vector< std::tuple<int,int,double> > ExactSpinDiagonalize(int myS, bool calc_eigenvectors);

        void BuildSpinBase();

        void PrintBase() const;

        void SaveBasis(const char *filename) const;

        void ReadBasis(const char *filename);

    protected:
        int Sz;

        double hopping(myint) const;

        int interaction(myint, myint, myint, myint) const;

        std::vector< std::unique_ptr<matrix> > blockmat;

        std::unique_ptr<class SpinBasis> spinbasis;
};


#endif /* HAM_SPIN_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
