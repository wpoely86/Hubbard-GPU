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

#ifndef HAM_MOM_H
#define HAM_MOM_H

#include <vector>
#include <memory>

#include "bare-ham.h"


/**
 * This is the main (base) class. It calculates the full MomHamiltonian matrix for 1D Hubbard. It can both exact diagonlize
 * or use a Lanczos algorithm to calculate the groundstate energy.
 * @author Ward Poelmans <wpoely86@gmail.com>
 */
class MomHamiltonian: public BareHamiltonian
{
    public:
        MomHamiltonian(int L, int Nu, int Nd, double J, double U);
        virtual ~MomHamiltonian();

        void BuildBase();
        void BuildFullHam();

        virtual void mvprod(double *x, double *y, double alpha) const;

        std::vector<double> ExactDiagonalizeFull(bool calc_eigenvectors);

        virtual std::vector< std::pair<int,double> > ExactMomDiagonalizeFull(bool calc_eigenvectors);

        void GenerateData(double Ubegin, double Uend, double step, std::string filename);

        void GenerateCurve(double Ubegin, double Uend, double step, std::string filename);

    protected:
        double hopping(myint) const;

        int interaction(int, int, int, int) const;

        std::vector< std::vector< std::pair<int,int> > > mombase;

        std::vector< std::unique_ptr<double []> > blockmat;
};

#endif /* HAM_MOM_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
