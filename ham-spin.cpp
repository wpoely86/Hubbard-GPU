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

#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <hdf5.h>
#include "ham-spin.h"

// macro to help check return status of HDF5 functions
#define HDF5_STATUS_CHECK(status) {                 \
    if(status < 0)                                  \
    std::cerr << __FILE__ << ":" << __LINE__ <<     \
    ": Problem with writing to file. Status code="  \
    << status << std::endl;                         \
}

/**
 * Constructor of the SpinHamiltonian class
 * @param L Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
SpinHamiltonian::SpinHamiltonian(int L, int Nu, int Nd, double J, double U): BareHamiltonian(L,Nu,Nd,J,U)
{
    if( (Nu+Nd) & 0x1 )
        std::cerr << "There will be trouble: make sure the number of particles is even!" << std::endl;

    Sz = Nu-Nd;

    if(Sz<0)
    {
        Sz *= -1;
        auto tmp = Nu;
        Nu = Nd;
        Nd = tmp;
        std::cout << "Swapping Nu and Nd to have positive Sz!" << std::endl;
    }
}

/**
 * Destructor of the SpinHamiltonian class
 */
SpinHamiltonian::~SpinHamiltonian()
{
}

/**
 * Builds all the up and down base kets in the momentum base
 * Sort them so we have a block diagonal matrix according to total momentum
 */
void SpinHamiltonian::BuildBase()
{
    BuildSpinBase();
}

/**
  * Builds the SpinHamiltonian matrix blocks for S = myS
  * @param myS the spin to use
  */
void SpinHamiltonian::BuildHamWithS(int myS)
{
    std::vector< std::unique_ptr<matrix> > tmp_blockmat(spinbasis->getnumblocks());

#pragma omp parallel for
    for(int i=0;i<spinbasis->getnumblocks();i++)
    {
        int K = spinbasis->getKS(i).first;
        int S = spinbasis->getKS(i).second;

        if(S!=myS)
            continue;

        std::cout << "Building K=" << K << " S=" << S << std::endl;

        int cur_dim = spinbasis->getBlock(i).getspacedim();

        // calculate all elements for this block first
        std::unique_ptr<double []> hop(new double[cur_dim]);
        std::unique_ptr<int []> interact(new int[cur_dim*cur_dim]);

        for(int a=0;a<cur_dim;a++)
        {
            auto bra = spinbasis->getBlock(i).Get(a);
            hop[a] = hopping(bra.first) + hopping(bra.second);

            for(int b=a;b<cur_dim;b++)
            {
                auto ket = spinbasis->getBlock(i).Get(b);

                interact[b+a*cur_dim] = interaction(bra.first, bra.second, ket.first, ket.second);
                interact[a+b*cur_dim] = interact[b+a*cur_dim];
            }
        }

        std::cout << "Filling hamiltonian" << std::endl;

        int mydim = spinbasis->getBlock(i).getdim();
        std::unique_ptr<matrix> tmp (new matrix (mydim,mydim));

        auto &sparse = spinbasis->getBlock(i).getSparse();

#pragma omp parallel for schedule(guided)
        for(int a=0;a<mydim;a++)
        {
            for(int b=a;b<mydim;b++)
            {
                (*tmp)(a,b) = 0;

                for(int k=0;k<sparse.NumOfElInCol(a);k++)
                    for(int l=0;l<sparse.NumOfElInCol(b);l++)
                    {
                        if(sparse.GetElementRowIndexInCol(a,k) == sparse.GetElementRowIndexInCol(b,l) )
                            (*tmp)(a,b) += sparse.GetElementInCol(a,k) * sparse.GetElementInCol(b,l) * \
                                           J * hop[sparse.GetElementRowIndexInCol(a,k)];

                        (*tmp)(a,b) += sparse.GetElementInCol(a,k) * sparse.GetElementInCol(b,l) * U/L * \
                                       interact[sparse.GetElementRowIndexInCol(a,k) + cur_dim * sparse.GetElementRowIndexInCol(b,l)];
                    }

                (*tmp)(b,a) = (*tmp)(a,b);
            }
        }

        tmp_blockmat[i] = std::move(tmp);
    }

    for(int i=0;i<tmp_blockmat.size();i++)
        if(tmp_blockmat[i])
            blockmat.push_back(std::move(tmp_blockmat[i]));
}

/**
  * Builds the full SpinHamiltonian matrix
  */
void SpinHamiltonian::BuildFullHam()
{
    blockmat.resize(spinbasis->getnumblocks());

#pragma omp parallel for
    for(int i=0;i<spinbasis->getnumblocks();i++)
    {
        int K = spinbasis->getKS(i).first;
        int S = spinbasis->getKS(i).second;

        std::cout << "Building K=" << K << " S=" << S << std::endl;

        int cur_dim = spinbasis->getBlock(i).getspacedim();

        // calculate all elements for this block first
        std::unique_ptr<double []> hop(new double[cur_dim]);
        std::unique_ptr<int []> interact(new int[cur_dim*cur_dim]);

        for(int a=0;a<cur_dim;a++)
        {
            auto bra = spinbasis->getBlock(i).Get(a);
            hop[a] = hopping(bra.first) + hopping(bra.second);

            for(int b=a;b<cur_dim;b++)
            {
                auto ket = spinbasis->getBlock(i).Get(b);

                interact[b+a*cur_dim] = interaction(bra.first, bra.second, ket.first, ket.second);
                interact[a+b*cur_dim] = interact[b+a*cur_dim];
            }
        }

        std::cout << "Filling hamiltonian" << std::endl;

        int mydim = spinbasis->getBlock(i).getdim();
        std::unique_ptr<matrix> tmp (new matrix (mydim,mydim));

        auto &sparse = spinbasis->getBlock(i).getSparse();

#pragma omp parallel for schedule(guided)
        for(int a=0;a<mydim;a++)
        {
            for(int b=a;b<mydim;b++)
            {
                (*tmp)(a,b) = 0;

                for(int k=0;k<sparse.NumOfElInCol(a);k++)
                    for(int l=0;l<sparse.NumOfElInCol(b);l++)
                    {
                        if(sparse.GetElementRowIndexInCol(a,k) == sparse.GetElementRowIndexInCol(b,l) )
                            (*tmp)(a,b) += sparse.GetElementInCol(a,k) * sparse.GetElementInCol(b,l) * \
                                           J * hop[sparse.GetElementRowIndexInCol(a,k)];

                        (*tmp)(a,b) += sparse.GetElementInCol(a,k) * sparse.GetElementInCol(b,l) * U/L * \
                                       interact[sparse.GetElementRowIndexInCol(a,k) + cur_dim * sparse.GetElementRowIndexInCol(b,l)];
                    }

                (*tmp)(b,a) = (*tmp)(a,b);
            }
        }

        blockmat[i] = std::move(tmp);
    }
}

/**
  * Builds the full SpinHamiltonian matrix but store the U part and J part
  * sperataly and save it to file after creating.
  */
void SpinHamiltonian::BuildPartFullHam()
{
    blockmat.resize(0);

#pragma omp parallel for
    for(int i=0;i<spinbasis->getnumblocks();i++)
    {
        int K = spinbasis->getKS(i).first;
        int S = spinbasis->getKS(i).second;

#pragma omp critical
        {
            std::cout << "Building K=" << K << " S=" << S << std::endl;
        }

        int cur_dim = spinbasis->getBlock(i).getspacedim();

        // calculate all elements for this block first
        std::unique_ptr<double []> hop(new double[cur_dim]);
        std::unique_ptr<int []> interact(new int[cur_dim*cur_dim]);

        for(int a=0;a<cur_dim;a++)
        {
            auto bra = spinbasis->getBlock(i).Get(a);
            hop[a] = hopping(bra.first) + hopping(bra.second);

            for(int b=a;b<cur_dim;b++)
            {
                auto ket = spinbasis->getBlock(i).Get(b);

                interact[b+a*cur_dim] = interaction(bra.first, bra.second, ket.first, ket.second);
                interact[a+b*cur_dim] = interact[b+a*cur_dim];
            }
        }

        int mydim = spinbasis->getBlock(i).getdim();
        std::unique_ptr<matrix> Jterm (new matrix (mydim,mydim));
        std::unique_ptr<matrix> Uterm (new matrix (mydim,mydim));

        auto &sparse = spinbasis->getBlock(i).getSparse();

#pragma omp parallel for schedule(guided)
        for(int a=0;a<mydim;a++)
        {
            for(int b=a;b<mydim;b++)
            {
                (*Jterm)(a,b) = 0;
                (*Uterm)(a,b) = 0;

                for(int k=0;k<sparse.NumOfElInCol(a);k++)
                    for(int l=0;l<sparse.NumOfElInCol(b);l++)
                    {
                        if(sparse.GetElementRowIndexInCol(a,k) == sparse.GetElementRowIndexInCol(b,l) )
                            (*Jterm)(a,b) += sparse.GetElementInCol(a,k) * sparse.GetElementInCol(b,l) * \
                                           1 * hop[sparse.GetElementRowIndexInCol(a,k)];

                        (*Uterm)(a,b) += sparse.GetElementInCol(a,k) * sparse.GetElementInCol(b,l) * 1.0/L * \
                                       interact[sparse.GetElementRowIndexInCol(a,k) + cur_dim * sparse.GetElementRowIndexInCol(b,l)];
                    }

                (*Jterm)(b,a) = (*Jterm)(a,b);
                (*Uterm)(b,a) = (*Uterm)(a,b);
            }
        }

#pragma omp critical
        {
            std::stringstream nameJ;
            nameJ << "Jterm-" << K << "-" << S << ".h5";
            SaveToFile(nameJ.str(),Jterm->getpointer(), mydim*mydim);

            std::stringstream nameU;
            nameU << "Uterm-" << K << "-" << S << ".h5";
            SaveToFile(nameU.str(),Uterm->getpointer(), mydim*mydim);
        }
    }
}

/**
 *  private method to calculate the interaction between a 2 particle ket and bra
 *  @param a the first base vector of the bra
 *  @param b the second base vector of the bra
 *  @param c the first base vector of the ket
 *  @param d the second base vector of the ket
  * @returns matrix element of the interaction term between the ket and the bra. You still
  * have to multiply this with the interaction strength U and divide by the lattice length L
  */
int SpinHamiltonian::interaction(myint a, myint b, myint c, myint d) const
{
    int result = 0;

    myint downket = d;
    myint upket, upbra;

    while(downket)
    {
        // select rightmost down state in the ket
        myint k2sp = downket & (~downket + 1);
        // set it to zero
        downket ^= k2sp;

        int k2 = CountBits(k2sp-1);

        int signk2 = 1;

        // when uneven, extra minus sign
        if(Nu & 0x1)
            signk2 *= -1;

        signk2 *= CalcSign(k2,Hbc,d);

        upket = c;

        while(upket)
        {
            // select rightmost up state in the ket
            myint k1sp = upket & (~upket + 1);
            // set it to zero
            upket ^= k1sp;

            int k1 = CountBits(k1sp-1);

            int signk1 = CalcSign(k1,Hbc,c);

            myint K = (k1+k2) % L;

            upbra = a;

            while(upbra)
            {
                // select rightmost up state in the ket
                myint k3sp = upbra & (~upbra + 1);
                // set it to zero
                upbra ^= k3sp;

                // k3: spin up in bra
                int k3 = CountBits(k3sp-1);

                if(! ((c ^ k1sp) == (a ^ k3sp)) )
                    continue;

                // k4: spin down in bra
                int k4 = (K-k3+L)%L;

                if(! ((1<<k4 & b) && ((d ^ k2sp) == (b ^ 1<<k4)) ))
                    continue;

                int signk3 = CalcSign(k3,Hbc,a);

                int signk4 = 1;

                // when uneven, extra minus sign
                if(Nu & 0x1)
                    signk4 *= -1;

                signk4 *= CalcSign(k4,Hbc,b);

                result += signk1*signk2*signk3*signk4;
            }
        }
    }

    return result;
}

/**
  * private method used to see if a hopping between state a and b is
  * possible and with which sign. Only for 1D Hubbard.
  * @param ket the ket to use
  * @returns matrix element of the hopping term between the ket and the bra. You still
  * have to multiply this with the hopping strength J
  */
double SpinHamiltonian::hopping(myint ket) const
{
    double result = 0;

    while(ket)
    {
        // select rightmost one
        myint hop = ket & (~ket + 1);
        // set it to zero in ket
        ket ^= hop;

        result += -2* cos(2.0*M_PI/L * CountBits(hop-1));
    }

    return result;
}

/**
 * Matrix vector product with SpinHamiltonian: y = ham*x + alpha*y
 * @param x the input vector
 * @param y the output vector
 * @param alpha the scaling value
 */
void SpinHamiltonian::mvprod(double *x, double *y, double alpha) const
{
    double beta = 1;
    int incx = 1;
    char uplo = 'U';

    int offset = 0;

    for(int B=0;B<blockmat.size();B++)
    {
        int dim = blockmat[B]->getn();

        dsymv_(&uplo,&dim,&beta,blockmat[B]->getpointer(),&dim,&x[offset],&incx,&alpha,&y[offset],&incx);

        offset += dim;
    }
}

/**
 * Exactly calculates the eigenvalues of the matrix.
 * Needs lapack.
 * @param calc_eigenvectors set to true to calculate the eigenvectors. The hamiltonian
 * matrix is then overwritten by the vectors.
 * @return the vector of the sorted eigenvalues
 */
std::vector<double> SpinHamiltonian::ExactDiagonalizeFull(bool calc_eigenvectors)
{
    std::vector<double> eigenvalues(dim);

    int offset = 0;

    for(int B=0;B<blockmat.size();B++)
    {
        int dim = blockmat[B]->getn();

        Diagonalize(dim, blockmat[B]->getpointer(), &eigenvalues[offset], calc_eigenvectors);

        offset += dim;
    }

    std::sort(eigenvalues.begin(), eigenvalues.end());

    return eigenvalues;
}

/**
 * Diagonalize the block diagonal matrix. Needs lapack. Differs from SpinHamiltonian::ExactDiagonalizeFull that
 * is diagonalize block per block and keeps the momentum associated with each eigenvalue.
 * @param calc_eigenvectors set to true to calculate the eigenvectors. The hamiltonian
 * matrix is then overwritten by the vectors.
 * @return a vector of pairs. The first member of the pair is the momentum, the second is
 * the eigenvalue. The vector is sorted to the eigenvalues.
 */
std::vector< std::tuple<int,int,double> > SpinHamiltonian::ExactSpinDiagonalizeFull(bool calc_eigenvectors)
{
    std::vector< std::tuple<int, int, double> > energy(dim);

#pragma omp parallel for
    for(int B=0;B<blockmat.size();B++)
    {
        int K = spinbasis->getKS(B).first;
        int S = spinbasis->getKS(B).second;

        int offset = 0;
        for(int i=0;i<B;i++)
            offset += blockmat[i]->getn();

        int mydim = blockmat[B]->getn();

        std::unique_ptr<double []> eigs(new double [mydim]);

        Diagonalize(mydim, blockmat[B]->getpointer(), eigs.get(), calc_eigenvectors);

        for(int i=0;i<mydim;i++)
            energy[i+offset] = std::make_tuple(K,S,eigs[i]);
    }

    // sort to energy
    std::sort(energy.begin(), energy.end(),
            [](const std::tuple<int,int,double> & a, const std::tuple<int,int,double> & b) -> bool
            {
            return std::get<2>(a) < std::get<2>(b);
            });

    return energy;
}

/**
 * Diagonalize the block diagonal matrix for a certian spin.
 * Needs lapack. Differs from SpinHamiltonian::ExactDiagonalizeFull that
 * is diagonalize block per block and keeps the momentum associated with each eigenvalue.
 * @param myS the S of the blocks to diagonalize
 * @param calc_eigenvectors set to true to calculate the eigenvectors. The hamiltonian
 * matrix is then overwritten by the vectors.
 * @return a vector of pairs. The first member of the pair is the momentum, the second is
 * the eigenvalue. The vector is sorted to the eigenvalues.
 */
std::vector< std::tuple<int,int,double> > SpinHamiltonian::ExactSpinDiagonalize(int myS, bool calc_eigenvectors)
{
    std::vector< std::tuple<int, int, double> > energy;

    std::vector< std::unique_ptr<double []> > eigs(L);
    std::vector<int> eigs_dim(L);
    int totaldim = 0;

#pragma omp parallel for
    for(int B=0;B<blockmat.size();B++)
    {
        int K = spinbasis->getKS(B).first;
        int S = spinbasis->getKS(B).second;

        if( S != myS)
            continue;

        int mydim = blockmat[B]->getn();

        std::unique_ptr<double []> cur_eigs(new double [mydim]);

        Diagonalize(mydim, blockmat[B]->getpointer(), cur_eigs.get(), calc_eigenvectors);

        eigs[K] = std::move(cur_eigs);
        eigs_dim[K] = mydim;
        totaldim += mydim;
    }

    energy.reserve(totaldim);

    for(int K=0;K<L;K++)
        if(eigs[K])
            for(int i=0;i<eigs_dim[K];i++)
                energy.push_back(std::make_tuple(K,myS,eigs[K][i]));

    // sort to energy
    std::sort(energy.begin(), energy.end(),
            [](const std::tuple<int,int,double> & a, const std::tuple<int,int,double> & b) -> bool
            {
            return std::get<2>(a) < std::get<2>(b);
            });

    return energy;
}

void SpinHamiltonian::PrintBase() const
{
    for(int i=0;i<spinbasis->getnumblocks();i++)
    {
        int K = spinbasis->getKS(i).first;
        int S = spinbasis->getKS(i).second;

        std::cout << "Block K=" << K << " S=" << S << std::endl;
        spinbasis->getBlock(i).Print();
    }
}

/**
 * Build a base with full spin symmetry using a M scheme: start from
 * hight possible S and Sz and work your way down using S^- and projections
 */
void SpinHamiltonian::BuildSpinBase()
{
    int Smax = (Nu+Nd)/2;

    BasisList basis(L, Nu, Nd);

    // number of up
    int Su = Nu+Nd;
    // number of down
    int Sd = 0;
    int cur_Sz = (Su - Sd)/2;

    // first do Sz=max (all up)
    std::unique_ptr<MomBasis> tmp_basis(new MomBasis(L,Su,Sd));
    // the K of the only possible state
    int tmp_k = ((L-1)*L)/2 % L;
    basis.Create(tmp_k,cur_Sz,cur_Sz,*tmp_basis,tmp_basis->getdimK(tmp_k));

    // go one level lower
    Su--;
    Sd++;
    cur_Sz = (Su - Sd)/2;
    tmp_basis.reset(new MomBasis(L,Su,Sd));
    basis.Create(tmp_k,cur_Sz+1,cur_Sz,*tmp_basis, basis.Get(tmp_k,cur_Sz+1,cur_Sz+1).getdim());
    basis.Get(tmp_k,cur_Sz+1,cur_Sz).Slad_min(basis.Get(tmp_k,cur_Sz+1,cur_Sz+1));
    // projection
    basis.Create(tmp_k,cur_Sz,cur_Sz,*tmp_basis,tmp_basis->getdimK(tmp_k) - basis.Get(tmp_k,cur_Sz+1,cur_Sz).getdim() );
    basis.DoProjection(tmp_k,cur_Sz,cur_Sz,*tmp_basis);

    for(int k=0;k<L;k++)
        if(k != tmp_k)
            basis.Create(k,cur_Sz,cur_Sz,*tmp_basis,tmp_basis->getdimK(k));

    basis.Clean(cur_Sz+1);

    // go one level lower
    Su--;
    Sd++;
    cur_Sz = (Su - Sd)/2;

    while( cur_Sz >= Sz )
    {
        tmp_basis.reset(new MomBasis(L,Su,Sd));

#pragma omp parallel for
        for(int K=0;K<L;K++)
        {
            int cur_dim = tmp_basis->getdimK(K);

            for(int cur_S=Smax;cur_S>cur_Sz;cur_S--)
            {
#pragma omp critical
            {
                std::cout << "At: Sz=" << cur_Sz << " K=" << K << " (" << cur_dim << ")  => S=" << cur_S << std::endl;
            }

                if(basis.Exists(K,cur_S,cur_Sz+1))
                {
                    basis.Create(K,cur_S,cur_Sz,*tmp_basis,basis.Get(K,cur_S,cur_Sz+1).getdim());

                    cur_dim -= basis.Get(K,cur_S,cur_Sz+1).getdim();
                    basis.Get(K,cur_S,cur_Sz).Slad_min(basis.Get(K,cur_S,cur_Sz+1));
                }
            }

#pragma omp critical
            {
                std::cout << "Projection to K=" << K << " S=" << cur_Sz << " Sz=" << cur_Sz << std::endl;
            }
            basis.Create(K,cur_Sz,cur_Sz,*tmp_basis,cur_dim);
            basis.DoProjection(K, cur_Sz, cur_Sz, *tmp_basis);
        }

        // drop what we don't need
        basis.Clean(cur_Sz+1);

        Su--;
        Sd++;
        cur_Sz = (Su - Sd)/2;
    }

    spinbasis.reset(new SpinBasis(L,Nu,Nd,basis));
}

void SpinHamiltonian::SaveBasis(const char *filename) const
{
    if(spinbasis)
        spinbasis->SaveBasis(filename);
}

void SpinHamiltonian::ReadBasis(const char *filename)
{
    if(spinbasis)
        spinbasis->ReadBasis(filename);
    else
        spinbasis.reset(new SpinBasis(filename));
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
