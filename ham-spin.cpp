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
void SpinHamiltonian::BuildFullHam()
{
    BuildHamWithS(-1);
}

/**
  * Builds the full SpinHamiltonian matrix
  */
void SpinHamiltonian::BuildHamWithS(int myS)
{
    blockmat.resize(spinbasis->getnumblocks());

#pragma omp parallel for
    for(int i=0;i<spinbasis->getnumblocks();i++)
    {
        int K = spinbasis->getKS(i).first;
        int S = spinbasis->getKS(i).second;

        if(myS != -1 && S != myS)
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
    int mydim = 0;

    for(int B=0;B<blockmat.size();B++)
        if(blockmat[B])
            mydim += blockmat[B]->getn();

    std::vector<double> eigenvalues(mydim);

    int offset = 0;

    for(int B=0;B<blockmat.size();B++)
    {
        if(!blockmat[B])
            continue;

        int dim = blockmat[B]->getn();

        Diagonalize(dim, blockmat[B]->getpointer(), &eigenvalues[offset], calc_eigenvectors);

        offset += dim;
    }

    std::sort(eigenvalues.begin(), eigenvalues.end());

    return eigenvalues;
}

/**
 * Diagonalize the block diagonal matrix. Needs lapack. Differs from SpinHamiltonian::ExactDiagonalizeFull that
 * is diagonalize block per block and keeps the momentum and spin associated with each eigenvalue.
 * @param calc_eigenvectors set to true to calculate the eigenvectors. The hamiltonian
 * matrix is then overwritten by the vectors.
 * @return a vector of tuples. The first member of the pair is the momentum, the second is
 * the spin and the thirth the actual eigenvalue. The vector is sorted to the eigenvalues.
 */
std::vector< std::tuple<int,int,double> > SpinHamiltonian::ExactSpinDiagonalizeFull(bool calc_eigenvectors)
{
    std::vector< std::tuple<int, int, double> > energy;

    std::vector< std::tuple<int,int,class matrix> > eigs (blockmat.size());

#pragma omp parallel for
    for(int B=0;B<blockmat.size();B++)
    {
        if(!blockmat[B])
            continue;

        int K = spinbasis->getKS(B).first;
        int S = spinbasis->getKS(B).second;

        int mydim = blockmat[B]->getn();

        matrix cur_eigs(mydim,1);

        Diagonalize(mydim, blockmat[B]->getpointer(), cur_eigs.getpointer(), calc_eigenvectors);

        eigs[B] = std::make_tuple(K, S, std::move(cur_eigs));
    }

    for(int i=0;i<eigs.size();i++)
        for(int j=0;j<std::get<2>(eigs[i]).getn();j++)
            energy.push_back(std::make_tuple(std::get<0>(eigs[i]), std::get<1>(eigs[i]), std::get<2>(eigs[i])[j]));

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

/**
 * This methods calculates the eigenvalues for a range of U values. The interval is [Ubegin, Uend]
 * with a stepsize of step. The resulting data is written to a file in the HDF5 file format.
 * @param Ubegin the startpoint for U
 * @param Uend the endpoint for U. We demand Ubegin < Uend
 * @param step the stepsize to use for U
 * @param filename the name of the file write to written the eigenvalues to
 */
void SpinHamiltonian::GenerateData(double Ubegin, double Uend, double step, std::string filename)
{
    hid_t       file_id, group_id, dataset_id, dataspace_id, attribute_id;
    herr_t      status;

    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    HDF5_STATUS_CHECK(file_id);

    group_id = H5Gcreate(file_id, "run", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    HDF5_STATUS_CHECK(group_id);

    dataspace_id = H5Screate(H5S_SCALAR);

    attribute_id = H5Acreate (group_id, "L", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_INT, &L );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (group_id, "Nu", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_INT, &Nu );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (group_id, "Nd", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_INT, &Nd );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (group_id, "J", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_DOUBLE, &J );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (group_id, "Ubegin", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_DOUBLE, &Ubegin );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (group_id, "Uend", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_DOUBLE, &Uend );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Acreate (group_id, "Ustep", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_DOUBLE, &step );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    status = H5Sclose(dataspace_id);
    HDF5_STATUS_CHECK(status);

    status = H5Gclose(group_id);
    HDF5_STATUS_CHECK(status);

    status = H5Fclose(file_id);
    HDF5_STATUS_CHECK(status);

    std::vector< std::unique_ptr<double []> > all_eigs;
    all_eigs.resize(blockmat.size());

    double Ucur = Ubegin;

    while(Ucur <= Uend)
    {
        std::cout << "U = " << Ucur << std::endl;
        setU(Ucur);

        BuildFullHam();

#pragma omp parallel for
        for(int B=0;B<blockmat.size();B++)
        {
            int mydim = blockmat[B]->getn();

            std::unique_ptr<double []> eigs(new double [mydim]);

            Diagonalize(mydim, blockmat[B]->getpointer(), eigs.get(), false);

            all_eigs[B] = std::move(eigs);
        }

        hid_t U_id;
        std::stringstream name;
        name << std::setprecision(5) << std::fixed << "/run/" << Ucur;

        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        HDF5_STATUS_CHECK(file_id);

        U_id = H5Gcreate(file_id, name.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        HDF5_STATUS_CHECK(U_id);

        for(int B=0;B<all_eigs.size();B++)
        {
            int K = spinbasis->getKS(B).first;
            int S = spinbasis->getKS(B).second;

            int mydim = blockmat[B]->getn();

            hsize_t dimarr = mydim;

            dataspace_id = H5Screate_simple(1, &dimarr, NULL);

            std::stringstream cur_block;
            cur_block << B;
            dataset_id = H5Dcreate(U_id, cur_block.str().c_str(), H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, all_eigs[B].get() );
            HDF5_STATUS_CHECK(status);

            status = H5Sclose(dataspace_id);
            HDF5_STATUS_CHECK(status);

            dataspace_id = H5Screate(H5S_SCALAR);

            attribute_id = H5Acreate (dataset_id, "K", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Awrite (attribute_id, H5T_NATIVE_INT, &K );
            HDF5_STATUS_CHECK(status);
            status = H5Aclose(attribute_id);
            HDF5_STATUS_CHECK(status);

            attribute_id = H5Acreate (dataset_id, "S", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Awrite (attribute_id, H5T_NATIVE_INT, &S );
            HDF5_STATUS_CHECK(status);
            status = H5Aclose(attribute_id);
            HDF5_STATUS_CHECK(status);

            status = H5Dclose(dataset_id);
            HDF5_STATUS_CHECK(status);
        }

        status = H5Gclose(U_id);
        HDF5_STATUS_CHECK(status);

        status = H5Fclose(file_id);
        HDF5_STATUS_CHECK(status);

        Ucur += step;
    }
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
