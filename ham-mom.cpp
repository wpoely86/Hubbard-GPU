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
#include <tuple>
#include <iomanip>
#include <hdf5.h>
#include "ham-mom.h"

// macro to help check return status of HDF5 functions
#define HDF5_STATUS_CHECK(status) {                 \
    if(status < 0)                                  \
    std::cerr << __FILE__ << ":" << __LINE__ <<     \
    ": Problem with writing to file. Status code="  \
    << status << std::endl;                         \
}

/**
 * Constructor of the MomHamiltonian class
 * @param L Number of lattice sites
 * @param Nu Number of Up Electrons
 * @param Nd Number of Down Electrons
 * @param J The hopping strengh
 * @param U The onsite interaction strength
 */
MomHamiltonian::MomHamiltonian(int L, int Nu, int Nd, double J, double U): BareHamiltonian(L,Nu,Nd,J,U)
{
}

/**
 * Destructor of the MomHamiltonian class
 */
MomHamiltonian::~MomHamiltonian()
{
}

/**
 * Builds all the up and down base kets in the momentum base
 * Sort them so we have a block diagonal matrix according to total momentum
 */
void MomHamiltonian::BuildBase()
{
    baseUp.reserve(CalcDim(L,Nu));
    baseDown.reserve(CalcDim(L,Nd));

    for(myint i=0;i<Hb;i++)
    {
	if(CountBits(i) == Nd)
	    baseDown.push_back(i);

	if(CountBits(i) == Nu)
	    baseUp.push_back(i);
    }

    std::vector< std::tuple<int,int,int> > totalmom;
    totalmom.reserve(dim);

    // count momentum of earch state
    for(unsigned int a=0;a<baseUp.size();a++)
        for(unsigned int b=0;b<baseDown.size();b++)
        {
            auto calcK = [] (myint cur)
            {
                int tot = 0;

                while(cur)
                {
                    // select rightmost up state in the ket
                    myint ksp = cur & (~cur + 1);
                    // set it to zero
                    cur ^= ksp;

                    tot += BareHamiltonian::CountBits(ksp-1);
                }

                return tot;
            };

            int K = calcK(baseUp[a]) + calcK(baseDown[b]);

            K = K % L;

            totalmom.push_back(std::make_tuple(a,b,K));
        }

    std::sort(totalmom.begin(), totalmom.end(),
            [](const std::tuple<int,int,int> & a, const std::tuple<int,int,int> & b) -> bool
            {
            return std::get<2>(a) < std::get<2>(b);
            });

    // a block for each momenta
    mombase.resize(L);

    std::for_each(totalmom.begin(), totalmom.end(), [this](std::tuple<int,int,int> elem)
            {
            auto tmp = std::make_pair(std::get<0>(elem), std::get<1>(elem));
            mombase[std::get<2>(elem)].push_back(tmp);
            } );

//    for(unsigned int i=0;i<mombase.size();i++)
//    {
//        std::cout << "K = " << i << " (" << mombase[i].size() << ")" << std::endl;
//        for(unsigned int j=0;j<mombase[i].size();j++)
//            std::cout << j << "\t" << mombase[i][j].first << "\t" << mombase[i][j].second << "\t" << print_bin(baseUp[mombase[i][j].first]) << "\t" << print_bin(baseDown[mombase[i][j].second]) << std::endl;
//    }
}

/**
  * Builds the full MomHamiltonian matrix
  */
void MomHamiltonian::BuildFullHam()
{
    if( !baseUp.size() || !baseDown.size() )
    {
        std::cerr << "Build base before building MomHamiltonian" << std::endl;
        return;
    }

    blockmat.resize(L);

    for(int B=0;B<L;B++)
    {
        int cur_dim = mombase[B].size();

        blockmat[B].reset(new double [cur_dim*cur_dim]);

        for(int i=0;i<cur_dim;i++)
        {
            int a = mombase[B][i].first;
            int b = mombase[B][i].second;

            for(int j=i;j<cur_dim;j++)
            {
                int c = mombase[B][j].first;
                int d = mombase[B][j].second;

                blockmat[B][j+cur_dim*i] = U/L*interaction(a,b,c,d);

                blockmat[B][i+cur_dim*j] = blockmat[B][j+cur_dim*i];

            }

            blockmat[B][i+cur_dim*i] += J * (hopping(baseUp[a]) + hopping(baseDown[b]));
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
int MomHamiltonian::interaction(int a, int b, int c, int d) const
{
    int result = 0;

    myint downket = baseDown[d];
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

        signk2 *= CalcSign(k2,Hbc,baseDown[d]);

        upket = baseUp[c];

        while(upket)
        {
            // select rightmost up state in the ket
            myint k1sp = upket & (~upket + 1);
            // set it to zero
            upket ^= k1sp;

            int k1 = CountBits(k1sp-1);

            int signk1 = CalcSign(k1,Hbc,baseUp[c]);

            myint K = (k1+k2) % L;

            upbra = baseUp[a];

            while(upbra)
            {
                // select rightmost up state in the ket
                myint k3sp = upbra & (~upbra + 1);
                // set it to zero
                upbra ^= k3sp;

                // k3: spin up in bra
                int k3 = CountBits(k3sp-1);

                if(! ((baseUp[c] ^ k1sp) == (baseUp[a] ^ k3sp)) )
                    continue;

                // k4: spin down in bra
                int k4 = (K-k3+L)%L;

                if(! ((1<<k4 & baseDown[b]) && ((baseDown[d] ^ k2sp) == (baseDown[b] ^ 1<<k4)) ))
                    continue;

                int signk3 = CalcSign(k3,Hbc,baseUp[a]);

                int signk4 = 1;

                // when uneven, extra minus sign
                if(Nu & 0x1)
                    signk4 *= -1;

                signk4 *= CalcSign(k4,Hbc,baseDown[b]);

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
double MomHamiltonian::hopping(myint ket) const
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
 * Matrix vector product with MomHamiltonian: y = ham*x + alpha*y
 * @param x the input vector
 * @param y the output vector
 * @param alpha the scaling value
 */
void MomHamiltonian::mvprod(double *x, double *y, double alpha) const
{
    double beta = 1;
    int incx = 1;
    char uplo = 'U';

    int offset = 0;

    for(int B=0;B<L;B++)
    {
        int dim = mombase[B].size();

        dsymv_(&uplo,&dim,&beta,blockmat[B].get(),&dim,&x[offset],&incx,&alpha,&y[offset],&incx);

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
std::vector<double> MomHamiltonian::ExactDiagonalizeFull(bool calc_eigenvectors)
{
    std::vector<double> eigenvalues(dim);

    int offset = 0;

    for(int B=0;B<L;B++)
    {
        int dim = mombase[B].size();

        Diagonalize(dim, blockmat[B].get(), &eigenvalues[offset], calc_eigenvectors);

        offset += dim;
    }

    std::sort(eigenvalues.begin(), eigenvalues.end());

    return eigenvalues;
}

/**
 * Diagonalize the block diagonal matrix. Needs lapack. Differs from MomHamiltonian::ExactDiagonalizeFull that
 * is diagonalize block per block and keeps the momentum associated with each eigenvalue.
 * @param calc_eigenvectors set to true to calculate the eigenvectors. The hamiltonian
 * matrix is then overwritten by the vectors.
 * @return a vector of pairs. The first member of the pair is the momentum, the second is
 * the eigenvalue. The vector is sorted to the eigenvalues.
 */
std::vector< std::pair<int,double> > MomHamiltonian::ExactMomDiagonalizeFull(bool calc_eigenvectors)
{
    std::vector<double> eigs(dim);
    std::vector< std::pair<int, double> > energy;

    int offset = 0;

    for(int B=0;B<L;B++)
    {
        int dim = mombase[B].size();

        Diagonalize(dim, blockmat[B].get(), &eigs[offset], calc_eigenvectors);

        for(int i=0;i<dim;i++)
            energy.push_back(std::make_pair(B,eigs[offset+i]));

        offset += dim;
    }

    std::sort(energy.begin(), energy.end(),
            [](const std::pair<int,double> & a, const std::pair<int,double> & b) -> bool
            {
            return a.second < b.second;
            });

    return energy;
}

/**
 * This methods calculates the eigenvalues for a range of U values. The interval is [Ubegin, Uend]
 * with a stepsize of step. The resulting data is written to a file in the HDF5 file format.
 * @param Ubegin the startpoint for U
 * @param Uend the endpoint for U. We demand Ubegin < Uend
 * @param step the stepsize to use for U
 * @param filename the name of the file write to written the eigenvalues to
 */
void MomHamiltonian::GenerateData(double Ubegin, double Uend, double step, std::string filename)
{
    double Ucur = Ubegin;

    if( !baseUp.size() || !baseDown.size() )
        BuildBase();

    std::vector<double> eigenvalues(dim);

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

    std::vector<double> diagonalelements(dim);

    std::vector< std::unique_ptr<double []> > offdiag;
    offdiag.resize(L);

    // make sure that we don't rebuild the whole hamiltonian every time.
    // store the hopping and interaction part seperate so we can just
    // add them in every step
#pragma omp parallel for
    for(int B=0;B<L;B++)
    {
        int cur_dim = mombase[B].size();
        int offset = 0;

        for(int tmp=0;tmp<B;tmp++)
            offset += mombase[tmp].size();

        offdiag[B].reset(new double [cur_dim*cur_dim]);

        for(int i=0;i<cur_dim;i++)
        {
            int a = mombase[B][i].first;
            int b = mombase[B][i].second;

            diagonalelements[offset+i] = hopping(baseUp[a]) + hopping(baseDown[b]);

            for(int j=i;j<cur_dim;j++)
            {
                int c = mombase[B][j].first;
                int d = mombase[B][j].second;

                offdiag[B][j+cur_dim*i] = 1.0/L*interaction(a,b,c,d);
                offdiag[B][i+cur_dim*j] = offdiag[B][j+cur_dim*i];
            }
        }
    }


    while(Ucur <= Uend)
    {
        std::cout << "U = " << Ucur << std::endl;
        setU(Ucur);

        // make hamiltonian
#pragma omp parallel for
        for(int B=0;B<L;B++)
        {
            int cur_dim = mombase[B].size();
            int offset = 0;

            for(int tmp=0;tmp<B;tmp++)
                offset += mombase[tmp].size();

            std::memcpy(blockmat[B].get(),offdiag[B].get(),cur_dim*cur_dim*sizeof(double));

            int tmp = cur_dim*cur_dim;
            int inc = 1;
            dscal_(&tmp,&Ucur,blockmat[B].get(),&inc);

            for(int i=0;i<cur_dim;i++)
                blockmat[B][i+cur_dim*i] += diagonalelements[offset+i];
        }


#pragma omp parallel for
        for(int B=0;B<L;B++)
        {
            int dim = mombase[B].size();
            int offset = 0;

            for(int tmp=0;tmp<B;tmp++)
                offset += mombase[tmp].size();

            Diagonalize(dim, blockmat[B].get(), &eigenvalues[offset], false);
        }

        hid_t U_id;
        std::stringstream name;
        name << std::setprecision(5) << std::fixed << "/run/" << Ucur;

        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        HDF5_STATUS_CHECK(file_id);

        U_id = H5Gcreate(file_id, name.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        HDF5_STATUS_CHECK(U_id);

        for(int B=0;B<L;B++)
        {
            int dim = mombase[B].size();
            int offset = 0;

            for(int tmp=0;tmp<B;tmp++)
                offset += mombase[tmp].size();

            hsize_t dimarr = dim;

            dataspace_id = H5Screate_simple(1, &dimarr, NULL);

            std::stringstream cur_block;
            cur_block << B;
            dataset_id = H5Dcreate(U_id, cur_block.str().c_str(), H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &eigenvalues[offset] );
            HDF5_STATUS_CHECK(status);

            status = H5Sclose(dataspace_id);
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
