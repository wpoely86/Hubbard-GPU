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
#include <sstream>
#include <assert.h>
#include <hdf5.h>
#include "helpers.h"

// macro to help check return status of HDF5 functions
#define HDF5_STATUS_CHECK(status) {                 \
    if(status < 0)                                  \
    std::cerr << __FILE__ << ":" << __LINE__ <<     \
    ": Problem with writing to file. Status code="  \
    << status << std::endl;                         \
}

/**
 * @param n_ number of rows
 * @param m_ number of columns
 */
matrix::matrix(int n_, int m_)
{
    assert(n_ && m_);
    this->n = n_;
    this->m = m_;
    mat.reset(new double [n*m]);
}

matrix::matrix(const matrix &orig)
{
    n = orig.n;
    m = orig.m;
    mat.reset(new double [n*m]);
    std::memcpy(mat.get(), orig.getpointer(), n*m*sizeof(double));
}

matrix::matrix(matrix &&orig)
{
    n = orig.n;
    m = orig.m;
    mat = std::move(orig.mat);
}

matrix& matrix::operator=(const matrix &orig)
{
    n = orig.n;
    m = orig.m;
    mat.reset(new double [n*m]);
    std::memcpy(mat.get(), orig.getpointer(), n*m*sizeof(double));
    return *this;
}

matrix& matrix::operator=(double val)
{
    for(int i=0;i<n*m;i++)
        mat[i] = val;

    return *this;
}

int matrix::getn() const
{
    return n;
}

int matrix::getm() const
{
    return m;
}

double matrix::operator()(int x,int y) const
{
    assert(x<n && y<m);
    return mat[x+y*n];
}

double& matrix::operator()(int x,int y)
{
    assert(x<n && y<m);
    return mat[x+y*n];
}

double& matrix::operator[](int x)
{
    assert(x<n*m);
    return mat[x];
}

double matrix::operator[](int x) const
{
    assert(x<n*m);
    return mat[x];
}

double* matrix::getpointer() const
{
    return mat.get();
}

matrix& matrix::prod(matrix const &A, matrix const &B)
{
    char trans = 'N';

    double alpha = 1.0;
    double beta = 0.0;

    assert(A.n == n && B.m == m);

    dgemm_(&trans,&trans,&A.n,&B.m,&A.m,&alpha,A.mat.get(),&A.n,B.mat.get(),&B.n,&beta,mat.get(),&A.n);

    return *this;
}

std::unique_ptr<double []> matrix::svd()
{
    char jobu = 'N';
    char jobvt = 'O';

    int count_sing = std::min(n,m);

    std::unique_ptr<double []> sing_vals(new double[count_sing]);

    int lwork = 5*count_sing; // MAX(1,3*MIN(M,N)+MAX(M,N),5*MIN(M,N)).
    std::unique_ptr<double []> work(new double[lwork]);

    int info;

    dgesvd_(&jobu,&jobvt,&n,&m,mat.get(),&n,sing_vals.get(),0,&n,0,&n,work.get(),&lwork,&info);

    if(info)
        std::cerr << "svd failed. info = " << info << std::endl;

    return sing_vals;
}

void matrix::Print() const
{
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            std::cout << i << " " << j << "\t" << (*this)(i,j) << std::endl;
}


KBlock::KBlock(int K, int L, int Nu, int Nd)
{
    this->K = K;
    this->L = L;
    this->Nu = Nu;
    this->Nd = Nd;
}

KBlock::KBlock(const KBlock &orig)
{
    K = orig.K;
    L = orig.L;
    Nu = orig.Nu;
    Nd = orig.Nd;

    basis = orig.basis;
}

KBlock::KBlock(KBlock &&orig)
{
    K = orig.K;
    L = orig.L;
    Nu = orig.Nu;
    Nd = orig.Nd;

    basis = std::move(orig.basis);
}

myint KBlock::getUp(int index) const
{
    assert(index<getdim());
    return basis[index].first;
}

myint KBlock::getDown(int index) const
{
    assert(index<getdim());
    return basis[index].second;
}

int KBlock::getdim() const
{
    return basis.size();
}

int KBlock::getK() const
{
    return K;
}

int KBlock::getL() const
{
    return L;
}

const std::pair<myint,myint>& KBlock::operator[](int index) const
{
    assert(index < basis.size());
    return basis[index];
}

void KBlock::Print() const
{
    int Hbc = BareHamiltonian::CountBits((1<<L)-1);

    for(int i=0;i<basis.size();i++)
        std::cout << BareHamiltonian::print_bin(basis[i].first, Hbc) << "\t" << BareHamiltonian::print_bin(basis[i].first, Hbc) << std::endl;
}


MomBasis::MomBasis(int L, int Nu, int Nd)
{
    this->L = L;
    this->Nu = Nu;
    this->Nd = Nd;

    int dim_up = BareHamiltonian::CalcDim(L, Nu);
    int dim_down = BareHamiltonian::CalcDim(L, Nd);
    dim = dim_up * dim_down;

    BuildBase();
}

myint MomBasis::getUp(int K, int index) const
{
    assert(K<L);
    assert(index<basisblocks[K]->getdim());
    return (*basisblocks[K])[index].first;
}

myint MomBasis::getDown(int K, int index) const
{
    assert(K<L);
    assert(index<basisblocks[K]->getdim());
    return (*basisblocks[K])[index].second;
}

int MomBasis::findUp(int K, myint ket) const
{
    assert(K<L);
    for(unsigned int i=0;i<basisblocks[K]->getdim();i++)
        if((*basisblocks[K])[i].first == ket)
            return i;

    assert(0 && "Should never be reached");
    return -1;
}

int MomBasis::findDown(int K, myint ket) const
{
    assert(K<L);
    for(unsigned int i=0;i<basisblocks[K]->getdim();i++)
        if((*basisblocks[K])[i].second == ket)
            return i;

    assert(0 && "Should never be reached");
    return -1;
}

int MomBasis::getdim() const
{
    return dim;
}

int MomBasis::getdimK(int K) const
{
    assert(K<L);
    return basisblocks[K]->getdim();
}

int MomBasis::getL() const
{
    return L;
}

int MomBasis::getNu() const
{
    return Nu;
}

int MomBasis::getNd() const
{
    return Nd;
}

void MomBasis::Print() const
{
    for(int K=0;K<L;K++)
    {
        std::cout << "K = " << K << " (" << basisblocks[K]->getdim() << ")" << std::endl;

        basisblocks[K]->Print();
    }
}

void MomBasis::getvec(int K, int i, myint &upket, myint &downket) const
{
    assert(K<L);
    upket = (*basisblocks[K])[i].first;
    downket = (*basisblocks[K])[i].second;
}

const std::pair<myint, myint>& MomBasis::operator()(int K, int index) const
{
    assert(K<L);
    return (*basisblocks[K])[index];
}

std::shared_ptr<class KBlock> MomBasis::getBlock(int K) const
{
    assert(K<L);
    return basisblocks[K];
}

void MomBasis::BuildBase()
{
    std::vector<myint> baseUp;
    std::vector<myint> baseDown;

    int dim_up = BareHamiltonian::CalcDim(L, Nu);
    int dim_down = BareHamiltonian::CalcDim(L, Nd);

    baseUp.reserve(dim_up);
    baseDown.reserve(dim_down);

    myint Hb = 1 << L;

    for(myint i=0;i<Hb;i++)
    {
        if(BareHamiltonian::CountBits(i) == Nd)
            baseDown.push_back(i);

        if(BareHamiltonian::CountBits(i) == Nu)
            baseUp.push_back(i);
    }

    std::vector< std::tuple<myint,myint,int> > totalmom;
    totalmom.reserve(dim);

    // count momentum of earch state
    for(unsigned int a=0;a<baseUp.size();a++)
        for(unsigned int b=0;b<baseDown.size();b++)
        {
            auto calcK = [] (myint cur) -> int
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

            K %= L;

            totalmom.push_back(std::make_tuple(baseUp[a],baseDown[b],K));
        }

    // not needed anymore, free the memory
    baseUp.clear();
    baseDown.clear();

    // sort to K
    std::sort(totalmom.begin(), totalmom.end(),
            [](const std::tuple<myint,myint,int> & a, const std::tuple<myint,myint,int> & b) -> bool
            {
                return std::get<2>(a) < std::get<2>(b);
            });

    basisblocks.reserve(L);
    for(int K=0;K<L;K++)
        basisblocks.push_back(std::shared_ptr<class KBlock>(new KBlock(K,L,Nu,Nd)));

    std::for_each(totalmom.begin(), totalmom.end(), [this](std::tuple<myint,myint,int> elem)
            {
                auto tmp = std::make_pair(std::get<0>(elem), std::get<1>(elem));
                basisblocks[std::get<2>(elem)]->basis.push_back(tmp);
            });
}

/**
 * @param K which K block
 * @param orig the basis from which to select a K block
 * @param dim the dimension of the subbasis
 */
SubBasis::SubBasis(int K, MomBasis &orig, int dim)
{
    this->L= orig.getL();
    this->Nu = orig.getNu();
    this->Nd = orig.getNd();

    int space_dim = orig.getdimK(K);

    coeffs.reset(new matrix(space_dim,dim));

//    s_coeffs.reset(new SparseMatrix_CCS(space_dim,dim));

    assert(dim <= space_dim);

    (*coeffs) = 0;
    for(int i=0;i<dim;i++)
        (*coeffs)(i,i) = 1;

//    s_coeffs->NewCol();
//    for(int i=0;i<dim;i++)
//    {
//        s_coeffs->PushToCol(i,1);
//        s_coeffs->NewCol();
//    }
//    s_coeffs->NewCol();

    basis = std::move(orig.getBlock(K));
}

SubBasis::SubBasis(const SubBasis &orig)
{
    L = orig.L;
    Nu = orig.Nu;
    Nd = orig.Nd;

    if(orig.coeffs)
        coeffs.reset(new matrix(*orig.coeffs.get()));

//    if(orig.s_coeffs)
//        s_coeffs.reset(new SparseMatrix_CCS(*orig.s_coeffs.get()));

    basis = orig.basis;
}

SubBasis::SubBasis(SubBasis &&orig)
{
    L = orig.L;
    Nu = orig.Nu;
    Nd = orig.Nd;

    coeffs = std::move(orig.coeffs);
    basis = std::move(orig.basis);
}

SubBasis& SubBasis::operator=(const SubBasis &orig)
{
    L = orig.L;
    Nu = orig.Nu;
    Nd = orig.Nd;

    if(orig.coeffs)
        coeffs.reset(new matrix(*orig.coeffs.get()));

//    if(orig.s_coeffs)
//        s_coeffs.reset(new SparseMatrix_CCS(*orig.s_coeffs.get()));

    basis = orig.basis;

    return *this;
}

SubBasis& SubBasis::operator=(SubBasis &&orig)
{
    L = orig.L;
    Nu = orig.Nu;
    Nd = orig.Nd;

//    if(orig.s_coeffs)
//        s_coeffs.reset(new SparseMatrix_CCS(*orig.s_coeffs.get()));

    coeffs = std::move(orig.coeffs);
    basis = std::move(orig.basis);

    return *this;
}

void SubBasis::Print() const
{
    int Hbc = BareHamiltonian::CountBits((1<<L)-1);

    std::cout << "dim: " << getdim() << " in " << getspacedim() << std::endl;
//    for(int s=0;s<getdim();s++)
//    {
//        for(int i=0;i<s_coeffs->NumOfElInCol(s);i++)
//            std::cout << s_coeffs->GetElementRowIndexInCol(s, i) << "\t" << \
//                BareHamiltonian::print_bin(basis->getUp(s_coeffs->GetElementRowIndexInCol(s,i)), Hbc) << " " << \
//                BareHamiltonian::print_bin(basis->getDown(s_coeffs->GetElementRowIndexInCol(s,i)), Hbc) << std::endl;
//
//        std::cout << std::endl;
//    }

    for(int s=0;s<getdim();s++)
    {
        for(int i=0;i<getspacedim();i++)
            std::cout << (*coeffs)(i,s) << "\t" << BareHamiltonian::print_bin(basis->getUp(i), Hbc) << " " << BareHamiltonian::print_bin(basis->getDown(i), Hbc) << std::endl;
        std::cout << std::endl;
    }
}

int SubBasis::getdim() const
{
    return coeffs->getm();
}

int SubBasis::getspacedim() const
{
    return coeffs->getn();
}

int SubBasis::getindex(myint upket, myint downket) const
{
    for(int i=0;i<basis->getdim();i++)
        if(upket == basis->getUp(i) && downket == basis->getDown(i))
            return i;

    assert(0 && "Impossible index for subbasis");
    return -1;
}

void SubBasis::Slad_min(SubBasis &orig)
{
//    std::cout << "Doing slamin: K=" << orig.basis->getK() << " start Sz=" << (orig.Nu-orig.Nd)/2 << " final Sz=" << (Nu-Nd)/2 << std::endl;
    matrix transform(getspacedim(),orig.getspacedim());
    transform = 0;

//    SparseMatrix_CCS transform(getspacedim(),orig.getspacedim());
//
//    transform.NewCol();

    for(int i=0;i<orig.getspacedim();i++)
    {
        myint upket = orig.basis->getUp(i);
        myint downket = orig.basis->getDown(i);
        myint cur = upket;

        while(cur)
        {
            myint ksp = cur & (~cur + 1);
            cur ^= ksp;

            // if spin down state is occupied, skip
            if(downket & ksp)
                continue;

            // will give trouble with 16 sites?
            myint fullstate = (upket << L) + downket;

            int sign = BareHamiltonian::CountBits( ( ((ksp<<L)-1) ^ (ksp-1) ) & fullstate );

            if( sign & 0x1)
                sign = -1;
            else
                sign = 1;

            int index = getindex(upket ^ ksp, downket | ksp);

            transform(index,i) = sign;
//            transform.PushToCol(index, sign);
        }
//        transform.NewCol();
    }
//    transform.NewCol();

    coeffs->prod(transform, *orig.coeffs);
//    orig.ToSparseMatrix();
//    s_coeffs->prod(transform, *orig.s_coeffs);
//
//    std::cout << "Mats: " << std::endl;
//    coeffs->Print();
//    std::cout << std::endl;
//    std::cout << *s_coeffs << std::endl;

//    s_coeffs->ConvertToMatrix(*coeffs);

    for(int i=0;i<getdim();i++)
    {
        double norm;
        int spacedim = getspacedim();
        int inc = 1;

        norm = ddot_(&spacedim,&(*coeffs)[i*spacedim],&inc,&(*coeffs)[i*spacedim],&inc);
        norm = 1.0/sqrt(norm);

        dscal_(&spacedim,&norm,&(*coeffs)[i*spacedim],&inc);
    }
}

void SubBasis::Get(int index, myint &upket, myint &downket) const
{
    assert(index < basis->getdim());

    upket = basis->getUp(index);
    downket = basis->getDown(index);
}

std::pair<myint,myint> SubBasis::Get(int index) const
{
    return (*basis)[index];
}

double SubBasis::GetCoeff(int i, int j) const
{
    return (*coeffs)(i,j);
}

void SubBasis::SetCoeff(int i, int j, double value)
{
    (*coeffs)(i,j) = value;
}

void SubBasis::Normalize()
{
    for(int i=0;i<getdim();i++)
    {
        double norm;
        int spacedim = getspacedim();
        int inc = 1;

        norm = ddot_(&spacedim,&(*coeffs)[i*spacedim],&inc,&(*coeffs)[i*spacedim],&inc);
        std::cout << "Norm is " << norm << std::endl;
        norm = 1.0/sqrt(norm);

        dscal_(&spacedim,&norm,&(*coeffs)[i*spacedim],&inc);
    }
}

void SubBasis::ToSparseMatrix()
{
    s_coeffs.reset(new SparseMatrix_CCS(coeffs->getn(),coeffs->getm()));

    s_coeffs->ConvertFromMatrix(*coeffs);
}

const SparseMatrix_CCS& SubBasis::getSparse() const
{
    assert(s_coeffs);
    return (*s_coeffs);
}

BasisList::BasisList(int L, int Nu, int Nd)
{
    this->L= L;
    this->Nu = Nu;
    this->Nd = Nd;

    Smax = (Nu+Nd)/2;

    totS = ((Smax+1)*(Smax+2))/2;

    int n = L*totS;

    ind_list.reset(new double [n]);

    list.resize(n);

    for(int i=0;i<n;i++)
        ind_list[i] = EMPTY;
}

bool BasisList::Exists(int K, int S, int Sz) const
{
    if( K >= L || S > Smax || Sz > S)
        return false;

    if( ind_list[K*totS + (S*(S+1))/2 + Sz] != EMPTY )
        return true;
    else
        return false;
}

SubBasis& BasisList::Get(int K, int S, int Sz)
{
    assert(Exists(K, S, Sz) && "Get on nonexisting block");

    return list[K*totS + (S*(S+1))/2 + Sz];
}

void BasisList::Create(int K, int S, int Sz, MomBasis &orig, int dim)
{
    list[K*totS + (S*(S+1))/2 + Sz] = SubBasis(K, orig, dim);
    ind_list[K*totS + (S*(S+1))/2 + Sz] = 1;
}

void BasisList::Print() const
{
    for(int S=0;S<=Smax;S++)
        for(int Sz=0;Sz<=S;Sz++)
            for(int K=0;K<L;K++)
                if(Exists(K,S,Sz))
                {
                    std::cout << "Block: S=" << S << " Sz=" << Sz << " K=" << K << std::endl;
                    list[K*totS + (S*(S+1))/2 + Sz].Print();
                }
}

void BasisList::DoProjection(int K, int S, int Sz, MomBasis const &orig)
{
    int dimK = orig.getdimK(K);

    std::unique_ptr<matrix> proj_matrix(new matrix(dimK,dimK));

    for(int i=0;i<dimK;i++)
        for(int j=0;j<dimK;j++)
        {
            (*proj_matrix)(i,j) = 0;

            // run over al higher S that need projecting
            for(int pS=S+1;pS<=Smax;pS++)
            {
                if(!Exists(K,pS,Sz))
                    continue;

                auto& cur_basis = Get(K,pS,Sz);

                for(int k=0;k<cur_basis.getdim();k++)
                    (*proj_matrix)(i,j) += cur_basis.GetCoeff(i,k) * cur_basis.GetCoeff(j,k);
            }
        }


    auto sing_vals = proj_matrix->svd();

    int sing_vals_start = 0;
    while( fabs(sing_vals[sing_vals_start]) > 1e-10 )
        sing_vals_start++;

    auto &finalbasis = Get(K,S,Sz);

    assert(finalbasis.getdim() == (dimK-sing_vals_start));

    for(int i=sing_vals_start;i<dimK;i++)
    {
        int start = i-sing_vals_start;

        for(int j=0;j<orig.getdimK(K);j++)
            finalbasis.SetCoeff(j,start,(*proj_matrix)(i,j));
    }
}

/**
 * delete all subbasis for Sz and higher
 */
void BasisList::Clean(int Sz)
{
    for(int S=Smax;S>=Sz;S--)
        for(int cur_Sz=Smax;cur_Sz>=Sz;cur_Sz--)
            for(int K=0;K<L;K++)
                if(Exists(K,S,cur_Sz))
                {
                    std::cout << "Deleting: S=" << S << " Sz=" << cur_Sz << " K=" << K << std::endl;
                    list[K*totS + (S*(S+1))/2 + Sz] = SubBasis();
                    ind_list[K*totS + (S*(S+1))/2 + cur_Sz] = EMPTY;
                }
}

/**
 * Delete everything in this BasisList
 */
void BasisList::MakeEmpty()
{
    int n = L*totS;

    for(int i=0;i<n;i++)
        ind_list[i] = EMPTY;

    list.clear();
}

/**
 * Store the spin basis
 * @param L the length of the system
 * @param Nu number of up electrons
 * @param Nd number of down electrons
 * @param orig the BasisList from which to extract the basis. The BasisList
 * will be empted after this
 */
SpinBasis::SpinBasis(int L,int Nu,int Nd,BasisList &orig)
{
    this->L = L;
    this->Nu = Nu;
    this->Nd = Nd;

    int Sz=(Nu-Nd) >= 0 ? (Nu-Nd)/2 : (Nd-Nu)/2;
    int Smax = (Nu+Nd)/2;

    // upper boundary: all S for every K block
    basis.reserve(L*Smax);
    ind.reserve(L*Smax);

    for(int K=0;K<L;K++)
        for(int S=0;S<=Smax;S++)
            if(orig.Exists(K,S,Sz))
            {
                basis.push_back(std::move(orig.Get(K,S,Sz)));
                ind.push_back(std::make_pair(K,S));
            }

    std::for_each(basis.begin(), basis.end(), [](SubBasis& x) { x.ToSparseMatrix(); });

//    std::for_each(basis.begin(), basis.end(), [](SubBasis& x) { x.Print(); x.getSparse().PrintRaw(); });

    orig.MakeEmpty();
}

SpinBasis::SpinBasis(const char *filename)
{
    ReadBasis(filename);
}

int SpinBasis::getnumblocks() const
{
    return basis.size();
}

void SpinBasis::SaveBasis(const char *filename) const
{
    hid_t       file_id, group_id, dataset_id, dataspace_id, attribute_id, matspace_id;
    herr_t      status;

    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    HDF5_STATUS_CHECK(file_id);

    group_id = H5Gcreate(file_id, "/SpinBasis", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
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

    int num = basis.size();
    attribute_id = H5Acreate (group_id, "num", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite (attribute_id, H5T_NATIVE_INT, &num );
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    for(int i=0;i<basis.size();i++)
    {
        hsize_t dimarr = basis[i].coeffs->getn()*basis[i].coeffs->getm();

        matspace_id = H5Screate_simple(1, &dimarr, NULL);

        std::stringstream name;

        name << i;

        dataset_id = H5Dcreate(group_id, name.str().c_str(), H5T_IEEE_F64LE, matspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        assert(basis[i].coeffs);
        status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, basis[i].coeffs->getpointer());
        HDF5_STATUS_CHECK(status);

        int K = ind[i].first;
        int S = ind[i].second;

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

        int n = basis[i].coeffs->getn();
        attribute_id = H5Acreate (dataset_id, "n", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Awrite (attribute_id, H5T_NATIVE_INT, &n);
        HDF5_STATUS_CHECK(status);
        status = H5Aclose(attribute_id);
        HDF5_STATUS_CHECK(status);

        int m = basis[i].coeffs->getm();
        attribute_id = H5Acreate (dataset_id, "m", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Awrite (attribute_id, H5T_NATIVE_INT, &m);
        HDF5_STATUS_CHECK(status);
        status = H5Aclose(attribute_id);
        HDF5_STATUS_CHECK(status);

        status = H5Sclose(matspace_id);
        HDF5_STATUS_CHECK(status);

        status = H5Dclose(dataset_id);
        HDF5_STATUS_CHECK(status);
    }

    status = H5Sclose(dataspace_id);
    HDF5_STATUS_CHECK(status);

    status = H5Gclose(group_id);
    HDF5_STATUS_CHECK(status);

    status = H5Fclose(file_id);
    HDF5_STATUS_CHECK(status);
}

void SpinBasis::ReadBasis(const char *filename)
{
    hid_t file_id, group_id, attribute_id, dataset_id;
    herr_t status;

    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    HDF5_STATUS_CHECK(file_id);

    group_id = H5Gopen(file_id, "/SpinBasis", H5P_DEFAULT);
    HDF5_STATUS_CHECK(group_id);

    attribute_id = H5Aopen(group_id, "L", H5P_DEFAULT);
    HDF5_STATUS_CHECK(attribute_id);
    status = H5Aread(attribute_id, H5T_NATIVE_INT, &L);
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Aopen(group_id, "Nu", H5P_DEFAULT);
    HDF5_STATUS_CHECK(attribute_id);
    status = H5Aread(attribute_id, H5T_NATIVE_INT, &Nu);
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    attribute_id = H5Aopen(group_id, "Nd", H5P_DEFAULT);
    HDF5_STATUS_CHECK(attribute_id);
    status = H5Aread(attribute_id, H5T_NATIVE_INT, &Nd);
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    MomBasis pbase(L, Nu, Nd);

    int num;
    attribute_id = H5Aopen(group_id, "num", H5P_DEFAULT);
    HDF5_STATUS_CHECK(attribute_id);
    status = H5Aread(attribute_id, H5T_NATIVE_INT, &num);
    HDF5_STATUS_CHECK(status);
    status = H5Aclose(attribute_id);
    HDF5_STATUS_CHECK(status);

    basis.resize(num);
    ind.reserve(num);

    for(int i=0;i<num;i++)
    {
        basis[i].L = L;
        basis[i].Nu = Nu;
        basis[i].Nd = Nd;

        std::stringstream name;
        name << i;

        int S, K;

        dataset_id = H5Dopen(group_id, name.str().c_str(), H5P_DEFAULT);
        HDF5_STATUS_CHECK(dataset_id);

        attribute_id = H5Aopen(dataset_id, "K", H5P_DEFAULT);
        HDF5_STATUS_CHECK(attribute_id);
        status = H5Aread(attribute_id, H5T_NATIVE_INT, &K);
        HDF5_STATUS_CHECK(status);
        status = H5Aclose(attribute_id);
        HDF5_STATUS_CHECK(status);

        attribute_id = H5Aopen(dataset_id, "S", H5P_DEFAULT);
        HDF5_STATUS_CHECK(attribute_id);
        status = H5Aread(attribute_id, H5T_NATIVE_INT, &S);
        HDF5_STATUS_CHECK(status);
        status = H5Aclose(attribute_id);
        HDF5_STATUS_CHECK(status);

        ind.push_back(std::make_pair(K,S));

        basis[i].basis = pbase.getBlock(K);

        int n,m;
        attribute_id = H5Aopen(dataset_id, "n", H5P_DEFAULT);
        HDF5_STATUS_CHECK(attribute_id);
        status = H5Aread(attribute_id, H5T_NATIVE_INT, &n);
        HDF5_STATUS_CHECK(status);
        status = H5Aclose(attribute_id);
        HDF5_STATUS_CHECK(status);

        attribute_id = H5Aopen(dataset_id, "m", H5P_DEFAULT);
        HDF5_STATUS_CHECK(attribute_id);
        status = H5Aread(attribute_id, H5T_NATIVE_INT, &m);
        HDF5_STATUS_CHECK(status);
        status = H5Aclose(attribute_id);
        HDF5_STATUS_CHECK(status);

        basis[i].coeffs.reset(new matrix(n,m));
        (*basis[i].coeffs) = -2;

        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, basis[i].coeffs->getpointer() );
        HDF5_STATUS_CHECK(status);

        status = H5Dclose(dataset_id);
        HDF5_STATUS_CHECK(status);
    }

    status = H5Gclose(group_id);
    HDF5_STATUS_CHECK(status);

    status = H5Fclose(file_id);
    HDF5_STATUS_CHECK(status);

    std::for_each(basis.begin(), basis.end(), [](SubBasis& x) { x.ToSparseMatrix(); });
}


std::pair<int,int> SpinBasis::getSK(int index) const
{
    return ind[index];
}

const SubBasis& SpinBasis::getBlock(int index) const
{
    return basis[index];
}

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
