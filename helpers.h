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

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <memory>
#include <vector>
#include <cstring>
#include <assert.h>
#include "bare-ham.h"
#include "SparseMatrix_CCS.h"

/**
 * Helper class, wrapper around a double array. Has methods to get the number of rows
 * and columns
 */
class matrix
{
    public:
        matrix(int n_, int m_);

        matrix(const matrix &orig);

        matrix(matrix &&orig);

        virtual ~matrix() { }

        matrix& operator=(const matrix &orig);

        matrix& operator=(double val);

        int getn() const;

        int getm() const;

        double operator()(int x,int y) const;

        double& operator()(int x,int y);

        double& operator[](int x);

        double operator[](int x) const;

        double* getpointer() const;

        matrix& prod(matrix const &A, matrix const &B);

        std::unique_ptr<double []> svd();

        void Print() const;

    private:
        //!n by m array of double
        std::unique_ptr<double []> mat;
        //! number of rows
        int n;
        //! number of columns
        int m;
};

class MomBasis
{
    public:
        MomBasis(int L, int Nu, int Nd);

        virtual ~MomBasis() {  };

        myint getUp(int K, int index) const;

        myint getDown(int K, int index) const;

        int findUp(int K, myint ket) const;

        int findDown(int K, myint ket) const;

        int getdim() const;

        int getdimK(int K) const;

        void Print() const;

        void getvec(int K, int i, myint &upket, myint &downket) const;

        int getL() const;

        int getNu() const;

        int getNd() const;

        const std::pair<myint, myint>& operator()(int K, int index) const;

        std::shared_ptr<class KBlock> getBlock(int K) const;

    private:
        void BuildBase();

        //! number of sites
        int L;
        //! number of up electrons
        int Nu;
        //! number of up electrons
        int Nd;

        //! dimension of the hilbert space
        int dim;

        std::vector< std::shared_ptr<class KBlock> > basisblocks;
};

class KBlock
{
    friend class MomBasis;

    public:
        KBlock(const KBlock &);
        KBlock(KBlock &&);
        virtual ~KBlock() {  };

        myint getUp(int index) const;

        myint getDown(int index) const;

        int getdim() const;

        int getK() const;

        int getL() const;

        const std::pair<myint,myint>& operator[](int index) const;

        void Print() const;

    private:
        KBlock(int K, int L, int Nu, int Nd);

        int L;
        int K;
        int Nu;
        int Nd;

        std::vector< std::pair<myint, myint> > basis;
};

class SubBasis
{
    friend class SpinBasis;

    public:
        SubBasis() {  };
        SubBasis(int K, MomBasis &orig, int dim);
        SubBasis(const SubBasis &orig);
        SubBasis(SubBasis &&orig);
        virtual ~SubBasis() { };

        SubBasis& operator=(const SubBasis &orig);

        SubBasis& operator=(SubBasis &&orig);

        int getdim() const;

        int getspacedim() const;

        void Print() const;

        void Slad_min(SubBasis &orig);

        void Get(int index, myint &upket, myint &downket) const;

        std::pair<myint,myint> Get(int index) const;

        double GetCoeff(int i, int j) const;

        void SetCoeff(int i, int j, double value);

        int getindex(myint upket, myint downket) const;

        void Normalize();

        void ToSparseMatrix();

        const SparseMatrix_CCS& getSparse() const;

        int getSz() const { return (Nu-Nd)/2; }

    private:
        //! number of sites
        int L;
        //! number of up electrons
        int Nu;
        //! number of up electrons
        int Nd;

        std::unique_ptr<class matrix> coeffs;

        std::unique_ptr<class SparseMatrix_CCS> s_coeffs;

        std::shared_ptr<class KBlock> basis;
};

class BasisList
{
    public:
        BasisList(int L, int Nu, int Nd);

        virtual ~BasisList() {  };

        bool Exists(int K, int S, int Sz) const;

        SubBasis& Get(int K, int S, int Sz);

        void Create(int K, int S, int Sz, MomBasis &orig, int dim);

        void Print() const;

        void DoProjection(int K, int S, int Sz, MomBasis const &orig);

        void MakeEmpty();

        void Clean(int Sz);

        static const int EMPTY = -1;

    private:
        //! number of sites
        int L;
        //! number of up electrons
        int Nu;
        //! number of up electrons
        int Nd;

        int totS;

        int Smax;

        std::unique_ptr<double []> ind_list;

        std::vector<SubBasis> list;
};

class SpinBasis
{
    public:
        SpinBasis(int L,int Nu,int Nd,BasisList &);
        SpinBasis(const char *filename);
        virtual ~SpinBasis() {  };

        int getnumblocks() const;

        void SaveBasis(const char *filename) const;

        void ReadBasis(const char *filename);

        std::pair<int,int> getSK(int index) const;

        const SubBasis& getBlock(int index) const;

    private:
        //! number of sites
        int L;
        //! number of up electrons
        int Nu;
        //! number of down electrons
        int Nd;

        std::vector<class SubBasis> basis;
        std::vector< std::pair<int,int> > ind;
};

#endif /* MATRIX_H */

/* vim: set ts=8 sw=4 tw=0 expandtab :*/
