#include "MatrixMultiply.hpp"

#include <exception>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <numeric>

scottgs::MatrixMultiply::MatrixMultiply() 
{
	;
}

scottgs::MatrixMultiply::~MatrixMultiply()
{
	;
}


scottgs::FloatMatrix scottgs::MatrixMultiply::operator()(const scottgs::FloatMatrix& lhs, const scottgs::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	scottgs::FloatMatrix result(lhs.size1(),rhs.size2());


	// YOUR ALGORIHM WITH COMMENTS GOES HERE:
	float sum;
	int	res_rows = result.size1(),
		res_cols = result.size2(),
		inner= lhs.size2(),
		lhs_cols = lhs.size2();

	// Transpose rhs matrix
	// scottgs::FloatMatrix rhsT = boost::numeric::ublas::trans(rhs);

	scottgs::FloatMatrix rhsT(rhs.size2(), rhs.size1());
	int rhsT_rows = rhsT.size1(), rhsT_cols = rhsT.size2();

	for(int i = 0 ; i< rhsT_rows ; i++){
		for(int j = 0 ; j< rhsT_cols ; j++){
			rhsT(i, j) = rhs(j, i);
		}
	}

	float *resultPtr = &result(0,0);
	const float *lhsPtr = &lhs(0,0);
	const float *rhsTPtr = &rhsT(0,0);

	// for every row in result
	for(int i = 0 ; i < res_rows ; i++){
		// Go through every column in result
		for(int j = 0 ; j < res_cols ; j++){
			// Multiply lhs row with rhs column
			sum = 0;
			for(int k = 0 ; k < inner ; k++){
				sum += lhsPtr[(i*lhs_cols) + k] * rhsTPtr[(j* rhsT_cols) + k];
			}
			resultPtr[(i*res_cols) + j] = sum;
		}
	}

	return result;
}

scottgs::FloatMatrix scottgs::MatrixMultiply::multiply(const scottgs::FloatMatrix& lhs, const scottgs::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	return boost::numeric::ublas::prod(lhs,rhs);
}

