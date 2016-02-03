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
	int sum,
		res_rows = result.size1(),
		res_cols = result.size2(),
		inner= rhs.size1();

	// for every row in result
	for(int i = 0 ; i < res_rows ; i++){
		// Go through every column in result
		for(int j = 0 ; j < res_cols ; j++){
			// Multiply lhs row with rhs column
			sum = 0;
			// result(i, j) = 0;
			for(int k = 0 ; k < inner ; k++){
				sum += lhs(i, k) * rhs(k, j);
				// result(i, j) += lhs(i, k) * rhs(k, j);
			}
			result(i, j) = sum;
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

