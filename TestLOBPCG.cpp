#include <iostream>
#include <Spectra/contrib/LOBPCGSolver.h>

typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> Vector;

typedef std::complex<long double> Complex;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMatrix;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVector;

typedef Eigen::SparseMatrix<long double> SparseMatrix;
typedef Eigen::SparseMatrix<Complex> SparseComplexMatrix;

int example() {
	// generate random sparse A
	Matrix a;
	a = (Matrix::Random(10, 10).array() > 0.6).cast<long double>() * Matrix::Random(10, 10).array() * 5;
	a = Matrix((a).triangularView<Eigen::Lower>()) + Matrix((a).triangularView<Eigen::Lower>()).transpose();
	for (int i = 0; i < 10; i++)
		a(i, i) = i + 0.5;
	std::cout << a << "\n";

	Eigen::SparseMatrix<long double> A(a.sparseView());
	// random X
	Eigen::Matrix<long double, 10, 2> x;
	x = Matrix::Random(10, 2).array();
	Eigen::SparseMatrix<long double> X(x.sparseView());
	// solve Ax = lambda*x
	Spectra::LOBPCGSolver<long double> solver(A, X);
	solver.compute(10, 1e-4); // 10 iterations, L2_tolerance = 1e-4*N
	std::cout << "info\n" << solver.info() << std::endl;
	std::cout << "eigenvalues\n" << solver.eigenvalues() << std::endl;
	std::cout << "eigenvectors\n" << solver.eigenvectors() << std::endl;
	std::cout << "residuals\n" << solver.residuals() << std::endl;
	return 0;
}

int main() {

	SparseMatrix matK(3, 3);         // default is column major
	matK.reserve(9);
	matK.insert(0, 0) = 7.14;                    // alternative: mat.coeffRef(i,j) += v_ij;
	matK.insert(0, 1) = -5.36;
	matK.insert(0, 2) = 1.79;
	matK.insert(1, 0) = -5.36;
	matK.insert(1, 1) = 5.58;
	matK.insert(1, 2) = -2.21;
	matK.insert(2, 0) = 1.79;
	matK.insert(2, 1) = -2.21;
	matK.insert(2, 2) = 0.96;
	matK.makeCompressed();

	std::cout << matK;

}

