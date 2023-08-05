#include <iostream>

#include "../matrix.hpp"

bool test_operator_eqeq() {
	bool pass = true;
	nn::Mat a(2,3, {1, 2, 3, 4, 5, 6});
	nn::Mat b(2,3, {1, 2, 3, 4, 5, 6});
	if (!(a == b)) pass = false;

	a.reshape(3, 2);
	if (a == b) pass = false;

	a.reshape(2, 3);
	b.at(0, 1) = -1;
	if (a == b) pass = false;

	std::cout << "test_operator== : " << (pass ? "PASS" : "FAIL") << std::endl;
	return pass;

}


bool test_mat_prod() {
	bool pass = true;
	nn::Mat a(1,3, {1, 2, 3});
	nn::Mat b(3,1, {4, 5, 6});
	nn::Mat c(1,1, {32});
	nn::Mat d(3,3, {4, 8, 12, 5, 10, 15, 6, 12, 18});
	if (a*b != c) pass = false;
	if (b*a != d) pass = false;

	std::cout << "test_mat_prod : " << (pass ? "PASS" : "FAIL") << std::endl;
	return pass;
}


void general_tests() {
	std::cout << "General tests : " << std::endl;
	nn::Mat a(2,2);
	a.at(0, 0) = 1;
	a.at(0, 1) = 2;
	a.at(1, 0) = 3;
	a.at(1, 1) = 4;
	a.print(std::cout, "A", 0);

	nn::Mat b(2,2);
	b.at(0, 0) = 5;
	b.at(0, 1) = 6;
	b.at(1, 0) = 7;
	b.at(1, 1) = 8;
	b.print(std::cout, "B", 0);
	
	a.transpose().print(std::cout, "A^t", 0);
	(a+b).print(std::cout, "A+B", 0);
	(a*3).print(std::cout, "A*3", 1);
	(a*b).print(std::cout, "A*B", 2);

	nn::Mat random = nn::Mat::Randn(5, 6);
	random.print(std::cout, "randn", 0);

	std::cout << "End of general tests.\n" << std::endl;
}


void test_block() {
	std::cout << "Test block :" << std::endl;
	nn::Mat a(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
	nn::Mat b = a.block(1, 1, 2, 3);
	a.print(std::cout, "A", 0);
	b.print(std::cout, "B", 0);
}


int main() {
	general_tests();
	test_operator_eqeq();
	test_mat_prod();
	test_block();
	return 0;
}