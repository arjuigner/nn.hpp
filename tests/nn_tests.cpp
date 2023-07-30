#include "../nn.hpp"

#include <iostream>

void test_mse() {
	const nn::Mat truth(3, 2, {0, 2, 2, 3, 5, 8});
	const nn::Mat pred(3, 2, {1, 2, 3, 4, 5, 6});
	nn::MSE mse;
	const nn::F cost = mse.calc_cost(pred, truth);
	const nn::Mat grad = mse.backward();

	pred.print(std::cout, "pred", 0);
	truth.print(std::cout, "truth", 0);

	std::cout << "Cost = " << cost << std::endl;
	grad.print(std::cout, "grad", 0);

}


void test_linear() {
	const nn::Mat A(2, 3, {1, -1, 2, 0, -1, 1});
	const nn::Mat x(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
	const nn::Mat y = A * x;
	std::cout << y << std::endl;

	nn::Linear linear(3, 2);
	nn::MSE mse;
	nn::Mat pred;
	nn::Mat grad_out;
	nn::F cost = 0;
	for (int i = 0; i < 1000; ++i) {
		pred = linear.forward(x);
		cost = mse.calc_cost(pred, y);
		std::cout << "Iteration " << i << ", cost = " << cost << std::endl;

		grad_out = mse.backward();
		linear.backward(grad_out);
		linear.grad_descent_step(0.01);
	}

	linear.get_weights().print(std::cout, "w", 0);
	(linear.get_weights() * x).print(std::cout, "pred", 0);
	y.print(std::cout, "truth", 0);
}


int main() {
	nn::Sigmoid sig1, sig2;
	//const nn::Mat A(2, 3, {1, -1, 2, 0, -1, 1});
	//const nn::Mat x(3, 4, {1, 1, 0, -1, 2, -2, -1, 1, 0, 2, 1, 0});
	//const nn::Mat y = sig1.forward(A * x);

	const nn::Mat x(2, 4, {1, 1, 0, 0, 1, 0, 1, 0});
	const nn::Mat y(1, 4, {0, 1, 1, 0});

	x.print(std::cout, "X", 0);
	y.print(std::cout, "Y", 0);

	nn::NN net({2, 2, nn::act::SIGMOID, 1, nn::act::SIGMOID});
	net.train(x, y, 30000, 1.0);
	const nn::Mat pred = net(x);
	pred.print(std::cout, "pred", 0);
	y.print(std::cout, "Y", 0);

	/*lin.get_weights().print(std::cout, "w", 0);
	sig.forward(linear.get_weights() * x).print(std::cout, "pred", 0);
	y.print(std::cout, "truth", 0);
*/
	return 0;
}