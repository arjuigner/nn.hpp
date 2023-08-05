#include "../nn.hpp"

#include <iostream>

int main() {
	//Create the data
	const nn::Mat x(2, 4, {1, 1, 0, 0, 1, 0, 1, 0});
	const nn::Mat y(1, 4, {0, 1, 1, 0});

	//Print it
	x.print(std::cout, "X", 0);
	y.print(std::cout, "Y", 0);

	//Train the nn
	nn::NN net({2, 3, nn::act::SIGMOID, 1, nn::act::SIGMOID});
	net.train(x, y, 30000, 1.0);
	

	//Print results
	const nn::Mat pred = net(x);
	pred.print(std::cout, "pred", 0);
	y.print(std::cout, "Y", 0);
	return 0;
}