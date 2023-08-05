#include "../nn.hpp"

#include <fstream>
#include <iostream>
#include <utility>

std::pair<nn::Mat, nn::Mat> load_mnist(const std::string& img_fn, const std::string& labels_fn) {
	std::ifstream imgf;
	std::ifstream labelf;

	imgf.open(img_fn, std::ios::in | std::ios::binary);
	labelf.open(labels_fn, std::ios::in | std::ios::binary);

	uint32_t magic_img, magic_label;
	uint32_t n_img, n_label;

	imgf.read((char*)(&magic_img), 4);
	magic_img = __builtin_bswap32(magic_img);
	if (magic_img != 2051) {
		std::cerr << "Wrong magic number for the image file : " << magic_img << std::endl;
		NN_ASSERT(false);
	}

	labelf.read((char*)(&magic_label), 4);
	magic_label = __builtin_bswap32(magic_label);
	if (magic_label != 2049) {
		std::cerr << "Wrong magic number for the label file : " <<  magic_label << std::endl;
		NN_ASSERT(false);
	}


	imgf.read((char*)&n_img, 4);
	labelf.read((char*)&n_label, 4);
	n_img = __builtin_bswap32(n_img);
	n_label = __builtin_bswap32(n_label);
	std::cout << n_img << " images, " << n_label << " labels" << std::endl;

	uint32_t nrows, ncols;
	imgf.read((char*)&nrows, 4);
	imgf.read((char*)&ncols, 4);
	nrows = __builtin_bswap32(nrows);
	ncols = __builtin_bswap32(ncols);
	if (nrows != 28 || ncols != 28) {
		std::cerr << "Problem : nrows=" << nrows << " and ncols=" << ncols << std::endl;
		NN_ASSERT(false);
	}

	//Load the images
	std::vector<uint8_t> rawdata(n_img*nrows*ncols);
	imgf.read((char*)&rawdata[0], n_img*nrows*ncols);
	nn::Mat data(nrows*ncols, n_img);
	for (int i = 0; i < n_img; ++i) {
		for (int j = 0; j < nrows*ncols; ++j) {
			data.at(j, i) = rawdata[i * nrows*ncols + j];
		}
	}

	//Load the labels
	std::vector<uint8_t> rawlabels(n_img);
	labelf.read((char*)&rawlabels[0], n_img);
	nn::Mat labels(1, n_img);
	for (int i = 0; i < n_img; ++i) {
		labels.at(0, i) = rawlabels[i];
	}

	imgf.close();
	labelf.close();

	return std::make_pair(data, labels);
}


nn::Mat onehot_encode(const nn::Mat& labels) {
	const unsigned N = labels.get_cols();
	nn::Mat Y(10, N);
	for (int j = 0; j < N; ++j) {
		Y.at(labels.at(0, j), j) = 1;
	}
	return Y;
}



int main() {
	const auto data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	const nn::Mat X = std::move(data.first);
	const nn::Mat labels = std::move(data.second);
	const nn::Mat Y = onehot_encode(labels);

	const auto testdata = load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	const nn::Mat testX = std::move(data.first);
	const nn::Mat testlabels = std::move(data.second);
	const nn::Mat testY = onehot_encode(labels);
	/*
	nn::NN net({784, 32, nn::act::RELU, 10, nn::act::RELU});
	//nn::NN net;
	net.train_batch(X, Y, 10000, 0.1, 100);
	net.save("testrelusmall.dat");

	net.test(testX, testY);
	//test MSE = 0.47974
	*/
	
	nn::NN net;
	net.load("testrelusmall.dat");
	std::cout << "A few examples : " << std::endl;
	const nn::Mat x = testX.block(0, 0, 784, 10);
	const nn::Mat y = testlabels.block(0, 0, 1, 10);
	const nn::Mat pred = net(x);
	const nn::Mat ans = pred.col_argmax();
	for (int i = 0; i < 10; ++i) {
		std::cout << "truth=" << y.at(0, i) << ", pred=" << ans.at(0, i) << std::endl;
		std::cout << pred.block(0, i, 10, 1) << std::endl;
	}
	
	return 0;
}