#pragma once

#include "matrix.hpp"

#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace nn {

	inline F __sqr(const F x) { return x*x; }
	inline F __sigmoid(const F x) { return 1.0 / ( 1.0 + exp(-x) ); }
	inline F __relu(const F x) { return x > 0 ? x : 0; }

	//Constants used by some functions to refer to different kind of activation layers.
	namespace act {
		constexpr int SIGMOID = -1;
		constexpr int RELU    = -2;
	}

	//Base Layer class ; all types of layers inherit from Layer.
	class Layer {
	public:
		Layer() = default;
		~Layer() = default;

		virtual Mat forward(const Mat& x) = 0;
		virtual Mat backward(const Mat& grad_out) = 0;
		virtual void grad_descent_step(const F lr) = 0;
		virtual std::string layer_name() const = 0;

		virtual void save_bin(std::ofstream& f) const { std::cerr << "Layer::save_bin called wtf are you doing" << std::endl; }
		virtual void load_bin(std::ifstream& f) { std::cerr << "Layer::load_bin called wtf are you doing" << std::endl; }
	};

	//Most basic type of Layer, a fully connected linear layer.
	class Linear : public Layer {
	public:
		Linear() = delete;
		Linear(const unsigned in_sz, const unsigned out_sz) :
			b(out_sz, 1)
			{ w = Mat::Randn(out_sz, in_sz); }
		
		~Linear() = default;
	
		Mat forward(const Mat& x) 		   override;
		Mat backward(const Mat& grad_out)  override;
		void grad_descent_step(const F lr) override;
		std::string layer_name() const override { return "Lin"; }
		
		void save_bin(std::ofstream& f) const override;
		void load_bin(std::ifstream& f) override;

		Mat& get_weights() { return w; }
		const Mat& get_weights() const { return w; }

	private:
		Mat w;
		Mat b;
		Mat _x;
		Mat grad_w;
		Mat grad_b;
	};


	//Sigmoid activation layer.
	class Sigmoid : public Layer {
	public:
		Sigmoid() = default;
		~Sigmoid() = default;

		Mat forward(const Mat& x) 		   override;
		Mat backward(const Mat& grad_out)  override;
		void grad_descent_step(const F lr) override {}
		std::string layer_name() const override { return "Sig"; }

	private:
		Mat _out;
	};


	class ReLU : public Layer {
	public:
		ReLU() = default;
		~ReLU() = default;
		Mat forward(const Mat& x) 		   override;
		Mat backward(const Mat& grad_out)  override;
		void grad_descent_step(const F lr) override {}
		std::string layer_name() const override { return "Rel"; }
	private:
		Mat _grad;
	};


	//Cost function "layer" (it is particular and doesn't inherit Layer because it is supposed to output 
	//a scalar instead of a matrix and might be used differently as well.)
	//
	//TODO : implement a base Cost class and build multiple children classes which represent
	//multiple possible cost functions.
	class MSE {
	public:
		MSE() : ppred(nullptr), ptruth(nullptr) {}

		F calc_cost(const Mat& pred, const Mat& truth);
		Mat backward();
	private:
		Mat const* ppred;
		Mat const* ptruth;
	};


	//class ReLU : public Layer {};

	/* Add more layers... */


	/*
	*	Represents a neural network, which itself contains a vector of layers and a cost function.
	*	This class also deals with the training.
	*/
	class NN {
	public:
		NN() = default;
		NN(const std::vector<int>& layers_sz);
		~NN() = default;

		Mat operator()(const Mat& X);

		void train(const Mat& X, const Mat& Y, const unsigned iterations, const F lr);
		void train_batch(const Mat& X, const Mat& Y, const unsigned iterations, const F lr, const unsigned batch_sz);

		void test(const Mat& X, const Mat& Y);

		int save(const std::string& filename);
		int load(const std::string& filename);

	private:
		std::vector<std::unique_ptr<Layer>> layers;
		std::vector<Mat> outputs;
		std::vector<Mat> grads;
		MSE mse;
		F cost;
	};


	//////////////////////
	// Implementations


	Mat Linear::forward(const Mat& x) {
		//Store the input for the backward pass.
		_x = x;
		

		//TODO explicitly write the calculations of the matrix product + the bias
		//here to avoid going through the whole matrix multiple times.
		//Cons : will help reduce the number of O(n^2) operations, however matrix product is O(n^3).
		//But it might still be useful for smaller layers.
		const unsigned N = x.get_cols(); //number of samples
		nn::Mat y = w * x;
		const unsigned R = y.get_rows(); //number of rows in the output

		for (int j = 0; j < N; ++j) {
			for (int i = 0; i < R; ++i) {
				y.at(i, j) += b.at(i, 0);
			}
		}
		return y;
	}


	Mat Linear::backward(const Mat& grad_out) {
		grad_w = Mat(w.get_rows(), w.get_cols()); //Also fills it with 0's.
		for (int i = 0; i < grad_w.get_rows(); ++i) {
			for (int j = 0; j < grad_w.get_cols(); ++j) {
				for (int k = 0; k < grad_out.get_cols(); ++k) {
					grad_w.at(i, j) += grad_out.at(i, k) * _x.at(j, k);
				}				
			}
		}


		//TODO : implement Mat::hsum, leading to:
		//grad_b = grad_out.hsum()
		grad_b = Mat(b.get_rows(), 1); //Also fills it with 0's.
		for (int i = 0; i < b.get_rows(); ++i) {
			for (int j = 0; j < grad_out.get_cols(); ++j) {
				grad_b.at(i, 0) += grad_out.at(i, j);
			}
		}
		return  w.transpose() * grad_out;
	}


	void Linear::grad_descent_step(const F lr) {
		w += -lr * grad_w;
		b += -lr * grad_b;
	}


	void Linear::save_bin(std::ofstream& f) const {
		uint16_t in = w.get_cols();
		uint16_t out = w.get_rows();
		f.write((char*)&in, sizeof(in));
		f.write((char*)&out, sizeof(out));
		f.write((char*)&w.at(0, 0), sizeof(F) * w.get_rows() * w.get_cols());
		f.write((char*)&b.at(0, 0), sizeof(F) * b.get_rows() * b.get_cols());
	}

	void Linear::load_bin(std::ifstream& f) {
		f.read((char*)&w.at(0, 0), sizeof(F) * w.get_rows() * w.get_cols());
		f.read((char*)&b.at(0, 0), sizeof(F) * b.get_rows() * b.get_cols());
	}


	Mat Sigmoid::forward(const Mat& x) {
		const unsigned R = x.get_rows();
		const unsigned C = x.get_cols();
		_out = Mat(R, C);
		for (int i = 0; i < R; ++i) {
			for (int j = 0; j < C; ++j) {
				_out.at(i, j) = __sigmoid(x.at(i, j));
			}
		}
		return _out;
	}


	Mat Sigmoid::backward(const Mat& grad_out) {
		const unsigned R = _out.get_rows();
		const unsigned C = _out.get_cols();
		Mat grad_in(R, C);
		for (int i = 0; i < R; ++i) {
			for (int j = 0; j < C; ++j) {
				grad_in.at(i, j) = _out.at(i, j) * (1.0 - _out.at(i, j)) * grad_out.at(i, j);
			}
		}
		return grad_in;
	}


	Mat ReLU::forward(const Mat& x) {
		const unsigned R = x.get_rows();
		const unsigned C = x.get_cols();
		Mat out = Mat(R, C);
		_grad = Mat(R, C);
		for (int i = 0; i < R; ++i) {
			for (int j = 0; j < C; ++j) {
				out.at(i, j) = __relu(x.at(i, j));
				_grad.at(i, j) = x.at(i, j) > 0 ? 1 : 0;
			}
		}
		return out;
	}


	Mat ReLU::backward(const Mat& grad_out) {
		return _grad;
	}


	F MSE::calc_cost(const Mat& pred, const Mat& truth) {
		const unsigned N = truth.get_cols();
		F cost = 0;
		for (unsigned j = 0; j < N; ++j) {
			for (unsigned i = 0; i < truth.get_rows(); ++i) {
				cost += 1.0/N * __sqr(pred.at(i, j) - truth.at(i, j));
			}
		}
		ppred = &pred;
		ptruth = &truth;
		return cost/2;
	}


	Mat MSE::backward() {
		Mat grad(ppred->get_rows(), ppred->get_cols());
		const unsigned N = ptruth->get_cols();
		for (int j = 0; j < N; ++j)
			for (int i = 0; i < ptruth->get_rows(); ++i) {
				grad.at(i, j) += 1.0/N * (ppred->at(i, j) - ptruth->at(i, j));
			}
		return grad;
	}


	NN::NN(const std::vector<int>& layers_sz) {
		unsigned cur_sz, prev_sz;
		NN_ASSERT(layers_sz[0] > 0);
		NN_ASSERT(layers_sz[1] > 0);
		cur_sz = layers_sz[0];
		for (int i = 1; i < layers_sz.size(); ++i) {
			if (layers_sz[i] > 0) {
				prev_sz = cur_sz;
				cur_sz = layers_sz[i];
				layers.push_back(std::make_unique<Linear>(prev_sz, cur_sz));
				std::cout << "Added a linear layer,  in=" << prev_sz << ", out=" << cur_sz << std::endl;
			} else if (layers_sz[i] == act::SIGMOID) {
				layers.push_back(std::make_unique<Sigmoid>());
				std::cout << "Added a sigmoid layer" << std::endl;
			} else if (layers_sz[i] == act::RELU) {
				layers.push_back(std::make_unique<Sigmoid>());
				std::cout << "Added a ReLU layer" << std::endl;
			}
		}

		outputs.resize(layers.size());
		grads.resize(layers.size());

		cost = -1;
	}


	Mat NN::operator()(const Mat& X) {
		outputs[0] = layers[0]->forward(X);
		for (int l = 1; l < layers.size(); ++l) {
			outputs[l] = layers[l]->forward(outputs[l - 1]);
		}
		return *outputs.crbegin();
	}


	void NN::train(const Mat& X, const Mat& Y, const unsigned iterations, const F lr) {
		std::cout << "Start training." << std::endl;

		for (int it = 1; it <= iterations + 1; ++it) {

			//Forward pass
			outputs[0] = layers[0]->forward(X);
			for (int l = 1; l < layers.size(); ++l) {
				outputs[l] = layers[l]->forward(outputs[l - 1]);
			}

			//Calculate and print error
			cost = mse.calc_cost(*outputs.crbegin(), Y);
			if (it <= iterations && it % 1 == 0) {
				std::cout << "Iteration #" << it <<" ; Cost=" << cost << std::endl;
			} else if (it == iterations + 1) {
				std::cout << "Cost on the next batch after the last update = " << cost << std::endl;
				break; 
			}

			//Backward pass
			*grads.rbegin() = mse.backward();
			for (int l = layers.size() - 2; l >= 0; --l) {
				grads[l] = layers[l+1]->backward(grads[l+1]);
			}
			layers[0]->backward(grads[0]);

			//Update weights
			for (int l = layers.size() - 1; l >= 0; --l) {
				layers[l]->grad_descent_step(lr);
			}
		}
	}


	void NN::train_batch(const Mat& X, const Mat& Y, const unsigned iterations, const F lr, const unsigned batch_sz) {
		NN_ASSERT(X.get_cols() % batch_sz == 0 && "Batch size must divide the number of samples.");
		//TODO get rid of the above condition.

		std::cout << "Start training." << std::endl;

		for (int it = 1; it <= iterations + 1; ++it) {
			const unsigned batch_start = (it * batch_sz) % X.get_cols();
			nn::Mat bX = X.block(0, batch_start, X.get_rows(), batch_sz);
			nn::Mat bY = Y.block(0, batch_start, Y.get_rows(), batch_sz);

			//Forward pass
			outputs[0] = layers[0]->forward(bX);
			for (int l = 1; l < layers.size(); ++l) {
				outputs[l] = layers[l]->forward(outputs[l - 1]);
			}


			//Calculate and print error
			cost = mse.calc_cost(*outputs.crbegin(), bY);
			if (it <= iterations && it % 1 == 0) {
				std::cout << "Iteration #" << it <<" ; Cost=" << cost << std::endl;
			} else if (it == iterations + 1) {
				std::cout << "Cost on the next batch after the last update = " << cost << std::endl;
				break; 
			}


			//Backward pass
			*grads.rbegin() = mse.backward();
			for (int l = layers.size() - 2; l >= 0; --l) {
				grads[l] = layers[l+1]->backward(grads[l+1]);
			}
			layers[0]->backward(grads[0]);


			//Update weights
			for (int l = layers.size() - 1; l >= 0; --l) {
				layers[l]->grad_descent_step(lr);
			}
		}
	}


	void NN::test(const nn::Mat& X, const nn::Mat& Y) {
		std::cout << "Stats on the test dataset :" << std::endl;
		const nn::Mat pred = this->operator()(X);
		const F mse_cost = mse.calc_cost(pred, Y);
		std::cout << "	MSE = " << mse_cost << std::endl;
	}


	int NN::load(const std::string& filename) {
		std::ifstream f;
		f.open(filename, std::ios::in | std::ios::binary);
		if (!f.is_open()) return 1;

		//Read the architecture and the data of the nn.
		//After each layer there is a magic number, 2002 if everything is right and it is not
		//the last layer, 2003 if it was the last layer.
		uint32_t magic = 0;
		while (magic != 2003) {
			std::string layername;
			layername.resize(3);
			f.read((char*)layername.c_str(), 3);
			if (layername == "Lin") {
				std::cout << "Loading a linear layer :";
				uint16_t in, out;
				f.read((char*)&in, sizeof(in));
				f.read((char*)&out, sizeof(out));
				std::cout << " in=" << in << ", out=" << out << std::endl; 
				auto l = std::make_unique<Linear>(in, out);
				l->load_bin(f);
				layers.push_back(std::move(l));
			} else if (layername == "Sig") {
				layers.push_back(std::make_unique<Sigmoid>());
				std::cout << "Loading a Sigmoid layer" << std::endl;
			} else if (layername == "Rel") {
				layers.push_back(std::make_unique<ReLU>());
				std::cout << "Loading a ReLU layer" << std::endl;
			} else {
				std::cout << "wtf is going on" << std::endl;
			}

			f.read((char*)&magic, sizeof(magic));
			if (magic != 2002 && magic != 2003) {
				std::cerr << "Wrong magic after layer, magic=" << magic << std::endl;
				f.close();
				return 1;
			}
		}
		f.close();

		outputs.resize(layers.size());
		grads.resize(layers.size());

		cost = -1;
		return 0;
	}
	

	int NN::save(const std::string& filename) {
		std::ofstream f;
		f.open(filename, std::ios::out | std::ios::binary);
		if (!f.is_open()) return 1;

		uint32_t magic_end = 2003;
		uint32_t magic = 2002;
		for (int i = 0; i < layers.size(); ++i) {
			std::string layername = layers[i]->layer_name();
			f.write(layername.c_str(), 3);
			if (layername == "Lin") {
				layers[i]->save_bin(f);
			}

			if (i == layers.size() - 1) { //Last layer
				f.write((char*)&magic_end, sizeof(magic_end));
			} else {
				f.write((char*)&magic, sizeof(magic));
			}
		}
		f.close();
		return 0;
	}
}

