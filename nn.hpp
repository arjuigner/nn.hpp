#pragma once

#include "matrix.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace nn {

	inline F __sqr(const F x) { return x*x; }
	inline F __sigmoid(const F x) { return 1.0 / ( 1.0 + exp(-x) ); }

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
	};

	//Most basic type of Layer, a fully connected linear layer.
	class Linear : public Layer {
	public:
		Linear() = delete;
		Linear(const unsigned in_sz, const unsigned out_sz)
			{ w = Mat::Randn(out_sz, in_sz); }
		
		~Linear() = default;
	
		Mat forward(const Mat& x) 		   override;
		Mat backward(const Mat& grad_out)  override;
		void grad_descent_step(const F lr) override;

		Mat& get_weights() { return w; }
		const Mat& get_weights() const { return w; }

	private:
		Mat w;
		Mat _x;
		Mat grad_w;
	};


	//Sigmoid activation layer.
	class Sigmoid : public Layer {
	public:
		Sigmoid() = default;
		~Sigmoid() = default;

		Mat forward(const Mat& x) 		   override;
		Mat backward(const Mat& grad_out)  override;
		void grad_descent_step(const F lr) override {}

	private:
		Mat _out;
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
		NN() = delete;
		NN(const std::vector<int>& layers_sz);
		~NN() = default;

		Mat operator()(const Mat& X);

		void train(const Mat& X, const Mat& Y, const unsigned iterations, const F lr);

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
		return w * x;
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
		return  w.transpose() * grad_out;
	}


	void Linear::grad_descent_step(const F lr) {
		w += -lr * grad_w;
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
			}

			if (layers_sz[i] == act::SIGMOID) {
				layers.push_back(std::make_unique<Sigmoid>());
				std::cout << "Added a sigmoid layer" << std::endl;
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

		for (int it = 0; it < iterations; ++it) {

			//Forward pass
			outputs[0] = layers[0]->forward(X);
			for (int l = 1; l < layers.size(); ++l) {
				outputs[l] = layers[l]->forward(outputs[l - 1]);
			}

			//Calculate and print error
			cost = mse.calc_cost(*outputs.crbegin(), Y);
			if (it % 100 == 0)
				std::cout << "Iteration #" << it <<" ; Cost=" << cost << std::endl;

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
}

