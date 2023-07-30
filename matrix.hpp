#pragma once

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define NN_ASSERT assert

namespace nn {

	//Type used for all calculations, also the type of the elements of a matrix.
	using F = double;

	//A tab, used to print matrices.
	const char _tab[] = "   ";

	class Mat {
	public:
		Mat() = default;
		Mat(const unsigned r, const unsigned c) : rows(r), cols(c), elem(r*c, F(0)) {}
		Mat(const unsigned r, const unsigned c, const std::vector<F>& el) : 
			rows(r), cols(c), elem(el) {}
		Mat(const unsigned r, const unsigned c, std::vector<F>&& el) :
			rows(r), cols(c), elem(std::move(el)) {}

		Mat(const Mat& mat) = default;
		Mat(Mat&& mat) : rows(mat.rows), cols(mat.cols), elem(std::move(mat.elem)) {}
		Mat& operator=(const Mat& mat) = default;
		Mat& operator=(Mat&& mat) = default; //TODO make sure it works as intended.

		~Mat() = default;

		static Mat Randn(const unsigned r, const unsigned c);

		unsigned get_rows() const { return rows; }
		unsigned get_cols() const { return cols; }
		F& at(const unsigned r, const unsigned c) { return elem[cols * r + c]; }
		const F& at(const unsigned r, const unsigned c) const  { return elem[cols * r + c]; }

		Mat& reshape(const unsigned r, const unsigned c);
		Mat& fill(F val);
		Mat transpose();

		Mat& operator+=(const Mat& rhs)      ;
		Mat  operator+ (const Mat& rhs) const;
		Mat  operator* (const Mat& rhs) const;
		Mat& operator*=(const F rhs)         ;
		Mat  operator* (const F rhs)    const;

		bool operator==(const Mat& rhs) const;
		bool operator!=(const Mat& rhs) const;

		//name is the name of the matrix, indent is the number of tabs at the beginning of each line.
		void print(std::ostream& s, const std::string& name, const unsigned indent) const;

	private:
		unsigned rows;
		unsigned cols;
		std::vector<F> elem;
	};

	//Product with a scalar on the left
	Mat operator*(F a, const Mat& b);

	//Print with the syntax such as std::cout << my_mat << some_more_stuff_to_print
	std::ostream& operator<<(std::ostream& s, const Mat& mat);


	//////////////////////////////////////////////////////////////////////////////////////////////
	// IMPLEMENTATIONS


	Mat Mat::Randn(const unsigned r, const unsigned c) {
		Mat mat(r, c);
		std::mt19937 gen(42);
		std::normal_distribution<F> dist(0, 1);
		for (unsigned i = 0; i < r; ++i) {
			for (unsigned j = 0; j < c; ++j) {
				mat.at(i, j) = dist(gen);
			}
		}
		return mat;
	}


	Mat& Mat::reshape(const unsigned r, const unsigned c) {
		//Make sure the total number of elements doesn't change.
		NN_ASSERT(r*c == rows*cols);
		rows = r;
		cols = c;
		return *this;
	}


	Mat& Mat::fill(F val) {
		std::fill(elem.begin(), elem.end(), val);
		return *this;
	}


	Mat Mat::transpose() {
		std::vector<F> newelem(rows * cols);
		for(int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				newelem[c * rows + r] = elem[r * cols + c];
			}
		}
		/*
		elem = std::move(newelem);
		
		//Swap rows and cols.
		auto temp = rows;
		rows = cols;
		cols = temp;
		return *this;
		*/
		return Mat(cols, rows, std::move(newelem));
	}


	Mat& Mat::operator+=(const Mat& rhs) {
		NN_ASSERT(this->rows == rhs.rows);
		NN_ASSERT(this->cols == rhs.cols);

		for (int i = 0; i < rows*cols; ++i) 
			elem[i] += rhs.elem[i];
		return *this;
	}



	Mat Mat::operator+(const Mat& rhs) const {
		NN_ASSERT(this->rows == rhs.rows);
		NN_ASSERT(this->cols == rhs.cols);

		Mat out(rows, cols);
		for (int i = 0; i < rows*cols; ++i) 
			out.elem[i] = elem[i] + rhs.elem[i];
		return out;
	}


	Mat Mat::operator*(const Mat& rhs) const {
		NN_ASSERT(cols == rhs.rows);

		Mat out(rows, rhs.cols);
		out.fill(F(0));
		for (int r = 0; r < rows; ++r) 
			for (int c = 0; c < rhs.cols; ++c)
				for (int k = 0; k < cols; ++k)
					out.at(r, c) += this->at(r, k) * rhs.at(k, c);
		return out;
	}


	Mat& Mat::operator*=(const F rhs) {
		for (int i = 0; i < rows*cols; ++i) 
			elem[i] *= rhs;
		return *this;
	}


	Mat Mat::operator*(F rhs) const {
		Mat out(rows, cols);
		for (int i = 0; i < rows*cols; ++i) 
			out.elem[i] = elem[i] * rhs;
		return out;
	}


	bool Mat::operator==(const Mat& rhs) const {
		if (rows != rhs.rows) return false;
		if (cols != rhs.cols) return false;
		for (int i = 0; i < rows * cols; ++i) {
			if (elem[i] != rhs.elem[i])
				return false;
		} 
		return true;
	}


	bool Mat::operator!=(const Mat& rhs) const {
		return !(*this == rhs);
	}


	void Mat::print(std::ostream& s, const std::string& name, const unsigned indent) const {
		std::string ind = "";
		for (int i = 0; i < indent; ++i) 
			ind += _tab;

		if (name != "") {
			s << ind << name << "=\n";
		}
		s << ind << "[\n";
		for (int r = 0; r < rows; ++r) {
			s << ind << _tab;
			for (int c = 0; c < cols; ++c) {
				s << this->at(r, c) << ' ';
			}
			s << '\n';
		}
		s << ind << "]\n"; 
	}


	Mat operator*(F a, const Mat& b) {
		return b * a;
	}


	std::ostream& operator<<(std::ostream& s, const Mat& mat) {
		mat.print(s, "", 0);
		return s;
	}


}