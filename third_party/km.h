// The code is modified from https://github.com/yfji/Kuhn-Munkres

#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Eigen>
using namespace std;

class KM {
public:
	//	KM(float* data, int m, int n);
	KM(const Eigen::MatrixXf & data) {
		init(data);
	}

	~KM();

	int N;
	int front;
	int back;
	int* matchX;
	int* matchY;
	float* weights;

	void init(const Eigen::MatrixXf & data);
	void del();
	void compute();
	float maxWeight() {
		return max_w;
	}
	vector<int> getMatch(bool front2back = true);

private:
	
	float max_w;	
	float* flagX;
	float* flagY;
	char* usedX;
	char* usedY;
	
	void constructMatrix(const Eigen::MatrixXf & data, int m, int n);
	bool dfs(int v);
};