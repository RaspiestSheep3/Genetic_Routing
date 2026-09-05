// GeneticRouting.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>

// TODO: Reference additional headers your program requires here.
#include <vector>
#include <cmath>

using namespace std;

class Perceptron;

class Edge {
public:
	Perceptron* start;
	Perceptron* end;
	double weight = 0.1;

	Edge(Perceptron* startNode, Perceptron* endNode, double w)
		: start(startNode), end(endNode), weight(w) {
	}
};

class Perceptron {
public:
	vector<unique_ptr<Edge>> edges = {};
	double bias = 0.1;
	double result = 0.1;
	string activationFunc;

	Perceptron(string activation) : activationFunc(activation) {}

	void ActivationFunction();

private:
	double CalculateSum();
};

class Model {
public:
	vector<vector<unique_ptr<Perceptron>>> layers = {}; //[0] is input layer, [numLayers - 1] is output layer
	int numLayers = layers.size();

	vector<double> RunAlgorithm(vector<double> inputs);

	void AddDenseLayer(int numNodes, string activationFunction);
};