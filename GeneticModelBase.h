#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

/*
My goal is to make a generic neuroevolving model base, which can be adapted for whatever else I want to
do.
*/

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
	int GetNumLayers() const {
		return layers.size();
	}

	vector<Perceptron*> rawPerceptrons = {};
	vector<Edge*> rawEdges = {};

	vector<double> RunAlgorithm(vector<double> inputs);

	void AddDenseLayer(int numNodes, string activationFunction);

	void Crossover(Model* otherModel, double mutationChance, double mutationSD);

	void GenerateRawList();

	unique_ptr<Model> Clone();
};

using namespace std;

random_device rd;
mt19937 gen(rd());
normal_distribution<double> norm(0.0, 1.0);
uniform_real_distribution<double> uniform(0.0, 1.0);

double Perceptron::CalculateSum() {
	double sum = bias;

	for (const auto& edge : edges) sum += edge->start->result * edge->weight;

	return sum;
}

void Perceptron::ActivationFunction() {
	double out = 0;

	double sum = CalculateSum();
	if (activationFunc == "tanh") out = tanh(sum);
	else if (activationFunc == "sigmoid") out = 1 / (1 + exp(-sum));

	else cout << "Activation function " << activationFunc << " unknown";

	result = out;
}

vector<double> Model::RunAlgorithm(vector<double> inputs) {
	auto& inputLayer = layers[0];

	for (int i = 0; i < inputLayer.size(); i++) inputLayer[i]->result = inputs[i];

	for (int i = 1; i < layers.size(); i++) {
		auto& targetLayer = layers[i];
		for (int j = 0; j < targetLayer.size(); j++) targetLayer[j]->ActivationFunction();
	}

	vector<double> out = {};
	for (const auto& perceptron : layers.back()) out.push_back(perceptron->result);

	return out;
}

void Model::AddDenseLayer(int numPerceptrons, string activationFunction) {
	vector<unique_ptr<Perceptron>> newLayer;

	for (int i = 0; i < numPerceptrons; i++) {
		auto perceptron = make_unique<Perceptron>(activationFunction);
		perceptron->bias = norm(gen);

		if (!layers.empty()) {
			auto& previousLayer = layers.back();
			for (const auto& previousPerceptron : previousLayer) {
				double randomWeight = norm(gen);
				perceptron->edges.push_back(make_unique<Edge>(previousPerceptron.get(), perceptron.get(), randomWeight));

			}
		}

		newLayer.push_back(move(perceptron));
	}

	layers.push_back(move(newLayer));
}

void Model::Crossover(Model* otherModel, double mutationChance, double mutationSD) {
	normal_distribution<double> mutator(0.0, mutationSD);

	for (int i = 0; i < rawPerceptrons.size(); i++) {
		if (uniform(gen) <= 0.5) rawPerceptrons[i]->bias = otherModel->rawPerceptrons[i]->bias;
		if (uniform(gen) <= mutationChance) rawPerceptrons[i]->bias += mutator(gen);
	}

	for (int i = 0; i < rawEdges.size(); i++) {
		if (uniform(gen) <= 0.5) rawEdges[i]->weight = otherModel->rawEdges[i]->weight;
		if (uniform(gen) <= mutationChance) rawEdges[i]->weight += mutator(gen);
	}
}

void Model::GenerateRawList() {
	rawPerceptrons.clear();
	rawEdges.clear();

	for (auto& layer : layers) {
		for (auto& perceptron : layer) {
			rawPerceptrons.push_back(perceptron.get());

			for (auto& edge : perceptron->edges) rawEdges.push_back(edge.get());
		}

	}
}

unique_ptr<Model> Model::Clone() {
	//cout << "Clone 1 " << endl;

	auto clonedModel = make_unique<Model>();

	for (const auto& layer : layers) {
		if (layer.empty()) continue;

		string activationFunc = layer[0]->activationFunc;
		clonedModel->AddDenseLayer(layer.size(), activationFunc);
	}

	clonedModel->GenerateRawList();

	//cout << "Clone 2 " << endl;

	//cout << rawEdges.size() << endl;

	for (size_t i = 0; i < rawEdges.size(); i++) {
		clonedModel->rawEdges[i]->weight = rawEdges[i]->weight;
	}

	//cout << "Clone 3" << endl;

	for (size_t i = 0; i < rawPerceptrons.size(); i++) {
		clonedModel->rawPerceptrons[i]->bias = rawPerceptrons[i]->bias;
	}
	return clonedModel;
}