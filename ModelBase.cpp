#include "GeneticRouting.h"

using namespace std;

/*
My goal is to make a generic neuroevolving model base, which can be adapted for whatever else I want to 
do.
*/

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

	for (int i = 1; i < numLayers; i++) {
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

		if (!layers.empty()) {
			auto& previousLayer = layers.back();
			for (const auto& previousPerceptron : previousLayer) perceptron->edges.push_back(make_unique<Edge>(previousPerceptron.get(), perceptron.get(), 0.1));
		}

		newLayer.push_back(move(perceptron));
	}

	layers.push_back(move(newLayer));
}