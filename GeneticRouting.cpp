// GeneticRouting.cpp : Defines the entry point for the application.
//

#include "GeneticModelBase.h"

using namespace std;

//Settings
int NUM_OF_MODELS = 100;
int NUM_MODELS_FOR_CROSSOVER = 10;
double MUTATION_CHANCE = 0.05;
double MUTATION_SD = 0.01;
int NUM_OF_ITERATIONS = 100;

random_device random;
mt19937 generator(random());
uniform_int_distribution uniformInt(0, NUM_MODELS_FOR_CROSSOVER - 1);

int main()
{
	vector<unique_ptr<Model>> models;
	vector<tuple<int, double>> outputs = {};

	for (int i = 0; i < NUM_OF_MODELS; i++) {
		auto model = make_unique<Model>();

		model->AddDenseLayer(2, "tanh");
		model->AddDenseLayer(3, "tanh");
		model->AddDenseLayer(1, "sigmoid");

		model->GenerateRawList();

		models.push_back(std::move(model));
		outputs.push_back({ i, 0 });
	}

	//For this genetic experiment, the model will take in 2 numbers between 0 and 1 ,a and b.
	//Output 0 if a < b, output 1 if a >= b

	vector<tuple<double, double, double>> inputs = {};

	double res;
	for (double i = 0.01; i < 1; i += 0.01) {
		for (double j = 0.01; j < 1; j += 1) {
			if (i < j) res = 0;
			else res = 1;

			inputs.push_back({i, j, res});
		}
	}

	for (int i = 0; i < NUM_OF_ITERATIONS; i++) {
		if ((i + 1) % 10 == 0) cout << "Iterations : " << i + 1 << "/" << NUM_OF_ITERATIONS << endl;

		for (int i = 0; i < outputs.size(); i++) {
			get<0>(outputs[i]) = i;
		}

		for (int j = 0; j < models.size(); j++) {
			double offset = 0;

			//Testing phase
			for (tuple<double, double, double> set : inputs) {
				vector<double> modelInputs = { get<0>(set), get<1>(set) };
				double out = models[j]->RunAlgorithm(modelInputs)[0];

				offset += fabs(get<2>(set) - out);
			}

			get<1>(outputs[j]) = offset;
		}

		//Evolution phase
		sort(outputs.begin(), outputs.end(),
			[](const tuple<int, double>& a, const tuple<int, double>& b) { return get<1>(a) < get<1>(b); }
		);

		//cout << "Most effective score : " << get<1>(outputs[0]) << endl;

		vector<unique_ptr<Model>> newGeneration = {};

		int bestModelIdx = get<0>(outputs[0]);

		//cout << "Best model index : " << bestModelIdx << endl;

		newGeneration.push_back(models[bestModelIdx]->Clone());

		//cout << outputs.size() << " Outputs size " << endl;

		while (newGeneration.size() < NUM_OF_MODELS) {
			int parent1Idx = get<0>(outputs[uniformInt(generator)]);
			int parent2Idx = get<0>(outputs[uniformInt(generator)]);

			while (parent2Idx == parent1Idx && NUM_MODELS_FOR_CROSSOVER > 1) {
				parent2Idx = get<0>(outputs[uniformInt(generator)]);
			}

			auto child = models[parent1Idx]->Clone();

			child->Crossover(models[parent2Idx].get(), MUTATION_CHANCE, MUTATION_SD);
			child->GenerateRawList();

			newGeneration.push_back(std::move(child));
		}

		models = std::move(newGeneration);
	}

	sort(outputs.begin(), outputs.end(),
		[](const tuple<int, double>& a, const tuple<int, double>& b) { return get<1>(a) < get<1>(b); }
	);

	cout << "Most effective score : " << get<1>(outputs[0]) << endl;

	const auto filepath = filesystem::current_path() / "Models" / "1.json";

	cout << "filepath exists?: " << filepath << endl;

	models[get<0>(outputs[0])]->SaveModelWeights(filepath);
	//for (auto output : outputs) cout << get<1>(output) << endl;

	return 0;
}
