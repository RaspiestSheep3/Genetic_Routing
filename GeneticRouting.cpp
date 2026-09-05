// GeneticRouting.cpp : Defines the entry point for the application.
//

#include "GeneticRouting.h"
//#include "ModelBase.cpp"

using namespace std;

int main()
{
	Model model;

	//Making a 2-8-1 model

	model.AddDenseLayer(2, "tanh");
	model.AddDenseLayer(8, "tanh");
	model.AddDenseLayer(1, "tanh");


	vector<double> inputs = {0.4, 0.3};
	vector<double> out = model.RunAlgorithm(inputs);

	cout << out[0] << endl;

	cout << "Hello CMake." << endl;
	return 0;
}
