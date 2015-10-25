/*
 * main.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: sabin
 */

#include <iostream>
#include <FwdNet.h>


int main( int argc, char** argv)
{
	ann::FwdNet::Topology topology(3);
	topology[0]=2;
	topology[1]=2;
	topology[2]=1;
	ann::FwdNet artificialNetwork(topology);

	ann::FwdNet::TrainingDataSet trainingDataSet;
	ann::FwdNet::InputData inputs(2);
	ann::FwdNet::TargetData targets(1);

	inputs[0]=0;
	inputs[1]=0;
	targets[0]=0;
	trainingDataSet.push_back(std::make_pair(inputs, targets));

	inputs[0]=0;
	inputs[1]=1;
	targets[0]=1;
	trainingDataSet.push_back(std::make_pair(inputs, targets));

	inputs[0]=1;
	inputs[1]=0;
	targets[0]=1;
	trainingDataSet.push_back(std::make_pair(inputs, targets));

	inputs[0]=1;
	inputs[1]=1;
	targets[0]=0;
	trainingDataSet.push_back(std::make_pair(inputs, targets));


	artificialNetwork.doTraining(trainingDataSet, 0.1);

	ann::FwdNet::OutputData outputs;
	inputs[0]=0;
	inputs[1]=0;
	artificialNetwork.processInputs(inputs);
	artificialNetwork.getOutputs(outputs);

	inputs[0]=0;
	inputs[1]=1;
	artificialNetwork.processInputs(inputs);
	artificialNetwork.getOutputs(outputs);

	inputs[0]=1;
	inputs[1]=0;
	artificialNetwork.processInputs(inputs);
	artificialNetwork.getOutputs(outputs);

	inputs[0]=1;
	inputs[1]=1;
	artificialNetwork.processInputs(inputs);
	artificialNetwork.getOutputs(outputs);

	artificialNetwork.dump();
	return 0;
}


