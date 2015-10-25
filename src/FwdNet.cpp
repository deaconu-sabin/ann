/*
 * FwdNet.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: sabin
 */

#include <cmath>
#include <cassert>
#include <FwdNet.h>
#include <Log.h>

#include <Neuron.h>

namespace ann {

FwdNet::Layer::Layer(unsigned neuronsNo):
		bias(NULL)
{
	for(unsigned neuronIdx=0; neuronIdx < neuronsNo; ++neuronIdx)
	{
		neurons.push_back(new Neuron());
		DEBUG("  created Neuron " << neurons.back());
	}
	DEBUG("  created Bias ");
	bias = new Neuron();
}

void FwdNet::Layer::dump()
{
	for(unsigned neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx)
	{
		std::cout<<"  Neuron["<<neurons[neuronIdx]<<"]: "<< *(neurons[neuronIdx]);
	}
	// log bias info
	std::cout<<"  Bias["<<bias<<"]: "<< *bias<<"\n";
}

FwdNet::FwdNet(const Topology &topology):
		m_error(47.0),
		m_eta(0.3),
		m_alpha(0.2),
		m_isTrained(false)
{
	unsigned nLayers = topology.size();

	///create first layer
	m_layers.push_back(Layer(topology[0]));

	/// create all layers except input layer
	for(unsigned layerIdx = 1; layerIdx < nLayers; ++layerIdx)
	{
		DEBUG("Creating layer "<< layerIdx);
		Layer layer(topology[layerIdx]);
		Layer& prevLayer =  m_layers[layerIdx-1];

		for(unsigned neuronIdx = 0;
			neuronIdx < layer.neurons.size();
			++neuronIdx)
		{
			for(unsigned neuronPrevLayer = 0 ;
				neuronPrevLayer < prevLayer.neurons.size();
				++neuronPrevLayer)
			{
				layer.neurons[neuronIdx]->connectTo(prevLayer.neurons[neuronPrevLayer]);
			}
			layer.neurons[neuronIdx]->connectTo(prevLayer.bias);
		}
		m_layers.push_back(layer);
	}
}

void FwdNet::processInputs(const InputData& inputData)
{
	feedForward(inputData);
}

void FwdNet::doTraining(const TrainingDataSet& trainingDataSet , double acceptedError)
{
	INFO("Training has started...");
	unsigned epochCounter = 0;
	do
	{
		double  epochError = 0.0; // keep the maximum error for an epoch
		m_isTrained = true;
		for(unsigned trainingDataIdx = 0; trainingDataIdx < trainingDataSet.size(); ++trainingDataIdx )
		{
			const InputData& inputData = trainingDataSet[trainingDataIdx].first;
			const TargetData& targetData = trainingDataSet[trainingDataIdx].second;

			assert(inputData.size() == m_layers[0].neurons.size());
			assert(targetData.size()==m_layers.back().neurons.size());

			DEBUG("Feeding forward the " << trainingDataIdx << "th trainingData");
			feedForward(inputData);

			if(calculateRootMeanSquareErr(targetData) > acceptedError )
			{
				m_isTrained = false;
			}
			if(epochError < m_error)
			{
				epochError = m_error;
			}
			DEBUG("For TrainingData["<<trainingDataIdx<<"]; RMS Error = " << epochError);
			propagateBack(targetData);
		}
		INFO("For Epoch "<< std::setw(5)<<std::right<< epochCounter++ <<" error is "<<  epochError);
	}while( !m_isTrained && epochCounter < 300 );
	INFO("Training finished.");
}
void FwdNet::feedForward(const InputData& inputs)
{
	assert(inputs.size()==m_layers[0].neurons.size());
	/// assign inputs to first layer
	for(unsigned i = 0; i < inputs.size(); ++i)
	{
		m_layers[0].neurons[i]->setOutput(inputs[i]);
		DEBUG("  output of Layer[0] Neuron ["<<m_layers[0].neurons[i]<<"] = " << m_layers[0].neurons[i]->getOutput());
	}

	//feed forward. start from second layer (not with the inputLayer)
	for(unsigned layer = 1; layer < m_layers.size(); ++layer)
	{
		for(unsigned neuronIdx = 0; neuronIdx < m_layers[layer].neurons.size(); ++neuronIdx )
		{
			Neuron* currentNeuron = m_layers[layer].neurons[neuronIdx];
			currentNeuron->calculateOutput();
			DEBUG("  output of Layer["<<layer<<"] Neuron ["<<currentNeuron<<"] = " << currentNeuron->getOutput());
		}
	}
}
void FwdNet::propagateBack(const TargetData& targets)
{
	assert(targets.size() == m_layers.back().neurons.size());
	Layer& outputLayer = m_layers.back();

	/// calculate the output layer gradients
	for(unsigned neuronIdx = 0; neuronIdx < outputLayer.neurons.size(); ++neuronIdx)
	{
		Neuron* neuron = outputLayer.neurons[neuronIdx];
		double neuronOutput = neuron->getOutput();
		double delta = targets[neuronIdx] - neuronOutput;
		neuron->setGradient( delta * neuron->activateDerivativeFunction(neuronOutput));
	}

	/// calculate the hidden layer gradients
	for(unsigned layer = m_layers.size()-2; layer > 0; --layer)
	{
		Layer& hiddenLayer = m_layers[layer];
		Layer& nextLayer = m_layers[layer+1];
		for(unsigned neuronIdx = 0; neuronIdx < hiddenLayer.neurons.size(); ++neuronIdx)
		{
			Neuron* neuron = hiddenLayer.neurons[neuronIdx];
			double neuronOutput = neuron->getOutput();
			double sum = 0.0;

			for(unsigned i = 0; i < nextLayer.neurons.size(); ++i)
			{
				sum += nextLayer.neurons[i]->getConnectionErrorWith(neuron);
			}
			neuron->setGradient(sum * neuron->activateDerivativeFunction(neuronOutput));
		}

		// calculate layer's bias gradient
		double sum = 0.0;
		for(unsigned i = 0; i < nextLayer.neurons.size(); ++i)
		{
			sum += nextLayer.neurons[i]->getConnectionErrorWith(hiddenLayer.bias);
		}
		hiddenLayer.bias->setGradient(sum * hiddenLayer.bias->activateDerivativeFunction(hiddenLayer.bias->getOutput()));
	}

	/// update connection weights
	for(unsigned layer = m_layers.size()-1; layer > 0; --layer)
	{
		Layer& prevLayer = m_layers[layer-1];
		for(unsigned neuronIdx = 0; neuronIdx < m_layers[layer].neurons.size() ; ++neuronIdx)
		{
			Neuron* neuron = m_layers[layer].neurons[neuronIdx];
			for(unsigned i = 0; i < prevLayer.neurons.size(); ++i )
			{
				Neuron* prevNeuron = prevLayer.neurons[i];

				double deltaWeight = neuron->getConnectionDeltaWeightWith(prevNeuron);

				double newDeltaWeight = m_eta * prevNeuron->getOutput() * neuron->getGradient()  +  // input from prevNeuron
										m_alpha * deltaWeight;  // a fraction of the previous delta weight

				neuron->setConnectionDeltaWeightWith(prevNeuron, newDeltaWeight);
				neuron->setConnectionWeightWith(prevNeuron, neuron->getConnectionWeightWith(prevNeuron)+newDeltaWeight);
			}

			//update bias of previous layer
			double deltaWeight = neuron->getConnectionDeltaWeightWith(prevLayer.bias);

			double newDeltaWeight = m_eta * prevLayer.bias->getOutput() * neuron->getGradient() +  // input from prevNeuron
									m_alpha * deltaWeight;  // a fraction of the previous delta weight

			neuron->setConnectionDeltaWeightWith(prevLayer.bias, newDeltaWeight);
			neuron->setConnectionWeightWith(prevLayer.bias, neuron->getConnectionWeightWith(prevLayer.bias)+newDeltaWeight);
		}
	}
}
void FwdNet::getOutputs(OutputData& outputs)
{
	outputs.clear();
	Layer& outputLayer = m_layers.back();

	for(unsigned outputNeuronIdx = 0; outputNeuronIdx < outputLayer.neurons.size(); ++outputNeuronIdx)
	{
		outputs.push_back(outputLayer.neurons[outputNeuronIdx]->calculateOutput());
		INFO(" -->Output["<<outputNeuronIdx<<"] = "<< outputLayer.neurons[outputNeuronIdx]->getOutput());
	}
}

double FwdNet::getError() const
{
	return m_error;
}

void FwdNet::dump()
{

	for(unsigned layer = 0; layer < m_layers.size(); ++layer)
	{
		std::cout<<"FwdNet: Layers "<< layer+1 << " / " << m_layers.size()<<"\n";
		m_layers[layer].dump();
	}
	std::cout<<"FwdNet::dump() ended\n";
}
double FwdNet::calculateRootMeanSquareErr(const TargetData& targets)
{
	assert(targets.size() == m_layers.back().neurons.size());
	Layer& outputLayer = m_layers.back();
	m_error = 0;
	/// calculate the error; use root mean square
	for(unsigned neuronIdx = 0; neuronIdx < outputLayer.neurons.size(); ++neuronIdx)
	{
		double delta = targets[neuronIdx] - outputLayer.neurons[neuronIdx]->calculateOutput();
		m_error += delta * delta;
	}
	m_error /= outputLayer.neurons.size();
	m_error = sqrt(m_error);  // root mean square

	DEBUG(" -->RMSE = "<<m_error);
	return m_error;
}

} /* namespace ann */
