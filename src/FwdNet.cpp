/*
 * FwdNet.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: sabin
 */

#include <cmath>
#include <cassert>
#include <fstream>

#include <FwdNet.h>
#include <Log.h>
#include <Neuron.h>

#define MAX_ITERETION 10000

namespace ann {

FwdNet::Layer::Layer(unsigned neuronsNo):
		bias(NULL)
{
	for(unsigned neuronIdx=0; neuronIdx < neuronsNo; ++neuronIdx)
	{
		neurons.push_back(new Neuron());
		DEBUG("  created Neuron " << neurons.back());
	}
	bias = new Neuron();
	DEBUG("  created Bias " << bias);
}

void FwdNet::Layer::dump()
{
	for(unsigned neuronIdx = 0; neuronIdx < neurons.size(); ++neuronIdx)
	{
		std::cout<<"  Neuron["<<neurons[neuronIdx]<<"]: "<< *(neurons[neuronIdx]);
	}
	// log bias info
	if(bias)
	{
		std::cout<<"  Bias["<<bias<<"]: "<< *bias<<"\n";
	}
}

FwdNet::FwdNet(const Topology &topology):
		m_error(47.0),
		m_eta(0.25),
		m_alpha(0.3),
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
		Layer& prevLayer =  m_layers.back();

		for(unsigned neuronIdx = 0;
			neuronIdx < layer.neurons.size();
			++neuronIdx)
		{
			for(unsigned neuronPrevLayer = 0 ;
				neuronPrevLayer < prevLayer.neurons.size();
				++neuronPrevLayer)
			{
				layer.neurons[neuronIdx]->connectTo(prevLayer.neurons[neuronPrevLayer]);
				INFO(layer.neurons[neuronIdx] << " connect to " << prevLayer.neurons[neuronPrevLayer]);
			}
			if(NULL != prevLayer.bias)
			{
				layer.neurons[neuronIdx]->connectTo(prevLayer.bias);
				INFO(layer.neurons[neuronIdx] << " connect to " << prevLayer.bias);
			}
		}
		if(NULL != prevLayer.bias)
		{
			prevLayer.bias->setOutput(1.0);
			prevLayer.bias->notifyObservers();
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
			if(calculateMeanSquaredErr(targetData) > acceptedError )
			{
				m_isTrained = false;
			}
			if(epochError < m_error)
			{
				epochError = m_error;
			}
			DEBUG("For TrainingData["<<trainingDataIdx<<"]; MS Error = " << epochError);
			propagateBack(targetData);
		}
		INFO("For Epoch "<< std::setw(5)<<std::right<< epochCounter++ <<"  error is "<<  epochError);
	}while( !m_isTrained && epochCounter < MAX_ITERETION );
	INFO("Training finished.");
}
void FwdNet::feedForward(const InputData& inputs)
{
	assert(inputs.size()==m_layers[0].neurons.size());
	/// assign inputs to first layer
	for(unsigned i = 0; i < inputs.size(); ++i)
	{
		m_layers[0].neurons[i]->setOutput(inputs[i]);
		m_layers[0].neurons[i]->notifyObservers();
	}
	if(NULL != m_layers[0].bias)
	{
		m_layers[0].bias->setOutput(1.0);
		m_layers[0].bias->notifyObservers();
	}

	//feed forward. start from second layer (not with the inputLayer)
	for(unsigned layer = 1; layer < m_layers.size(); ++layer)
	{
		for(unsigned neuronIdx = 0; neuronIdx < m_layers[layer].neurons.size(); ++neuronIdx )
		{
			Neuron* currentNeuron = m_layers[layer].neurons[neuronIdx];
			double sum = currentNeuron->addConnectionInputs();
			currentNeuron->activate(sum);
			currentNeuron->notifyObservers();
		}
		if(NULL != m_layers[layer].bias)
		{
			m_layers[layer].bias->setOutput(1.0);
			m_layers[layer].bias->notifyObservers();
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
		double errorDerivative = targets[neuronIdx] - neuronOutput;
		neuron->setGradient( errorDerivative * neuron->activateDerivative(neuron->addConnectionInputs()));
	}

	/// calculate the hidden layer gradients
	for(unsigned layer = m_layers.size()-2; layer > 0; --layer)
	{
		Layer& hiddenLayer = m_layers[layer];
		Layer& nextLayer = m_layers[layer+1];
		for(unsigned neuronIdx = 0; neuronIdx < hiddenLayer.neurons.size(); ++neuronIdx)
		{
			Neuron* neuron = hiddenLayer.neurons[neuronIdx];
			//double neuronOutput = neuron->getOutput();
			double sum = 0.0;

			for(unsigned i = 0; i < nextLayer.neurons.size(); ++i)
			{
				sum += nextLayer.neurons[i]->getConnectionWith(neuron)->getWeight() *
					   nextLayer.neurons[i]->getGradient();
			}
			neuron->setGradient(sum * neuron->activateDerivative(neuron->addConnectionInputs()));
		}
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

				double deltaWeight = neuron->getConnectionWith(prevNeuron)->getDeltaWeight();

				double newDeltaWeight = m_eta * prevNeuron->getOutput() * neuron->getGradient() +  // input from prevNeuron
										m_alpha * deltaWeight;  // a fraction of the previous delta weight

				neuron->getConnectionWith(prevNeuron)->updateDeltaWeight(newDeltaWeight);
				neuron->getConnectionWith(prevNeuron)->updateWeight(
						neuron->getConnectionWith(prevNeuron)->getWeight()+ newDeltaWeight);
			}

			//update bias of previous layer
			//double deltaWeight = neuron->getConnectionDeltaWeightWith(prevLayer.bias);
			if(NULL != prevLayer.bias)
			{
				double deltaWeight = neuron->getConnectionWith(prevLayer.bias)->getDeltaWeight();
				double newDeltaWeight = m_eta * prevLayer.bias->getOutput() * neuron->getGradient() + // input from prevNeuron
									    m_alpha * deltaWeight;  // a fraction of the previous delta weight

				neuron->getConnectionWith(prevLayer.bias)->updateDeltaWeight(newDeltaWeight);
				neuron->getConnectionWith(prevLayer.bias)->updateWeight(
						neuron->getConnectionWith(prevLayer.bias)->getWeight() + newDeltaWeight);
			}
		}
	}
}
void FwdNet::getOutputs(OutputData& outputs)
{
	outputs.clear();
	Layer& outputLayer = m_layers.back();

	for(unsigned outputNeuronIdx = 0; outputNeuronIdx < outputLayer.neurons.size(); ++outputNeuronIdx)
	{
		outputs.push_back(outputLayer.neurons[outputNeuronIdx]->getOutput());
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

void FwdNet::saveGraph(std::string fileName)
{
	std::ofstream outFile(fileName.c_str(), std::ios_base::out | std::ios_base::trunc);

	outFile << "digraph g {\n";
	for(unsigned layerIdx = 1; layerIdx < m_layers.size(); ++layerIdx)
	{
		for(unsigned neuronIdx = 0; neuronIdx < m_layers[layerIdx].neurons.size(); ++neuronIdx)
		{
			for(unsigned prevNeuronIdx = 0; prevNeuronIdx < m_layers[layerIdx-1].neurons.size(); ++prevNeuronIdx)
			{
					//write node
				outFile << std::setprecision(3) << std::setw(5)<< std::setfill(' ')
						<<"\t\""
						<< m_layers[layerIdx-1].neurons[prevNeuronIdx]
						<<"\" [label=\""
						<<m_layers[layerIdx-1].neurons[prevNeuronIdx]
						<<"\\n out="
						<< m_layers[layerIdx-1].neurons[prevNeuronIdx]->getOutput()
						<<"\"];"<<std::endl;
						// write connection
				outFile << "\t\""
						<< m_layers[layerIdx-1].neurons[prevNeuronIdx]
						<<"\" -> \""
						<< m_layers[layerIdx].neurons[neuronIdx]
						<< "\" [label=\""
						<< *(m_layers[layerIdx].neurons[neuronIdx]->getConnectionWith(m_layers[layerIdx-1].neurons[prevNeuronIdx]))
						<<"\"];" << std::endl;
			}
			if( NULL != m_layers[layerIdx-1].bias)
			{
				outFile <<"\t\""
						<< m_layers[layerIdx-1].bias
						<<"\" [label=\"bias\\n out="
						<< m_layers[layerIdx-1].bias->getOutput()
						<<"\"];"
						<< std::endl;

				outFile <<"\t\""
						<< m_layers[layerIdx-1].bias
						<<"\" -> \""
						<< m_layers[layerIdx].neurons[neuronIdx]
						<< "\" [label=\""
						<< *(m_layers[layerIdx].neurons[neuronIdx]->getConnectionWith(m_layers[layerIdx-1].bias))
						<<"\"];" << std::endl;
			}
		}
		outFile <<"\t { rank=same; ";
		for(unsigned prevNeuronIdx = 0; prevNeuronIdx < m_layers[layerIdx-1].neurons.size(); ++prevNeuronIdx)
		{
			outFile << "\""
					<< m_layers[layerIdx-1].neurons[prevNeuronIdx]
			        << "\" ";
		}
		if( NULL != m_layers[layerIdx-1].bias)
		{
			outFile <<"\""
					<<m_layers[layerIdx-1].bias
					<<"\" ";
		}
		outFile	<<"}\n";
	}

	for(unsigned outputNeuronIdx = 0; outputNeuronIdx < m_layers.back().neurons.size(); ++outputNeuronIdx)
	{
			//write node
		outFile << std::setprecision(3) << std::setw(5)
				<<"\t\""
				<< m_layers.back().neurons[outputNeuronIdx]
				<<"\" [label=\""
				<<m_layers.back().neurons[outputNeuronIdx]
				<<"\\n out="
				<< m_layers.back().neurons[outputNeuronIdx]->getOutput()
				<<"\"];"<<std::endl;
	}
	outFile << "}" << std::endl;
	outFile.close();

}


double FwdNet::calculateMeanSquaredErr(const TargetData& targets)
{
	assert(targets.size() == m_layers.back().neurons.size());
	Layer& outputLayer = m_layers.back();
	m_error = 0;
	// calculate the mean square error
	for(unsigned neuronIdx = 0; neuronIdx < outputLayer.neurons.size(); ++neuronIdx)
	{
		double delta = targets[neuronIdx] - outputLayer.neurons[neuronIdx]->getOutput();
		m_error += (delta * delta);
	}
	m_error /= outputLayer.neurons.size();

	DEBUG(" -->MSE = "<<m_error);
	return m_error;
}

} /* namespace ann */
