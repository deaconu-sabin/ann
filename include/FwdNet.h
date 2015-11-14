/*
 * FwdNet.h
 *
 *  Created on: Aug 30, 2015
 *      Author: sabin
 */

#ifndef FWDNET_H_
#define FWDNET_H_

#include <vector>
#include <utility>

namespace ann {

// forward declaration
class Neuron;

class FwdNet {
public:
	typedef std::vector<unsigned>				Topology;
	typedef std::vector<double> 				InputData;
	typedef std::vector<double> 				OutputData;
	typedef std::vector<double> 				TargetData;
	typedef std::pair< InputData, TargetData > 	TrainingData;
	typedef std::vector<TrainingData> 			TrainingDataSet;

	FwdNet(const Topology &topology);

	void processInputs(const InputData& inputData);
	void doTraining(const TrainingDataSet& trainingDataSet, double acceptedError = 0.20);
	void getOutputs(OutputData& outputs);

	double getError() const;
	void dump();
	void saveGraph(std::string fileName);

protected:
	void feedForward(const InputData& inputs);
	void propagateBack(const TargetData& targets);
	double calculateMeanSquaredErr(const TargetData& targets);

private:
	struct Layer
	{
		Layer(unsigned neuronsNo);
		void dump(void);
		std::vector<Neuron*> neurons;
		Neuron*			    bias;
	};

	std::vector<Layer>	  	m_layers;
	double 					m_error;
	double					m_eta;    /// learning rate
	double					m_alpha;  /// momentum learning rate
	bool 					m_isTrained;
};

} /* namespace ann */

#endif /* FWDNET_H_ */
