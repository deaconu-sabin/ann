/*
 * Neuron.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: sabin
 */

#include <cstdlib>
#include <cassert>
#include <cmath>

#include <Neuron.h>
#include <Log.h>

namespace ann {

static double sigmoidFunction(double value);

Neuron::Neuron():
		m_output(1.0),
		m_gradient(0.0)
{
}

void Neuron::connectTo(Neuron* neuron)
{
	assert(NULL != neuron);
	if(m_inputConnections.find(neuron) != m_inputConnections.end())
	{
		delete m_inputConnections[neuron];
	}
	m_inputConnections[neuron] = new Connection(*neuron);
}

void Neuron::disconnectFrom(Neuron* neuron)
{
	assert(NULL != neuron);
	std::map<Neuron*,Connection*>::iterator it = m_inputConnections.find(neuron);
	if(it != m_inputConnections.end())
	{
		m_inputConnections.erase(it);
	}
}

Neuron::Connection* Neuron::getConnectionWith(Neuron* neuron)
{
	return m_inputConnections[neuron];
}

double Neuron::activate(double input)
{
	m_output = sigmoidFunction(input);
    return m_output;
}

double Neuron::activateDerivative(double value)
{
	double functionValue = sigmoidFunction(value);
	return (functionValue * ( 1 - functionValue));
}

double Neuron::addConnectionInputs()
{
	double sum = 0;

	for(std::map<Neuron*,Connection*>::iterator connectionIt = m_inputConnections.begin();
		connectionIt != m_inputConnections.end();
		++connectionIt)
	{
		sum += connectionIt->second->m_value *
			   connectionIt->second->m_weight;
	}

    return sum;
}

std::ostream& operator<<(std::ostream& output, const ann::Neuron& obj)
{
	output<<"output="<<obj.getOutput()<<"; gradient="<<obj.m_gradient<<";\n";
	for(std::map<Neuron*, Neuron::Connection*>::const_iterator connectionIt = obj.m_inputConnections.begin();
		connectionIt != obj.m_inputConnections.end();
		++connectionIt)
	{
		output<<"    connection["<<connectionIt->first
				<< "] = " << *(connectionIt->second) <<"\n";
	}
	return output;
}

double sigmoidFunction(double value)
{
	return ( 1.0 /
		   ( 1+ std::exp(-value)) );
}

} /* namespace ann */


