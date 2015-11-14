/*
 * Connection.cpp
 *
 *  Created on: Nov 1, 2015
 *      Author: sabin
 */

#include <Neuron.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
namespace ann
{

Neuron::Connection::Connection(Neuron& neuronObserved):Observer(neuronObserved),
		m_neuronObserved(&neuronObserved),
		m_value(0.0),
		m_weight(0.0),
		m_deltaWeight(0.0)
{
	m_weight = rand()/double (RAND_MAX);
}

void Neuron::Connection::processNotifcation()
{
	m_value = m_neuronObserved->getOutput();
}

std::ostream& operator<<(std::ostream& output, const ann::Neuron::Connection& obj)
{
	output	<< obj.m_value
			<<"\\n"
			<<	obj.m_weight
			<<"\\n"
			<<obj.m_deltaWeight;
	return output;
}

};

