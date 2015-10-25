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

static double activate(double);
static double activateDerivative(double);

Neuron::Connection::Connection(Neuron* inNeuron):
		neuron(inNeuron),
		weight(0.0),
		deltaWeight(0.0)
{
	weight = rand()/double (RAND_MAX);
}
Neuron::Neuron():
		activateFunction(activate),
		activateDerivativeFunction(activateDerivative),
		m_output(1.0),
		m_Gradient(0.0)
{
}

void Neuron::setOutput(double output)
{
	m_output=output;
}
double Neuron::getOutput() const
{
    return m_output;
}

void Neuron::connectTo(Neuron* neuron)
{
	if(NULL == neuron )
		return;
    m_inputConnections.push_front(Connection(neuron));
    //neuron->m_neuronsConnectedToMe.push_front(this);
}

void Neuron::disconnectFrom(Neuron* neuron)
{
	if(NULL == neuron )
		return;

	//neuron->m_neuronsConnectedToMe.remove(this);

	std::list<Connection>::iterator it = m_inputConnections.begin();
	while(it != m_inputConnections.end() && it->neuron != neuron)
	{
		++it;
	}

	if(it != m_inputConnections.end())
	{
		m_inputConnections.erase(it);
	}
}
double Neuron::getConnectionErrorWith(Neuron* neuron)
{
	Connection* conn = getConnectionWith(neuron);
	if(conn)
	{
		return conn->weight * m_Gradient;
	}
	ERROR(" NOT FOUND");
	return 0.0;
}
double Neuron::getConnectionWeightWith(Neuron* neuron)
{
	Connection* conn = getConnectionWith(neuron);
	if(conn)
	{
		return conn->weight;
	}
	return 0.0;
}
double Neuron::getConnectionDeltaWeightWith(Neuron* neuron)
{
	Connection* conn = getConnectionWith(neuron);
	if(conn)
	{
		return conn->deltaWeight;
	}
	return 0.0;
}
void Neuron::setConnectionWeightWith(Neuron* neuron, double weight)
{
	Connection* conn = getConnectionWith(neuron);
	if(conn)
	{
		conn->weight = weight;
	}
}
void Neuron::setConnectionDeltaWeightWith(Neuron* neuron, double deltaweight)
{
	Connection* conn = getConnectionWith(neuron);
	if(conn)
	{
		 conn->deltaWeight = deltaweight;
	}
}

double Neuron::sumInputs()
{
	double sum = 0.0;
	for(std::list<Connection>::iterator connection = m_inputConnections.begin();
		connection != m_inputConnections.end();
		++connection)
	{
		sum += connection->neuron->getOutput() * connection->weight;
	}
    return sum;
}
Neuron::Connection* Neuron::getConnectionWith(Neuron* neuron)
{
	Connection* conn = NULL;

	std::list<Connection>::iterator it = m_inputConnections.begin();
	while(it != m_inputConnections.end() && it->neuron != neuron)
	{
		++it;
	}
	if(it != m_inputConnections.end())
	{
		conn = &(*it);
	}
	return conn;
}
double Neuron::calculateOutput()
{
	m_output = activateFunction(sumInputs());
    return m_output;
}
std::ostream& operator<<(std::ostream& output, const ann::Neuron& obj)
{
	output<<"output="<<obj.getOutput()<<"; gradient="<<obj.m_Gradient<<";\n";
	for(std::list<Neuron::Connection>::const_iterator connection = obj.m_inputConnections.begin();
		connection != obj.m_inputConnections.end();
		++connection)
	{
		output<<"    connection["<<connection->neuron<<"]: "<< *connection<<"\n";
	}
	return output;
}
std::ostream& operator<<(std::ostream& output, const ann::Neuron::Connection& obj)
{
	output<<"weight="<<obj.weight<<"; deltaWeight="<<obj.deltaWeight<<";";
	return output;
}

double activate(double value)
{
	return std::tanh(value);
}
double activateDerivative(double value)
{
	return (1.0-tanh(value)*tanh(value));
}

} /* namespace ann */


