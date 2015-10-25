/*
 * Neuron.h
 *
 *  Created on: Aug 30, 2015
 *      Author: sabin
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <vector>
#include <ostream>
#include <list>

namespace ann {

struct Connection;
class Neuron
{
public:
	typedef double (*pfnActivate)(double);
	typedef double (*pfnDerivative)(double);

    Neuron();

    void 	setOutput(double);
    double  getOutput() const;
    void    connectTo(Neuron* neuron);
    void    disconnectFrom(Neuron* neuron);

    double 	getConnectionErrorWith(Neuron* );
    double  getConnectionWeightWith(Neuron*);
    double  getConnectionDeltaWeightWith(Neuron*);
    void    setConnectionWeightWith(Neuron*, double weight);
    void    setConnectionDeltaWeightWith(Neuron*, double deltaweight);

    double calculateOutput();

    void    setGradient(double gradient){m_Gradient=gradient;}
    double	getGradient(){ return m_Gradient;}

public:
	pfnActivate 	activateFunction;
	pfnDerivative 	activateDerivativeFunction;

private:
    struct Connection
    {
    	Connection(Neuron* neuron);
        const Neuron* neuron;
        double weight;
        double deltaWeight;
    };
    Neuron(const Neuron&);
    Neuron& operator=(const Neuron&);

    double sumInputs();
    Connection* getConnectionWith(Neuron*);

private:
    double                      m_output;
    std::list<Connection>       m_inputConnections;
    double m_Gradient;
    //std::list<Neuron*>          m_neuronsConnectedToMe;

public:
    friend std::ostream& operator<<(std::ostream& output, const ann::Neuron& obj);
    friend std::ostream& operator<<(std::ostream& output, const ann::Neuron::Connection& obj);
};

} /* namespace ann */

#endif /* NEURON_H_ */
