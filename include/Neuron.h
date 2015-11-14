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
#include <map>

#include <Observer.h>

namespace ann {

/** Neuron class can be observable by others connections for being
 * informed when output has changed.
 */
class Neuron : public Observable
{
public:
    /** Connection keeps informations which can be used by this neuron for
     * calculating its output.
     */
	class Connection : public Observer
    {
    public:
    	virtual void processNotifcation();

    	double getValue() { return m_value;}

    	double getWeight(){ return m_weight;}
    	void   updateWeight(double newWeight) {m_weight = newWeight; }

    	double getDeltaWeight() const { return m_deltaWeight; }
    	void   updateDeltaWeight(double value) { m_deltaWeight = value;}

    	friend class Neuron;
    	friend std::ostream& operator<<(std::ostream& output, const ann::Neuron::Connection& obj);
    private:
    	Connection(Neuron& neuronObserved);
    	Neuron* m_neuronObserved;
        double m_value;			/// value is the input of this connection
        double m_weight;		/// weight is the binding power of connection
        double m_deltaWeight;  	/// deltaWeight is the difference between actual and last value
    };

    Neuron();

    void    	connectTo(Neuron* neuron);
    void    	disconnectFrom(Neuron* neuron);
    Connection* getConnectionWith(Neuron*);

    double activate(double input);
    double activateDerivative(double);
    double addConnectionInputs();

    double  getOutput() const { return m_output; };
    void    setOutput(double value) { m_output = value;}

    double	getGradient()const { return m_gradient; }
    void    setGradient(double gradient){m_gradient=gradient;}

    friend std::ostream& operator<<(std::ostream& output, const ann::Neuron& obj);

private:
	// non-copyable
    Neuron(const Neuron&);
    Neuron& operator=(const Neuron&);

private:
    double                      	m_output;
    double 							m_gradient;
    std::map<Neuron*,Connection*>   m_inputConnections;
};

} /* namespace ann */

#endif /* NEURON_H_ */
