/**
 * /NeuralNetwork/src/Observer.cpp
 *
 *  Created on: Nov 13, 2015
 *      Author: Sabin Deaconu
 */

#include <Observer.h>

Observer::Observer(Observable& observable)
{
	observable.registerObserver(this);
}


void Observable::registerObserver(Observer* observer)
{
	m_observers.push_front(observer);
}
void Observable::notifyObservers()
{
	std::list<Observer*>::iterator it = m_observers.begin();

	while(it != m_observers.end())
	{
		(*it)->processNotifcation();
		++it;
	}
}
