/*
 * ObserverPattern.h
 *
 *  Created on: Nov 1, 2015
 *      Author: sabin
 */

#ifndef OBSERVERPATTERN_H_
#define OBSERVERPATTERN_H_

#include <list>

class Observable;

class Observer
{
public:
	Observer(Observable& observable);
	virtual void processNotifcation() = 0;
	virtual ~Observer(){}
};

class Observable
{
public:
	void registerObserver(Observer* observer);
	void notifyObservers();

private:
	std::list<Observer*> m_observers;
};

#endif /* OBSERVERPATTERN_H_ */
