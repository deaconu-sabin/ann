/*
 * Log.h
 *
 *  Created on: Aug 30, 2015
 *      Author: sabin
 */

#ifndef LOG_H_
#define LOG_H_


#include <iostream>
#include <iomanip>

#define ERROR(x) std::cerr<<"ERROR: "<<__PRETTY_FUNCTION__<<" "<<x<<"\n";

#define INFO(x) std::cout<<"INFO: "<<__PRETTY_FUNCTION__<<" "<<x<<"\n";

#define DEBUG(x) // std::cout<<"DEBUG: "<<__PRETTY_FUNCTION__<<" "<<x<<"\n";



#endif /* LOG_H_ */
