#ifndef TYPE_H
#define TYPE_H


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <sstream>



class Type
{
    public:
    virtual void detect(cv::dnn::Net& net, std::vector<std::string>& classes) = 0; //pure virtual function

    virtual ~Type() = default; // virtual destructor

    protected:
    std::string windowName;
};

#endif