#ifndef VIDEO_DETECTION_H
#define VIDEO_DETECTION_H

#include "type.h"

class VideoDetector : public Type
{
    public:
    VideoDetector(const std::string &path);
    void detect(cv::dnn::Net &net, std::vector<std::string>& classes) override;

    private:
    std::string videoPath;
};


#endif