#ifndef CAMERA_DETECTION
#define CAMERA_DETECTION

#include "type.h"

class CameraDetector : public Type
{
    public:
    CameraDetector(const int &path);
    void detect(Net &net, vector<string> &classes) override;

    private:
    int cam;

};

#endif