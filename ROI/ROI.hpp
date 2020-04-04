#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

#define MWHITE 180 //140
#define mWHITE 5    //0

#define MSAT 55    //25
#define mSAT 0     //0

#define MVAL 255    //255
#define mVAL 210    //220

using namespace cv;

class ROI
{
public:

    Mat mainROI;
    std::vector<Rect> posNumbers;

    ROI(Mat);
    ROI(Mat, Rect);
    void set(Mat);
    void show();
    int area();
    void findPosNumbers();
    void showAllRects();

private:

    std::vector<std::vector<Point>> contours;
    Mat kernel;

    bool setContours();
};