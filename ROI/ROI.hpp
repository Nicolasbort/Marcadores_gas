#include <opencv2/opencv.hpp>
#include <iostream>

#define MWHITE 55 //140
#define mWHITE 0    //0

#define MSAT 40    //25
#define mSAT 0     //0

#define MVAL 30    //255
#define mVAL 0    //220

using namespace cv;


class ROI
{
public:

    Mat image;
    std::vector<Rect> posNumbers;
    RotatedRect biggest_rect;

    ROI();
    ROI(Mat);
    ROI(Mat, Rect);
    ROI(Mat, RotatedRect);

    void set(Mat);
    void set(Mat, RotatedRect);
    void show(const char*);
    void showPossibleNumbers();
    void invertColor();
    void drawRotated(Mat);
    void rotatedToImage(Mat);
    void resize(int, int);
    void fill(Scalar);

    int area();
    
    bool found(Mat);

    Mat getImage();

private:

    std::vector<std::vector<Point>> contours;
    Mat kernel;

    Point2f* getCorners(RotatedRect);
    bool setContours();
    void findPossibleNumbers();
};