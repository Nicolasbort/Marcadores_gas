#include <opencv2/opencv.hpp>
#include <iostream>

#define MWHITE 55 //140
#define mWHITE 0    //0

#define MSAT 40    //25
#define mSAT 0     //0

#define MVAL 30    //255
#define mVAL 0    //220

using namespace cv;

void invert_color(Mat);

class ROI
{
public:

    Mat image;
    Mat editable_image, clean_img;
    std::vector<Rect> numbers;
    RotatedRect biggest_rect;

    ROI();
    ROI(Mat);
    ROI(Mat, Rect);
    ROI(Mat, RotatedRect);

    void set(Mat);
    void set(Mat, RotatedRect);
    void show(const char*);
    void show_editable();
    void show_clean();
    void invertColor();
    void drawRotated(Mat);
    void rotateImage(double);
    void rotatedToImage(Mat);
    void resize(int, int);
    void fill(Scalar);
    void clean_image();
    void getRectNumbersDynamic(Mat);
    void getRectNumbersStatic(Mat);
    void improve_image();

    int area();
    
    bool found(Mat);

    
    Mat getImage();

private:

    std::vector<std::vector<Point>> contours;
    Mat kernel;

    Point2f* getCorners(RotatedRect);
    bool setContours();
};