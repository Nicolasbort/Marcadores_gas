#include "include/defines.hpp"

#define DEBUGMODE true
#define DEBBUGCOLOR false



class LandingMark
{
public:

    //// Variaveis ////
    cv::Mat main_img_C3, main_imgHSV_C3, img_blue_C1, img_yellow_C1, img_final_C1;
    cv::Mat img_lab_can1_C1, img_hsv_can3_C1;
    cv::Mat morph_kernel;

    int rows, cols;
    int centerX, centerY;

    bool success, fstTime;

	cv::Rect mark;
    cv::RotatedRect mark_rot;

    std::vector< std::vector<cv::Point>> contours;


    //// Funcoes /////
    LandingMark();

    void camParam(cv::Mat);
    void setImage(cv::Mat);
    void processImage();
    void drawSquare();
    void printDistance();
    void show();
    
    cv::Mat imfill(cv::Mat);
    cv::Mat imlimiares(cv::Mat, int[], int[]);
    cv::Mat rotatedToImage();

    bool findSquare();
};