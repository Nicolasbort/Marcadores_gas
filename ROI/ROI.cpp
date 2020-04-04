#include "ROI.hpp"

ROI::ROI(Mat img)
{
    ROI::mainROI = img;

    ROI::kernel = Mat::ones(3, 3, CV_8U);
}

ROI::ROI(Mat img, Rect rectROI)
{
    ROI::mainROI = img(rectROI);

    ROI::kernel = Mat::ones(3, 3, CV_8U);
}

void ROI::set(Mat img)
{
    ROI::mainROI = img;
}

void ROI::show()
{
    imshow("ROI", ROI::mainROI);
}

int ROI::area()
{
    return ROI::mainROI.rows * ROI::mainROI.cols;
}

void ROI::findPosNumbers()
{
    if(ROI::setContours())
    {
        for (int i=0; i<ROI::contours.size(); i++)
        {
            Rect current = boundingRect(ROI::contours[i]);
            if (current.area() > ROI::area()*0.005)
            {
                ROI::posNumbers.push_back(current);
            }
        }  
    }
    else
    {
        return; 
    } 
}

void ROI::showAllRects()
{
    for (int i=0; i<ROI::posNumbers.size(); i++)
    {
        rectangle(ROI::mainROI, ROI::posNumbers[i], Scalar(255, 255, 0));
    }
}

bool ROI::setContours()
{
    //Mat tempImgHSV = ROI::mainROI.clone(), tempImgC1;

    //cvtColor(tempImgHSV, tempImgHSV, COLOR_BGR2HSV);

    //inRange(tempImgHSV, Scalar(mWHITE, mSAT, mVAL), Scalar(MWHITE, MSAT, MVAL), tempImgC1);

    //morphologyEx(tempImgC1, tempImgC1, MORPH_CLOSE, ROI::kernel);
    
    findContours(ROI::mainROI, ROI::contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    if (ROI::contours.size() == 0)
        return 0;
    else 
        return 1;
}