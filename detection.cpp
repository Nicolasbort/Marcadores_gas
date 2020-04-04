#include "ROI/ROI.hpp"

using namespace cv;


ROI findROI(Mat img)
{
  Rect biggest, current;
  int biggestArea = 0;

  std::vector<std::vector<Point>> contours;
  findContours(img, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

  if (contours.size() >= 1)
  {
    for (int i=0; i<contours.size(); i++)
    {
      current = boundingRect(contours[i]);

      if (current.area() > biggestArea)
      {
        biggest = current;
        biggestArea = current.area();
      }
    }

    ROI roi(img, biggest);

    return roi;
  }
  else
  {
    ROI roi(img);

    return roi;
  }
  
}


int main()
{

  std::vector<std::vector<Point>> contours;

  Mat kernel = Mat::ones(Size(3, 3), CV_8U);

  bool VIDEO = true;

  if (VIDEO)
  {

    VideoCapture cap("sensorbom.mp4");

      if (!cap.isOpened())
      {
        std::cout << "Erro ao abrir o video" << std::endl;
        return -1;
      }

    Mat frame, frame_C1, frameHSV;

    while(true)
    {
      cap >> frame;

      if (frame.empty())
        break;

      cvtColor(frame, frameHSV, COLOR_BGR2HSV);
      inRange(frameHSV, Scalar(mWHITE, mSAT, mVAL), Scalar(MWHITE, MSAT, MVAL), frame_C1);
      morphologyEx(frame_C1, frame_C1, MORPH_CLOSE, kernel);

      ROI roi = findROI(frame_C1);

      roi.findPosNumbers();
      roi.showAllRects();
      roi.show();

      imshow("frame", frame);

      int key = waitKey(50);

      if (key == 32)
      {
        imwrite("frame.jpeg", frame);
      }
    } 
  } 
  else
  {
    Mat img = imread("ftsensor1.jpeg"), img_output;

    cvtColor(img, img, COLOR_RGB2HSV);
    inRange(img, Scalar(mWHITE, mSAT, mVAL), Scalar(MWHITE, MSAT, MVAL), img_output);
    morphologyEx(img_output, img_output, MORPH_CLOSE, kernel);

    ROI roi = findROI(img_output);

    roi.findPosNumbers();
    roi.showAllRects();
    roi.show();

    imshow("frame", frame);

    waitKey();
  }
}