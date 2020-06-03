#include "ROI.hpp"

#define KERNELSIZE 1


class Line
{
public:
  
  float lenght; 
  
  Line(Point2f first, Point2f second)
  {
    lenght = hypot ( second.x - first.x, second.y - first.y );
  }
};



float rotated_area(RotatedRect rect)
{
  Point2f pts[4];
  float area;

  rect.points( pts );

  Line vertical(pts[0], pts[1]);
  Line horizontal(pts[1], pts[2]);

  area = vertical.lenght * horizontal.lenght;

  return area;
}


float razao(RotatedRect rect)
{
    Point2f pts[4];

    rect.points(pts);

    Line horizontal(pts[0], pts[1]);
    Line vertical(pts[1], pts[2]);

    return float(horizontal.lenght / vertical.lenght);
}



ROI::ROI()
{
    ROI::kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

}

ROI::ROI(Mat img)
{
    ROI::image = img;

    ROI::kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);
}


ROI::ROI(Mat img, Rect rect)
{
    ROI::kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

    ROI::image = img(rect);
}

ROI::ROI(Mat img, RotatedRect rectROI)
{
    ROI::kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

    Mat M, rotated, cropped;
    float angle = rectROI.angle;
    Size size = rectROI.size;

    if (rectROI.angle < -45.) {
        angle += 90.0;
        swap(size.width, size.height);
    }

    M = getRotationMatrix2D(rectROI.center, angle, 1.0);

    warpAffine(img, rotated, M, img.size(), INTER_CUBIC);

    getRectSubPix(rotated, size, rectROI.center, cropped);

    ROI::image = cropped;
}

void ROI::set(Mat img)
{
    ROI::image = img;
}


void ROI::set(Mat img, RotatedRect rectROI)
{
    ROI::kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

    Mat M, rotated, cropped;
    float angle = rectROI.angle;
    Size size = rectROI.size;

    if (rectROI.angle < -45.) {
        angle += 90.0;
        swap(size.width, size.height);
    }

    M = getRotationMatrix2D(rectROI.center, angle, 1.0);

    warpAffine(img, rotated, M, img.size(), INTER_CUBIC);

    getRectSubPix(rotated, size, rectROI.center, cropped);

    ROI::image = cropped;
}

Mat ROI::getImage()
{
    return this->image;
}

void ROI::rotatedToImage(Mat img)
{

	Mat rotated, cropped;
	float angle = this->biggest_rect.angle;
    Size rect_size = this->biggest_rect.size;

	if (biggest_rect.angle < -45.) {
		angle += 90.0;
        swap(rect_size.width, rect_size.height);
	}

	Mat M = getRotationMatrix2D(biggest_rect.center, angle, 1.0);

    // Rotaciona a imagem
	warpAffine(img, rotated, M, img.size());

	// Isola o rotatedrect para dentro do cropped
	getRectSubPix(rotated, rect_size, biggest_rect.center, cropped);


    this->image = cropped;
}


void ROI::drawRotated(Mat img)
{
    Point2f pts[4];

    biggest_rect.points(pts);

    std::vector<Point> ptss(4);

    for (int i=0;i<4; i++)
    {
        ptss[i] = pts[i];
    }


    fillConvexPoly(img, ptss, Scalar(255, 0, 0));
}


bool ROI::found(Mat img)
{
    RotatedRect current;
    float biggest_area = 0, current_area;
    bool found = false;

    std::vector<std::vector<Point>> contours;
    findContours(img, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    if (contours.size() >= 1)
    {
        for (int i=0; i<contours.size(); i++)
        {
            current = minAreaRect( contours[i] );
        
            current_area = rotated_area(current);

            if ( current_area < 25 )
                continue;

            //std::cout << currentArea << std::endl;

            if (current_area > biggest_area)
            {
                this->biggest_rect = current;
                biggest_area = current_area;
            } 
        }
    }

    if (biggest_area == 0)
    {
        return false;
    }
    else if (razao(biggest_rect) <= 1.5)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void ROI::show(const char* title)
{
    imshow(title, ROI::image);
}

int ROI::area()
{
    return ROI::image.rows * ROI::image.cols;
}

void ROI::findPossibleNumbers()
{
    //morphologyEx(image, image, MORPH_CLOSE, kernel, Point(-1, -1), 1);

    if (image.channels() > 1)
        cvtColor(image, image, COLOR_BGR2GRAY);

    threshold(image, image, 90, 255, THRESH_BINARY);

    if(ROI::setContours())
    {
        for (int i=0; i<ROI::contours.size(); i++)
        {
            Rect current = boundingRect(ROI::contours[i]);

            if (current.area() > ROI::area()*0.008)
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


void ROI::fill(Scalar color)
{
    if (this->image.empty())
        return;
    this->image.setTo(color);
}

void ROI::resize(int width, int height)
{
    cv::resize(this->image, this->image, Size(width, height));
}

void ROI::showPossibleNumbers()
{
    ROI::findPossibleNumbers();

    for (int i=0; i<ROI::posNumbers.size(); i++)
    {
        rectangle(ROI::image, ROI::posNumbers[i], Scalar(255, 255, 0));
    }
}

void ROI::invertColor()
{
    Mat output;

    bitwise_not(ROI::image, output);

    ROI::image = output;
}

bool ROI::setContours()
{   
    findContours(ROI::image, ROI::contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    if (ROI::contours.size() == 0)
        return 0;
    else 
        return 1;
}