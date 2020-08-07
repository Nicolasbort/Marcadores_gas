#include "ROI.hpp"

#define KERNELSIZE 1

#define DEBUG(x) std::cout << x << "\n"


class Line
{
public:
  
  float lenght; 
  
  Line(Point2f first, Point2f second)
  {
    lenght = hypot ( second.x - first.x, second.y - first.y );
  }
};


void invert_color(Mat img)
{
    bitwise_not(img, img);
}


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
    ROI::image = img.clone();

    ROI::editable_image = image.clone();

    ROI::kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);
}


ROI::ROI(Mat img, Rect rect)
{
    ROI::kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

    ROI::image = img(rect);

    ROI::editable_image = image.clone();
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

    ROI::image = cropped.clone();

    ROI::editable_image = image.clone();
}


void ROI::set(Mat img)
{
    ROI::image = img.clone();

    ROI::editable_image = image.clone();
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

    ROI::image = cropped.clone();

    ROI::editable_image = image.clone();
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

	// if (biggest_rect.angle < -45.) {
	// 	angle += 90.0;
    //     swap(rect_size.width, rect_size.height);
	// }

	Mat M = getRotationMatrix2D(biggest_rect.center, angle, 1.0);

    // Rotaciona a imagem
	warpAffine(img, rotated, M, img.size());

	// Isola o rotatedrect para dentro do cropped
	getRectSubPix(rotated, rect_size, biggest_rect.center, cropped);


    this->image = cropped.clone();

    this->editable_image = this->image.clone();
}


void ROI::clean_image()
{
    if (this->editable_image.channels() != 1)
        cvtColor(this->editable_image, this->editable_image, CV_BGR2GRAY);

    threshold(this->editable_image, this->editable_image, 70, 255, THRESH_BINARY);

    this->clean_img = this->editable_image.clone();
}


void ROI::getRectNumbersDynamic(Mat img)
{
    //DEBUG("Entrou");
    int element_lenght = this->editable_image.cols * 0.09;
    int count_rect = 0;

    Mat kernel = Mat::ones(Size(7, 7), CV_8U);
    Mat element = getStructuringElement(CV_SHAPE_RECT, Size(element_lenght, 1));


    morphologyEx(this->editable_image, this->editable_image, MORPH_ERODE, kernel);
    morphologyEx(this->editable_image, this->editable_image, MORPH_GRADIENT, element);


    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    Mat mask = Mat::zeros(this->editable_image.size(), CV_8UC1);

    findContours(this->editable_image, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // filter contours
    if (hierarchy.size() > 0)
    {
        for(int index = 0; index >= 0; index = hierarchy[index][0])
        {
            Rect rect = boundingRect(contours[index]);
            Mat maskROI(mask, rect);
            maskROI = Scalar(0, 0, 0);

            drawContours(mask, contours, index, Scalar(255, 255, 255), CV_FILLED);

            // ratio of non-zero pixels in the filled region
            double r = (double)countNonZero(maskROI)/(rect.width*rect.height);

            if (r > .45 /* assume at least 45% of the area is filled if it contains text */
                && 
                rect.area() > (this->editable_image.rows*this->editable_image.cols) * 0.05 && rect.width/float(rect.height) >= 1.3 /* constraints on region size */
                /* these two conditions alone are not very robust. better to use something 
                like the number of significant peaks in a horizontal projection as a third condition */
                )
            {
                count_rect++;
                rectangle(img, rect, Scalar(0, 255, 0), 2);
            }
        }
    }
}


void ROI::getRectNumbersStatic(Mat img)
{
    int TOP_LEFT_CORNER_X = img.cols * 0.06;
    int TOP_LEFT_CORNER_Y_1 = img.rows * 0.03;
    int TOP_LEFT_CORNER_Y_2 = img.rows * 0.53;

    int BOT_RIGHT_CORNER_X = img.cols * 0.63;
    int BOT_RIGHT_CORNER_Y_1 = img.rows * 0.48;
    int BOT_RIGHT_CORNER_Y_2 = img.rows * 0.95;

    // Rect da porcentagem
    int P_TOP_LEFT_CORNER_X = img.cols * 0.64;
    int P_TOP_LEFT_CORNER_Y = img.rows * 0.03;

    int P_BOT_RIGHT_CORNER_X = img.cols * 0.95;
    int P_BOT_RIGHT_CORNER_Y = img.rows * 0.48;

    Rect PERCENTE( Point(P_TOP_LEFT_CORNER_X, P_TOP_LEFT_CORNER_Y), Point(P_BOT_RIGHT_CORNER_X, P_BOT_RIGHT_CORNER_Y) );
    Rect TOP( Point(TOP_LEFT_CORNER_X, TOP_LEFT_CORNER_Y_1), Point(BOT_RIGHT_CORNER_X, BOT_RIGHT_CORNER_Y_1) );
    Rect BOT( Point(TOP_LEFT_CORNER_X, TOP_LEFT_CORNER_Y_2), Point(BOT_RIGHT_CORNER_X, BOT_RIGHT_CORNER_Y_2) );

    numbers.push_back(TOP);
    numbers.push_back(BOT);
    numbers.push_back(PERCENTE);

    // rectangle(img, TOP, Scalar(0, 255, 0), 2);
    // rectangle(img, BOT, Scalar(0, 255, 0), 2); 
    // rectangle(img, PERCENTE, Scalar(0, 255, 0), 2);
}


void ROI::improve_image()
{
    detailEnhance(this->image, this->image);
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

void ROI::rotateImage(double angle)
{
    Point center = Point(ROI::image.rows/2, ROI::image.cols/2);
    Mat rotated;

    Mat M = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(ROI::image, rotated, M, ROI::image.size(), INTER_LINEAR);

    ROI::image = rotated.clone();
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

void ROI::show_editable()
{
    imshow("Editable", ROI::editable_image);
}

void ROI::show_clean()
{
    imshow("Clean", this->clean_img);
}

int ROI::area()
{
    return ROI::image.rows * ROI::image.cols;
}

void ROI::fill(Scalar color)
{
    if (this->clean_img.empty())
        return;
    this->clean_img.setTo(color);
}

void ROI::resize(int width, int height)
{
    cv::resize(this->image, this->image, Size(width, height));

    ROI::editable_image = ROI::image.clone();
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