#include "landingMark.hpp"

using namespace cv;
using namespace std;


size_t countoursSize;
Mat pointsf;
RotatedRect box;


// Arrays utilizados no inRange
int ARR_MAXBLUE[3]   = {MAXBLUE, MAXSATBLUE, MAXVALBLUE};
int ARR_MINBLUE[3]   = {MINBLUE, MINSATBLUE, MINVALBLUE};

int ARR_MAXYELLOW[3] = {MAXYELLOW, MAXSATYELLOW, MAXVALYELLOW};
int ARR_MINYELLOW[3] = {MINYELLOW, MINSATYELLOW, MINVALYELLOW};

int ARR_MAXBLACK[3] = {MAXBLACK, MAXSATBLACK, MAXVALBLACK};
int ARR_MINBLACK[3] = {MINBLACK, MINSATBLACK, MINVALBLACK};


bool isSquare(Rect rectangle)
{
	if (rectangle.x >= rectangle.y)
	{
		if ( (rectangle.x - rectangle.y) <= 0.2 * rectangle.x )
			return true;
		else
			return false;
	}
	else
	{
		if ( (rectangle.y - rectangle.x) <= 0.2 * rectangle.y )
			return true;
		else
			return false;
	}
}



LandingMark::LandingMark()
{
	morph_kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

	success = false;
	fstTime = true;
}


void LandingMark::camParam(Mat img)
{
	rows = img.rows;
	cols = img.cols;

	centerX = img.size().width/2;
	centerY = img.size().height/2;
}


void LandingMark::setImage(Mat img)
{
	resize(img, img, Size(img.cols*0.8, img.rows*0.8));

	//GaussianBlur(img, img, Size(GAUSSIANFILTER, GAUSSIANFILTER), 0);

	if (fstTime)
	{
		LandingMark::camParam(img);
		fstTime = false;
	}

	main_img_C3 = img;
}


Mat LandingMark::imfill(Mat img)
{
	morphologyEx(img, img, MORPH_CLOSE, morph_kernel, Point(-1, -1), 3);

	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> hull( contours.size() );

	for( size_t i = 0; i < contours.size(); i++ )
	{
		convexHull( contours[i], hull[i] );
	}

	if (hull.size() == 1)
	{
		drawContours(img, hull, 0, 255, -1);
	}
	else if (hull.size() > 1)
	{
		float biggestArea = 0;
		vector<Point> biggestContour;

		for ( size_t i = 0; i < hull.size(); i++ )
		{
			float area = contourArea(hull[i]);

			if (area > biggestArea)
			{
				biggestArea = area;
				biggestContour = hull[i];
			}
		}
		vector<vector<Point>> bigContours;
		bigContours.push_back(biggestContour);
		drawContours(img, bigContours, 0, 255, -1);
	}

	return img;
}

Mat LandingMark::imlimiares(Mat hsv, int hsvMin[3], int hsvMax[3])
{
	Mat hsvtresh;

	inRange(hsv, Scalar(hsvMin[0], hsvMin[1], hsvMin[2]), Scalar(hsvMax[0], hsvMax[1], hsvMax[2]), hsvtresh);

	hsvtresh = imfill(hsvtresh);

	return hsvtresh;
}


void LandingMark::processImage()
{
	Mat img_aux;

	cvtColor(main_img_C3, main_imgHSV_C3, COLOR_BGR2HSV);
	//cvtColor(main_img_C3, img_aux, COLOR_BGR2Lab);

	//extractChannel(img_aux, img_lab_can1_C1, 0);
	//extractChannel(main_imgHSV_C3, img_hsv_can3_C1, 2);
	
	//imshow("lab", img_lab_can1_C1);
	//imshow("hsv", img_hsv_can3_C1);

	Mat hsv, output;

	// Pega a area azul
	img_blue_C1 = imlimiares(main_imgHSV_C3, ARR_MINBLUE, ARR_MAXBLUE);
	bitwise_and(main_imgHSV_C3, main_imgHSV_C3, hsv, img_blue_C1);

	// Pega a area amarela
	img_yellow_C1 = imlimiares(main_imgHSV_C3, ARR_MINYELLOW, ARR_MAXYELLOW);
	bitwise_and(hsv, hsv, output, img_yellow_C1);

	// Pega apenas a area do mark
	bitwise_and(img_blue_C1, img_yellow_C1, img_final_C1);

	
	//bitwise_and(img_hsv_can3_C1, img_hsv_can3_C1, bitwise_hsv, img_final_C1);
	//bitwise_and(img_lab_can1_C1, img_lab_can1_C1, bitwise_lab, img_final_C1);

	//imshow("lab", bitwise_lab);
	//imshow("hsv", bitwise_hsv);

	// Mat marc_lab = imlimiares(bitwise_lab, ARR_MINBLACK, ARR_MAXBLACK);
	// Mat marc_hsv = imlimiares(bitwise_hsv, ARR_MINBLACK, ARR_MAXBLACK);

	// imshow("marc_hsv", marc_hsv);
	// imshow("marc_lab", marc_lab);


		// Parte utilizada no detection.cpp
		/*// Pega apenas a base final com cores para detectar o marcador
		bitwise_and(this->mainImagem_C3, this->mainImagem_C3, bitwise_base_final, this->image_final_C1);

		this->image_base_C3 = bitwise_base_final;

		Mat img_temp;

		inRange(bitwise_base_final, Scalar(ARR_MINWHITE[0], ARR_MINWHITE[1], ARR_MINWHITE[2]), Scalar(ARR_MAXWHITE[0], ARR_MAXWHITE[1], ARR_MAXWHITE[2]), img_temp);
		morphologyEx(img_temp, img_temp, MORPH_CLOSE, temp_kernel);

		this->image_marcador_final_C1 = img_temp;
		*/

#if DEBBUGCOLOR
	imshow("Yellow", img_yellow_C1);
	imshow("Blue", img_blue_C1);
	imshow("Final_bitwise", img_final_C1);
#endif
}


bool LandingMark::findSquare()
{
	this->processImage();

	findContours(img_final_C1, this->contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	bool succ = false;

	RotatedRect rotRect;

	for (int i=0; i<this->contours.size(); i++)
	{
		Rect currentRect = boundingRect( this->contours[i] );
		rotRect = minAreaRect( this->contours[i] );

		this->mark = currentRect;
		this->mark_rot = rotRect;
		succ = true;
	}

	return succ;
}


void LandingMark::drawSquare()
{
	//rectangle(main_img_C3, mark, COLOR_RED, 2);

	    // We take the edges that OpenCV calculated for us
	Point2f vertices2f[4];
    mark_rot.points(vertices2f);

    // Convert them so we can use them in a fillConvexPoly
    Point vertices[4];    
    for(int i = 0; i < 4; ++i){
        vertices[i] = vertices2f[i];
    }

    // Now we can fill the rotated rectangle with our specified color
    cv::fillConvexPoly(main_img_C3, vertices, 4, COLOR_RED);
}


Mat LandingMark::rotatedToImage()
{
	// Encontra o RotatedRect
	this->findSquare();

	Mat rotated;
	float angle = mark_rot.angle;

	if (mark_rot.angle < -45.) {
		angle += 90.0;
	}

	Mat M = getRotationMatrix2D(mark_rot.center, angle, 1.0);

	//cout << mark_rot.angle << "\n";

	warpAffine(main_img_C3, rotated, M, main_img_C3.size());

	return rotated;
	// if (mark_rot.angle < -45.) {
	// 	angle += 90.0;
	// 	swap(rect_size.width, rect_size.height);
	// }

	//M = getRotationMatrix2D(mark_rot.center, angle, 1.0);

	//warpAffine(main_imgHSV_C3, rotated, M, main_imgHSV_C3.size(), INTER_CUBIC);

	// crop the resulting image
	//getRectSubPix(rotated, rect_size, mark_rot.center, cropped);

	//imshow("cropped", cropped);
	//imshow("rotated", rotated);

}



void LandingMark::printDistance()
{
	cout << "( " << (mark.x + (mark.width/2)) - centerX << ", " << (mark.y + (mark.height/2)) - centerY << " )\n";
}


void LandingMark::show()
{
	imshow("Main_image", main_img_C3);
}