#include "ROI/ROI.hpp"
#include "landingMark.hpp"
#include <math.h>

using namespace cv;
using namespace std;

#define DEBBUGCOLOR false


// Arrays utilizados no inRange

int ARR_MAXWHITE[3] = {MAXWHITE, MAXSATWHITE, MAXVALWHITE};
int ARR_MINWHITE[3] = {MINWHITE, MINSATWHITE, MINVALWHITE};

int ARR_MAX_C2[3] = {65, 60, 45};
int ARR_MIN_C2[3] = {10, 10, 5};





ROI findROI(Mat img_c1, Mat img_main)
{
  Rect biggest, current;
  float biggestArea = 0, currentArea;

  std::vector<std::vector<Point>> contours;
  findContours(img_c1, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

  if (contours.size() >= 1)
  {
    for (int i=0; i<contours.size(); i++)
    {
      current = boundingRect( contours[i] );
    
      currentArea = current.area();

      //std::cout << currentArea << std::endl;

      if (currentArea > biggestArea)
      {
        biggest = current;
        biggestArea = currentArea;
      } 
      
    }

    if (biggestArea == 0)
    {
      ROI roi(img_main);
      return roi;
    }
    else
    {
      ROI roi(img_main, biggest);
      return roi;
    }
    
  }
  else
  {
    ROI roi(img_main);
    return roi;
  }
}

#define VIDEO true


int main(int argc, char *argv[])
{

  if (argc < 2)
  {
    std::cout << "Nome do arquivo!" << std::endl;
    return -1;
  }


#if VIDEO

  VideoCapture cap(argv[1]);

  if (!cap.isOpened())
  {
    std::cout << "Erro ao abrir o video" << std::endl;
    return -1;
  }

  Mat frame, frame_sensor, frame_yellow, frameHSV;

  LandingMark main_image;

  int contImwrite = 0;

  while(true)
  {
    cap >> frame;

    if (frame.empty())
		  break;

    main_image.setImage(frame);
    main_image.processImage();


    //ROI roi(main_image.image_marcador_final_C1);

    //roi.showPossibleNumbers();
    //roi.invertColor();
    //roi.show();

    // if ( main_image.findSquare() )
    // {
    // 	main_image.drawSquare();
    // }
    ROI base, marcador;
    Mat base_C1;

    if (main_image.findSquare())
    {
      Mat kernel = Mat::ones(Size(1, 1), CV_8U);

      base.set( main_image.main_img_C3, main_image.mark_rot );

      inRange(base.image, Scalar(ARR_MIN_C2[0], ARR_MIN_C2[1], ARR_MIN_C2[2]), Scalar(ARR_MAX_C2[0], ARR_MAX_C2[1], ARR_MAX_C2[2]), base_C1);

      //morphologyEx(base_C1, base_C1, MORPH_CLOSE, kernel);

      if ( marcador.found(base_C1) )
      {
        marcador.rotatedToImage(base.image);
        marcador.resize(400, 400);
        marcador.showPossibleNumbers();
        //marcador.drawRotated(base.image);
        marcador.show("Marcador");
      }
      else
      {
        // Preenche a imagem com preto para não confundir o algoritmo
        marcador.fill(COLOR_BLACK);
      }

      //imshow("marcador", marcador.mainROI);

      // ISOLAR O MARCADOR DA IMAGEM DA BASE
      // TENTAR CRIAR UMA FUNCAO FINDROI PARA ROTATEDRECT

      //imshow("base_c1", base_C1);

      base.show("Base");
    }


    main_image.show();



    int key = waitKey(30);

    if (key == 32)
    {
      waitKey(-1);
      //imwrite("Imagens/baseHLS_" + std::to_string(contImwrite+1) + ".jpeg", base.mainROI);
      //std::cout << "Salvando frame[" << contImwrite << "]\n";
	    //contImwrite++;
    }
    
  } 
#else
  
	Mat img = imread(argv[1]), img_output;

  LandingMark mark;

  mark.setImage(img);
  mark.processImage();

  if (mark.findSquare())
  {
    ROI roi(mark.main_img_C3, mark.mark_rot);

    roi.show();
  }

  mark.show();

  //cvtColor(img, img, COLOR_RGB2HSV);
  //inRange(img, Scalar(mWHITE, mSAT, mVAL), Scalar(MWHITE, MSAT, MVAL), img_output);
  //morphologyEx(img_output, img_output, MORPH_CLOSE, kernel);

  //ROI roi = findROI(img_output);



	waitKey();

#endif
}