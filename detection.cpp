#include "ROI/ROI.hpp"
#include "LandingMark/landingMark.hpp"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <math.h>
#include <sstream>

using namespace cv;
using namespace std;

#define DEBUGCOLOR false
#define DEBUG_MODE false

// Arrays utilizados no inRange

int ARR_MAXWHITE[3] = {MAXWHITE, MAXSATWHITE, MAXVALWHITE};
int ARR_MINWHITE[3] = {MINWHITE, MINSATWHITE, MINVALWHITE};

int ARR_MAX_C2[3] = {65, 60, 45};
int ARR_MIN_C2[3] = {10, 10, 5};

int ARR_MAXMARCADOR_HSV[3] = {140, 40, 255};
int ARR_MINMARCADOR_HSV[3] = {0, 0, 130};

const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());

class TesseractExtract
{
public:

  tesseract::TessBaseAPI *ocr;



  TesseractExtract()
  {
    ocr = new tesseract::TessBaseAPI();

    ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);

    ocr->SetVariable("tessedit_char_whitelist","0123456789-");

    ocr->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
  }

  ~TesseractExtract()
  {
    ocr->End();
  }
  
  string extract(Mat img, int channels=1)
  {
    string numbers;

    imshow("68", img);

    ocr->SetImage(img.data, img.cols, img.rows, channels, img.step);

    ocr->SetSourceResolution(70);

    numbers = std::string(ocr->GetUTF8Text());

    return numbers;
  }
};


class ContourWithData 
{
public:
    vector<Point> ptContour;
    Rect boundingRect;
    float fltArea;


    bool checkIfContourIsValid() {
        if (fltArea < MIN_CONTOUR_AREA) return false;
        return true;
    }

    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) { 
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
    }

};


bool train()
{
    Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector

    FileStorage fsClassifications("classifications.xml", FileStorage::READ);        // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
        cout << "ERRO, falha ao abrir o xml classificador\n";    // show error message
        return false;                                                                                  // and exit program
    }

    fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
    fsClassifications.release();                                        // close the classifications file

                                                                        // read in training images ////////////////////////////////////////////////////////////

    Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

    FileStorage fsTrainingImages("images.xml", FileStorage::READ);          // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "ERRO, falha ao abrir o xml imagens\n";         // show error message
        return false;                                                                              // and exit program
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
    fsTrainingImages.release();                                                 // close the traning images file

                                                                                // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
                                                                                // even though in reality they are multiple images / numbers
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts); 

    cout << "\nTreino finalizado\n";
}


string getKNNChar(cv::Mat img, const char* name)
{
    Mat matTestingNumbers = img.clone();

    Rect rect_left( Point(10, 0), Point(img.cols * 0.52, img.rows) );
    Rect rect_right( Point(img.cols * 0.52, 0), Point(img.cols, img.rows) );


    // rectangle(matTestingNumbers, rect_left, 1, 2);
    // rectangle(matTestingNumbers, rect_right, 1, 2);


    if (matTestingNumbers.empty()) {
        cout << "ERRO: a imagem está vazia\n";
        return ""; 
    }

    Mat matThresh;   

    // filter image from grayscale to black and white
    adaptiveThreshold(matTestingNumbers, matThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);                                   // constant subtracted from the mean or weighted mean


    Mat LEFT_ROI = matThresh(rect_left);
    Mat RIGHT_ROI = matThresh(rect_right);

    resize(LEFT_ROI, LEFT_ROI, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));  
    resize(RIGHT_ROI, RIGHT_ROI, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));


    imshow(name + '0', LEFT_ROI);
    imshow(name + '1', RIGHT_ROI);

    Mat LEFT_FLOAT, RIGHT_FLOAT;

    LEFT_ROI.convertTo(LEFT_FLOAT, CV_32FC1); 
    RIGHT_ROI.convertTo(RIGHT_FLOAT, CV_32FC1);      

    Mat LEFT_FLATTEN = LEFT_FLOAT.reshape(1, 1);
    Mat RIGHT_FLATTEN = RIGHT_FLOAT.reshape(1, 1);

    Mat LEFT_CHAR(0, 0, CV_32F);
    Mat RIGHT_CHAR(0, 0, CV_32F);

    kNearest->findNearest(LEFT_FLATTEN, 1, LEFT_CHAR); 
    kNearest->findNearest(RIGHT_FLATTEN, 1, RIGHT_CHAR);

    float FLOAT_LEFT_CHAR = (float)LEFT_CHAR.at<float>(0, 0);
    float FLOAT_RIGHT_CHAR = (float)RIGHT_CHAR.at<float>(0, 0);

    string strFinalString;
    strFinalString = char(int(FLOAT_LEFT_CHAR));
    strFinalString += char(int(FLOAT_RIGHT_CHAR));

    imshow(name, matTestingNumbers);
 
    return strFinalString;
}


string getPercent(Mat img)
{
  if (img.empty()) {
      cout << "ERRO: a imagem está vazia\n";
      return ""; 
  }

  Mat matThresh;   

  // filter image from grayscale to black and white
  adaptiveThreshold(img, matThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);    


  resize(matThresh, matThresh, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));


  imshow("PERCENTresize", matThresh);

  Mat matROIFloat;
  matThresh.convertTo(matROIFloat, CV_32FC1);  
  Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
  Mat matCurrentChar(0, 0, CV_32F);

  kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);

  string strFinalString = "";
  float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);   
  strFinalString += char(int(fltCurrentChar));

  return strFinalString;
}


void draw_rectangle(Mat img, Point pt1, Point pt2)
{
  rectangle(img, Rect(pt1, pt2), 0, 2);
}


void remove_border(Mat img, int thickness)
{
    Rect borda1(Point(0, 0),                Point(img.cols, thickness)         );
    Rect borda2(Point(img.cols,    0),      Point(img.cols-thickness, img.rows)); 
    Rect borda3(Point(img.cols, img.rows),  Point(0, img.rows-thickness)       );
    Rect borda4(Point(0, img.rows),         Point(thickness, 0)                );

    rectangle(img, borda1, 0, -1);
    rectangle(img, borda2, 0, -1);
    rectangle(img, borda3, 0, -1);
    rectangle(img, borda4, 0, -1);
}


void clean_image(Mat img)
{
  if (img.channels() > 1)
    cvtColor(img, img, CV_BGR2GRAY);

  threshold(img, img, 70, 255, THRESH_BINARY);
}


void drawRotated(Mat img, RotatedRect rotated, Scalar color = cvCOLOR_RED)
{
  Point2f vertices2f[4];
  rotated.points(vertices2f);

  // Convert them so we can use them in a fillConvexPoly
  Point vertices[4];    
  for(int i = 0; i < 4; ++i){
      vertices[i] = vertices2f[i];
  }

  // Now we can fill the rotated rectangle with our specified color
  cv::fillConvexPoly(img, vertices, 4, color);
}


Mat getImageFromRect(Rect rect)
{
  Mat img = img(rect);
  return img;
}


Mat draw_numbers(Mat img)
{
    int element_lenght = img.cols * 0.06;

    Mat kernel = Mat::ones(Size(7, 7), CV_8U);
    Mat img_output = img.clone();
    Mat element = getStructuringElement(CV_SHAPE_RECT, Size(element_lenght, 1));


    cvtColor(img_output, img_output, CV_GRAY2BGR);

    morphologyEx(img, img, MORPH_ERODE, kernel);
    morphologyEx(img, img, MORPH_GRADIENT, element);


    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat mask = Mat::zeros(img.size(), CV_8UC1);

    findContours(img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = cv::Scalar(0, 0, 0);

        drawContours(mask, contours, idx, cv::Scalar(255, 255, 255), CV_FILLED);

        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);

        if (r > .45 /* assume at least 45% of the area is filled if it contains text */
            && 
            rect.area() > (img.rows*img.cols) * 0.05 && rect.width/float(rect.height) >= 1.44 /* constraints on region size */
            /* these two conditions alone are not very robust. better to use something 
            like the number of significant peaks in a horizontal projection as a third condition */
            )
        {
            cout << "Desenhou!\n";
            rectangle(img_output, rect, cv::Scalar(0, 255, 0), 2);
        }
    }

    return img_output;
}


Mat rotateImage(Mat img, double angle)
{
  Point center = Point(img.rows/2, img.cols/2);
  Mat rotated;

  Mat M = getRotationMatrix2D(center, angle, 1.0);
  warpAffine(img, rotated, M, img.size(), INTER_LINEAR);

  return rotated;
}


Mat rotateToImage(Mat img, RotatedRect rect)
{

	Mat rotated, cropped;
	float angle = rect.angle;
    Size rect_size = rect.size;

	// if (biggest_rect.angle < -45.) {
	// 	angle += 90.0;
    //     swap(rect_size.width, rect_size.height);
	// }

	Mat M = getRotationMatrix2D(rect.center, angle, 1.0);

    // Rotaciona a imagem
	warpAffine(img, rotated, M, img.size());

	// Isola o rotatedrect para dentro do cropped
	getRectSubPix(rotated, rect_size, rect.center, cropped);

    return cropped;
}


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
    cout << "Insira o nome do video!" << endl;
    return -1;
  }


    VideoCapture cap(argv[1]);

    if (!cap.isOpened())
    {
        cout << "Erro: video não encontrado" << endl;
        return -1;
    }

    vector<Mat> number_images(2);

    Mat frame, frame_sensor, frame_yellow, frameHSV;

    LandingMark main_image;

    TesseractExtract tess;


    int SUM_ANGLE = 0;
    int contImwrite = 0;
    int denominador = 1, numerador = 0;
    long cont_68_tess = 0, cont_10_tess = 0, cont_68_knn = 0, cont_10_knn = 0;
    long contador_frames = 0;

    long count_non_percent = 0, count_percent = 0;
    long knn_10_percent = 0, knn_68_percent = 0, tess_68_percent = 0;

    Mat img_rotate_test; 
    bool rotated = false; 

    ROI base, marcador;
    Mat base_C1;

    #if DEBUG_MODE
        cout << "\x1B[2J\x1B[H"; 
        cout << "##########################\n\n";
        cout << "Entrando no debug mode\n";
        cout << "Pressione ESPAÇO para editar ou ESC para fechar\n\n";
        cout << "##########################\n\n";

        Mat kernel;
        int keyPressed = 255, keySpace = 255;
        int count_current_stage = 0, imwrite_cont = 0, color_selector = 0;
        string ARRAY_NAME = "BLUE";
        bool change = false;
        
        int ARR_MAXBLUE[3]   = {MAXBLUE, MAXSATBLUE, MAXVALBLUE};
        int ARR_MINBLUE[3]   = {MINBLUE, MINSATBLUE, MINVALBLUE};

        int ARR_MAXYELLOW[3] = {MAXYELLOW, MAXSATYELLOW, MAXVALYELLOW};
        int ARR_MINYELLOW[3] = {MINYELLOW, MINSATYELLOW, MINVALYELLOW};


        while (keyPressed != ESC)
        {
            cap >> frame;

            if (frame.empty())
                break;

            main_image.setImage(frame);
            main_image.processImage(ARR_MINBLUE, ARR_MAXBLUE, ARR_MINYELLOW, ARR_MAXYELLOW);
            
            imshow("MAIN", main_image.image);
            imshow("HSV", main_image.main_imgHSV_C3);
            imshow("BITWISE_BLUE_YELLOW", main_image.img_final_C1);
            imshow("BLUE", main_image.img_blue_C1);
            imshow("YELLOW", main_image.img_yellow_C1);

            if (main_image.foundMark())
            {
                base.set( main_image.image, main_image.markRotatedRect );

                if (base.image.rows > 0 && base.image.cols > 0)
                {
                    kernel = Mat::ones(Size(1, 1), CV_8U);

                    inRange(base.image, Scalar(ARR_MIN_C2[0], ARR_MIN_C2[1], ARR_MIN_C2[2]), Scalar(ARR_MAX_C2[0], ARR_MAX_C2[1], ARR_MAX_C2[2]), base_C1);
                    morphologyEx(base_C1, base_C1, MORPH_CLOSE, kernel);
                    detailEnhance(base.image, base.image);

                    base.show("Base");
                }
            }

            switch (keyPressed)
            {
                case SPACE:

                    cout << "Entrando no editor de limiares\n";
                    cout << "Pressione ESC para sair\n\n";
                    cout << "##########################\n\n";

                    cout << "DESLIGAR O CAPSLOCK PARA FUNCIONAR\n\n";
                    cout << "AZUL (<-)  AMARELO (->)\n\n";

                    cout << "Incrementar valores minimos:  Q   W   E\n";
                    cout << "Drecrementar valores minimos: A   S   D\n\n";
                    
                    cout << "Incrementar valores maximos:  T   Y   U\n";
                    cout << "Drecrementar valores maximos: G   H   J\n\n";


                    while (keySpace != ESC)
                    {
                        keySpace = waitKey(0);


                        switch (keySpace)
                        {
                            // ALTERA OS VALORES MINIMOS
                                case KEY_Q:
                                    if (color_selector == 0){
                                        ARR_MINBLUE[0] += 5;
                                    }else{
                                        ARR_MINYELLOW[0] += 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_A: 
                                    if (color_selector == 0){
                                        ARR_MINBLUE[0] -= 5;
                                    }else{
                                        ARR_MINYELLOW[0] -= 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_W: 
                                    if (color_selector == 0){
                                        ARR_MINBLUE[1] += 5;
                                    }else{
                                        ARR_MINYELLOW[1] += 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_S: 
                                    if (color_selector == 0){
                                        ARR_MINBLUE[1] -= 5;
                                    }else{
                                        ARR_MINYELLOW[1] -= 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_E:
                                    if (color_selector == 0){
                                        ARR_MINBLUE[2] += 5;
                                    }else{
                                        ARR_MINYELLOW[2] += 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_D:
                                    if (color_selector == 0){
                                        ARR_MINBLUE[2] -= 5;
                                    }else{
                                        ARR_MINYELLOW[2] -= 5;
                                    }
                                    change = true;
                                    break;

                            // ALTERA OS VALORES MAXIMOS
                                case KEY_T:
                                    if (color_selector == 0){
                                        ARR_MAXBLUE[0] += 5;
                                    }else{
                                        ARR_MAXYELLOW[0] += 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_G: 
                                    if (color_selector == 0){
                                        ARR_MAXBLUE[0] -= 5;
                                    }else{
                                        ARR_MAXYELLOW[0] -= 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_Y: 
                                    if (color_selector == 0){
                                        ARR_MAXBLUE[1] += 5;
                                    }else{
                                        ARR_MAXYELLOW[1] += 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_H: 
                                    if (color_selector == 0){
                                        ARR_MAXBLUE[1] -= 5;
                                    }else{
                                        ARR_MAXYELLOW[1] -= 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_U:
                                    if (color_selector == 0){
                                        ARR_MAXBLUE[2] += 5;
                                    }else{
                                        ARR_MAXYELLOW[2] += 5;
                                    }
                                    change = true;
                                    break;

                                case KEY_J:
                                    if (color_selector == 0){
                                        ARR_MAXBLUE[2] -= 5;
                                    }else{
                                        ARR_MAXYELLOW[2] -= 5;
                                    }
                                    change = true;
                                    break;

                            case KEY_LEFT:
                                cout << "BLUE SELECIONADO\n";
                                ARRAY_NAME = "BLUE";
                                color_selector = 0;
                                break;

                            case KEY_RIGHT:
                                cout << "YELLOW SELECIONADO\n";
                                ARRAY_NAME = "YELLOW";
                                color_selector = 1;
                                break;

                            default:
                                break;
                        }

                        if (change)
                        {
                            cout << "\x1B[2J\x1B[H"; 
                            cout << ARRAY_NAME << "\n\n";

                            main_image.processImage(ARR_MINBLUE, ARR_MAXBLUE, ARR_MINYELLOW, ARR_MAXYELLOW);
                            imshow("BITWISE_BLUE_YELLOW", main_image.img_final_C1);
                            imshow("BLUE", main_image.img_blue_C1);
                            imshow("YELLOW", main_image.img_yellow_C1);

                            if (color_selector == 0){
                                cout << "MAX: " << ARR_MAXBLUE[0] << " " << ARR_MAXBLUE[1] << " " << ARR_MAXBLUE[2] << "\n";
                                cout << "MIN: " << ARR_MINBLUE[0] << " " << ARR_MINBLUE[1] << " " << ARR_MINBLUE[2] << "\n";
                            }else{
                                cout << "MAX: " << ARR_MAXYELLOW[0] << " " << ARR_MAXYELLOW[1] << " " << ARR_MAXYELLOW[2] << "\n";
                                cout << "MIN: " << ARR_MINYELLOW[0] << " " << ARR_MINYELLOW[1] << " " << ARR_MINYELLOW[2] << "\n";
                            }
                            change = false;
                        }
                    }

                    keySpace = 255;

                    break;

                default:
                    break;
            }

            keyPressed = waitKey(50);
        }

        cout << "\nESC pressionado. Fechando o programa\n";

    #else

        train();

        while(true)
        {
            cap >> frame;

            if (frame.empty())
                break;

            main_image.setImage(frame);
            main_image.processImage();


            ROI base, marcador;
            Mat base_C1;

            Mat img_teste, img_inrange;
            Mat img_percent;
            Mat img_68, img_10;



            //SE ACHOU A BASE
            if (main_image.foundMark())
            {
                Mat kernel = Mat::ones(Size(1, 1), CV_8U);

                base.set( main_image.image, main_image.markRotatedRect );

                inRange(base.image, Scalar(ARR_MIN_C2[0], ARR_MIN_C2[1], ARR_MIN_C2[2]), Scalar(ARR_MAX_C2[0], ARR_MAX_C2[1], ARR_MAX_C2[2]), base_C1);
                morphologyEx(base_C1, base_C1, MORPH_CLOSE, kernel);
                detailEnhance(base.image, base.image);


                if ( marcador.found(base_C1) )
                {
                    marcador.biggest_rect.angle += SUM_ANGLE;
                    marcador.image = rotateToImage(base.image, marcador.biggest_rect);
                    marcador.resize(400, 400);
                    marcador.improve_image();


                    ////// INICIO ROTAÇÃO DE IMAGEM //////

                        Mat marcador_copy = marcador.image.clone();

                        cvtColor(marcador.image, img_teste, COLOR_BGR2HSV);
                        inRange(img_teste, Scalar(ARR_MINMARCADOR_HSV[0], ARR_MINMARCADOR_HSV[1], ARR_MINMARCADOR_HSV[2]), Scalar(ARR_MAXMARCADOR_HSV[0], ARR_MAXMARCADOR_HSV[1], ARR_MAXMARCADOR_HSV[2]), img_inrange );
                        invert_color(img_inrange);


                        marcador.getRectNumbersStatic(img_inrange);
                        img_percent = img_inrange(marcador.numbers[2]);


                        string n_10_knn;
                        string n_68_tess;
                        string returned = getPercent(img_percent);

                        if (returned.compare("%") != 0){
                            count_non_percent++;

                            if (count_percent > 0){
                                count_percent--;
                            }
                        }
                        else{
                            contador_frames++;

                            img_10 = img_inrange(marcador.numbers[1]);
                            img_68 = img_inrange(marcador.numbers[0]);
                            copyMakeBorder(img_68, img_68, 10, 10, 10, 10, BORDER_ISOLATED, 255);

                            n_10_knn = getKNNChar(img_10, "10");
                            n_68_tess = tess.extract(img_68);

                            if (n_10_knn.compare("10") == 0){
                                knn_10_percent++;
                            }

                            if (n_68_tess.compare("68\n") == 0){
                                tess_68_percent++;
                            }

                            cout << fixed;
                            cout << setprecision(1);
                            cout << "Taxa 10: " << (knn_10_percent / (float)contador_frames)*100 << "%     Taxa 68: " << (tess_68_percent / (float)contador_frames)*100 << "%\n";

                            count_percent++;

                            if (count_non_percent > 0){
                                count_non_percent--;
                            }
                        }



                        if (count_non_percent >= 3)
                        {
                            SUM_ANGLE += 90;
                            count_non_percent--;
                            if (SUM_ANGLE == 360){
                                SUM_ANGLE = 0;
                            }
                        }
                        
                        marcador.show("Marcador");

                    ////// FIM ROTAÇÃO DE IMAGEM //////




                    ////// INICIO NÃO PEGAR VALORES ENQUANTO % FOR DIRETENRE //////

                        // cvtColor(marcador.image, img_teste, COLOR_BGR2HSV);
                        // inRange(img_teste, Scalar(ARR_MINMARCADOR_HSV[0], ARR_MINMARCADOR_HSV[1], ARR_MINMARCADOR_HSV[2]), Scalar(ARR_MAXMARCADOR_HSV[0], ARR_MAXMARCADOR_HSV[1], ARR_MAXMARCADOR_HSV[2]), img_inrange );
                        // invert_color(img_inrange);


                        // marcador.getRectNumbersStatic(img_inrange);
                        // img_percent = img_inrange(marcador.numbers[2]);
                        // img_10 = img_inrange(marcador.numbers[1]);
                        // img_68 = img_inrange(marcador.numbers[0]);

                        // string knn_10, knn_68;
                        // string n_68_tess;
                        // string returned = getPercent(img_percent);

                        // copyMakeBorder(img_68, img_68, 10, 10, 10, 10, BORDER_ISOLATED, 255);
                        // copyMakeBorder(img_10, img_10, 5, 5, 5, 5, BORDER_ISOLATED, 255);
                        

                        // if (returned.compare("%") != 0)
                        // {
                        //     if (count_non_percent <= 5)
                        //     {
                        //         knn_10 = getKNNChar(img_10, "10");
                        //         n_68_tess = tess.extract(img_68);
                                
                        //         if (knn_10.compare("10") == 0){
                        //             knn_10_percent++;
                        //         }

                        //         if (n_68_tess.compare("68\n") == 0){
                        //             tess_68_percent++;
                        //         }

                        //         contador_frames++;
                        //         count_non_percent++;
                        //     }
                        //     else
                        //     {
                        //         cout << "LIMITE ATINGIDO\n\n";
                        //     }
                        // }
                        // else
                        // {
                        //     if (count_non_percent > 0){
                        //         count_non_percent--;
                        //     }
                    

                        //     knn_10 = getKNNChar(img_10, "10");
                        //     n_68_tess = tess.extract(img_68);
                            
                        //     if (knn_10.compare("10") == 0){
                        //       knn_10_percent++;
                        //     }

                        //     if (n_68_tess.compare("68\n") == 0){
                        //       tess_68_percent++;
                        //     }
                            
                        //     contador_frames++;
                        // }

                        // imshow("inRange", img_inrange);
                    
                        // cout << fixed;
                        // cout << setprecision(1);
                        // cout <<"Taxa 10: " << (knn_10_percent / (float)contador_frames)*100 << "%    Taxa 68: " << (tess_68_percent / (float)contador_frames)*100 << "%\n";
                        
                    ////// FIM NÃO PEGAR VALORES ENQUANTOS % FOR DIRETENRE //////




                    ////// TESTE DE ROTAÇÃO DA IMAGEM ///////
                    
                        // numerador++;

                        // if (numerador % 15 == 0)
                        // {
                        //     SUM_ANGLE += 90;
                        // }


                        // imshow("main", marcador.image);

                    ///// FIM TESTE DE ROTAÇÃO DA IMAGEM //////

                }
            }

            main_image.show();

            int key = waitKey(30);

            if (key == SPACE)
            {
                imwrite("Imagens/10"+to_string(contImwrite)+".jpeg", img_10);
                imwrite("Imagens/68"+to_string(contImwrite)+".jpeg", img_68);
                cout << "Salvando frame[" << contImwrite << "]\n";
                contImwrite++;
                // while (temp_key != SPACE)
                // {
                //   if (temp_key == ESC)
                //   {
                //     Mat outimg, temp;

                //     //temp = marcador.clean_img(marcador.numbers[0]);
                //     //int borda = 50;

                //     //copyMakeBorder(temp, outimg, borda, borda, borda, borda, BORDER_ISOLATED, 255);
                //     //imwrite("Imagens/numbers_" + std::to_string(contImwrite+1) + ".jpeg", main_image.image);

                //     imwrite("Imagens/frame_basec1.jpeg", base_C1);
                //     std::cout << "Salvando frame[" << contImwrite << "]\n";
                //     contImwrite++;
                //     temp_key = 255;
                //   }
                // }
            }
        }
    #endif
}