/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2013-, Filippo Basso and Matteo Munaro
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * demo_adaboost.cpp
 *
 *  Created on: Feb 19, 2013
 *      Author: Filippo Basso and Matteo Munaro
 *
 * -----------------------------------------------------------------------------------------------------------------------
 * Demo file showing the learning process of an Online Adaboost classifier based on color features.
 * An example image with 9 colored squares is used.
 * A color classifier is learned for a selected square (by default it is the central one).
 * As features, color features extracted from the RGB histogram of the target are used.
 * The target square is used as positive example, while negative examples are randomly selected from the rest of the image.
 * After every iteration, the confidence values of every square with respect to the classifier learned for the target
 * square are shown.
 * Moreover, the mean color value of the most weighted features is shown in a histogram where the height is proportional
 * to the feature weight. The written number, instead, represents the volume of the corresponding parallelepiped.
 * -----------------------------------------------------------------------------------------------------------------------
 *
 * For Online Adaboost, we used our implementation of the algorithm described in:
 * [1] H. Grabner and H. Bischof. On-line boosting and vision.
 * 	   In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition, pages 260–267, Washington, DC, USA, 2006.
 *
 * For the color features, we used the original implementation described in:
 * [2] F. Basso, M. Munaro, S. Michieletto and E. Menegatti. Fast and robust multi-people tracking from RGB-D data for a mobile robot.
 *     In Proceedings of the 12th Intelligent Autonomous Systems (IAS) Conference, Jeju Island (Korea), 2012.
 *
 * If you use part of this code, please cite [2].
 */

#include <opencv2/opencv.hpp>
#include <adaboost/StrongClassifier.hpp>
#include <adaboost/KalmanWeakClassifier.hpp>
#include <adaboost/ColorFeature.hpp>

#define BINS 16		  	// number of histogram bins for every dimension
#define CHANNELS 3		// number of histogram dimensions (equal to the number of image channels)

typedef adaboost::ColorFeature_<BINS, CHANNELS> _ColorFeature;
typedef adaboost::KalmanWeakClassifier<_ColorFeature> _ColorWeakClassifier;

void calcColorHistogram(const cv::Mat& image, const cv::Mat& blobMask, const int& bins, const int& channels,
                        const int& colorspace, cv::Mat& histogram)
{
  const int histSize[] = {bins, bins, bins};
  const float range[] = {0, 255};
  const float* histRanges[3];
  if (colorspace == 1)
  {
    const float rangeH[] = {0, 179};
    histRanges[0] = rangeH;
    histRanges[1] = range;
    histRanges[2] = range;
  }
  else
  {
    //histRanges = {range, range, range};
    histRanges[0] = range;
    histRanges[1] = range;
    histRanges[2] = range;
  }
  int ch[channels];
  for(int i = 0; i < channels; i++)
    ch[i] = i;

  cv::Mat imageToHist;											// image used to compute the color histogram
  image.copyTo(imageToHist, blobMask);			// the image is filtered with a mask

  if (colorspace == 1) // HSV
    cv::cvtColor(imageToHist, imageToHist, CV_BGR2HSV);
  else
  {
    if (colorspace == 2) // Lab
      cv::cvtColor(imageToHist, imageToHist, CV_BGR2Lab);
    else
    {
      if (colorspace == 3) // Luv
        cv::cvtColor(imageToHist, imageToHist, CV_BGR2Luv);
    }
  }

  cv::calcHist(&imageToHist, 1, ch, blobMask, histogram, channels, histSize, histRanges, true, false);	// color histogram computation
  histogram /= cv::countNonZero(blobMask);		// histogram normalization
}

void createNegativeHistograms(const cv::Mat& image, const cv::Mat& mask, const int n, const int bins, std::vector<cv::Mat>& negative_histograms)
{
	// creates histograms of random image patches to be used as negative examples
  int min_height = 10;
  int min_width = 10;

  for(int i = 0; i < n; ++i)
  {
    int x_min;
    int y_min;
    int width;
    int height;

    bool is_contained = true;

    while(is_contained)
    {
      x_min = rand() % (image.cols - min_width - 1);
      y_min = rand() % (image.rows - min_height - 1);
      width = rand() % (image.cols - x_min - 1);
      height = rand() % (image.rows - y_min - 1);
      is_contained = cv::countNonZero(mask(cv::Rect(x_min, y_min, width, height))) == 0;
    }

    cv::Rect neg_rect(x_min, y_min, width, height);

    cv::Mat neg_image = image(neg_rect);
    cv::Mat neg_mask = mask(neg_rect);
    cv::Mat histogram;
    //std::cout << image.channels() << " " << std::flush;
    calcColorHistogram(neg_image, neg_mask, bins, image.channels(), 0, histogram);
    negative_histograms.push_back(histogram);
  }     
}

void drawFeatures(const std::vector<std::pair<std::vector<std::pair<int, int> >, float> >& bestFeaturesVector, int colorspace)
{
  // draw the mean value of the most weighted features chosen by the classifier
  if (bestFeaturesVector.size() > 0)
  {
    float maxWeight = 0;
    for(size_t i = 0; i < bestFeaturesVector.size(); i++)
    {
      if (bestFeaturesVector[i].second > maxWeight)
        maxWeight = bestFeaturesVector[i].second;
    }

    int mainDim = 50;
    cv::Mat featuresImage(mainDim, mainDim*bestFeaturesVector.size(), CV_8UC3, cv::Scalar(0,0,0));

    float scaleFactorForPlotting;
    if (maxWeight == 0)
      scaleFactorForPlotting = 0;
    else
      scaleFactorForPlotting = mainDim/maxWeight;

    for(size_t i = 0; i < bestFeaturesVector.size(); i++)   // for every feature (for every weak classifier)
    {
      std::vector<std::pair<int, int> > f = bestFeaturesVector[i].first;
      int B1 = f[0].first ;
      int B2 = f[0].second ;
      int G1 = f[1].first ;
      int G2 = f[1].second ;
      int R1 = f[2].first ;
      int R2 = f[2].second ;

      // Mean values:
      float meanB = (B1+B2)* 256 / BINS / 2;
      float meanG = (G1+G2)* 256 / BINS / 2;
      float meanR = (R1+R2)* 256 / BINS / 2;

      //std::cout << "Point before transform: " << meanB << " " << meanG << " " << meanR << std::endl;

      if (colorspace == 1)                    // HSV
      {
        meanB = meanB*360/255;          // [0,360]
        meanG = meanG/255;                      // [0,1]
        meanR = meanR/255;                      // [0,1]
        cv::Mat meanPoint(1,3,CV_32F);
        meanPoint.at<float>(0,0) = meanB;
        meanPoint.at<float>(0,1) = meanG;
        meanPoint.at<float>(0,2) = meanR;
        //cv::Scalar(meanB,meanG,meanR);
        //cv::Scalar meanPoint(meanB,meanG,meanR);
        cv::cvtColor(meanPoint, meanPoint, CV_HSV2BGR);
        meanB = meanPoint.at<float>(0,0);
        meanG = meanPoint.at<float>(0,1);
        meanR = meanPoint.at<float>(0,2);
      }
      else
      {
        if (colorspace == 2)            // Lab
        {
          meanB = meanB*100/255;  // [0,100]
          meanG = meanG - 128;    // [-127,127]
          meanR = meanR - 128;    // [-127,127]
          //cv::Mat meanPoint = cv::Scalar(meanB,meanG,meanR);
          cv::Mat meanPoint(1,3,CV_32F);
          meanPoint.at<float>(0,0) = meanB;
          meanPoint.at<float>(0,1) = meanG;
          meanPoint.at<float>(0,2) = meanR;
          cv::cvtColor(meanPoint, meanPoint, CV_Lab2BGR);
          meanB = meanPoint.at<float>(0,0);
          meanG = meanPoint.at<float>(0,1);
          meanR = meanPoint.at<float>(0,2);
        }
        else
        {
          if (colorspace == 3)    // Luv
          {
            meanB = meanB*100/255;  // [0,100]
            meanG = meanG*354/255 - 134;    // [-134,220]
            meanR = meanR*256/255 - 140;    // [-140,122]
            cv::Mat meanPoint(1,3,CV_32F);
            meanPoint.at<float>(0,0) = meanB;
            meanPoint.at<float>(0,1) = meanG;
            meanPoint.at<float>(0,2) = meanR;
            //cv::Mat meanPoint = cv::Scalar(meanB,meanG,meanR);
            cv::cvtColor(meanPoint, meanPoint, CV_Luv2BGR);
            meanB = meanPoint.at<float>(0,0);
            meanG = meanPoint.at<float>(0,1);
            meanR = meanPoint.at<float>(0,2);
          }
        }
      }

      //std::cout << "Point after transform: " << meanB << " " << meanG << " " << meanR << std::endl;

      // Parallelepiped size:
      int featureSize = (B2-B1+1)*(G2-G1+1)*(R2-R1+1);

      std::stringstream string2;
      string2 << featureSize;

      cv::rectangle(featuresImage, cv::Point(i*mainDim, mainDim - scaleFactorForPlotting*bestFeaturesVector[i].second),
                    cv::Point((i+1)*mainDim-1, mainDim-1), cv::Scalar(meanB,meanG,meanR), CV_FILLED);
      cv::putText(featuresImage, string2.str(), cv::Point(i*mainDim,mainDim*0.9), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(255,255,255), 1.7, CV_AA);

      cv::imshow("Mean value of the chosen features", featuresImage);
      cv::waitKey(5);
    }

  }
}

int main(int argc, char** argv)
{
  // Algorithm parameters:
  int classifiers = 250;			// number of weak classifiers considered at each iteration
  int selectors = 50;					// number of weak classifier that compose the strong classifier
  int numFeaturesToDraw = 8;	// number of chosen features whose mean value is drawn for visualization
  int N_iterations = 1000;	  // number of iterations in this demo

  // The demo image is composed of nine colored squares.
  // Select square 0 -> 8 (default 4) to be used as TARGET (positive example):
  int square = 4;
  if(argc > 1)
    square = atoi(argv[1]);

  // Load and show the example image:
  cv::Mat image = cv::imread("../data/example.jpg");
  cv::Mat image_clone = image.clone();
  cv::imshow("ADABOOST", image_clone);
  cv::waitKey(20);

  // Create ROIs and compute color histograms for the 9 squares:
  std::vector<cv::Mat> histogram_vector;			// vector containing color histograms for every square
  std::vector<cv::Rect> square_roi_vector;		// vector containing bounding box of every square
  for(int i = 0; i < 9; ++i)
  {
    cv::Rect square_roi(cv::Point(43 + 191 * (i % 3), 43 + 191 * (i / 3)), cv::Size(147, 147));
    square_roi_vector.push_back(square_roi);
    cv::Mat target_image = image(square_roi);
    cv::Mat histogram;
    cv::Mat blobMask(target_image.rows, target_image.cols, CV_8UC1, cv::Scalar(255));
    calcColorHistogram(target_image, blobMask, BINS, CHANNELS, 0, histogram);
    histogram_vector.push_back(histogram);
  }

  //Init Adaboost: create random weak classifiers (which are parallelepipeds in the color space (e.g.: in the RGB space)):
  adaboost::StrongClassifier classifier(classifiers, selectors);
  for(int i = 0; i < classifiers; i++)
    classifier.createWeakClassifier<_ColorWeakClassifier>();

  // Create a mask that doesn't contain the target:
  cv::Mat negativeMask(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
  negativeMask(square_roi_vector[square]) = cv::Scalar(0);

  // Update classifier:
  for (int i = 0; i < N_iterations; ++i)		// main loop
  {
    // Update with positive example:
    _ColorWeakClassifier* weak;
    classifier.update(cv::Point(0, 0), 1, histogram_vector[square], numFeaturesToDraw);
    weak = classifier.replaceWorstWeakClassifier<_ColorWeakClassifier>();

    // Create negative examples (randomly chosen from the rest of the image):
    std::vector<cv::Mat> wrongHistograms;
    createNegativeHistograms(image, negativeMask, 10, BINS, wrongHistograms);

    // Update with negative examples:
    for(int j = 0; j < wrongHistograms.size(); j++)
    {
      classifier.update(cv::Point(0, 0), -1, wrongHistograms[j], numFeaturesToDraw);
      weak = classifier.replaceWorstWeakClassifier<_ColorWeakClassifier>();
    }

    if(true) //or i == 0 or i % 10 == 0)
    {
      // Draw the mean color of the most weighted features:
      drawFeatures(classifier.getBestFeatures(), 0);

      // Compute Adaboost classification score for each square:
      for(int j = 0; j < 9; ++j)
      {
        float prediction = classifier.predict(cv::Point(0, 0), histogram_vector[j]);
        std::stringstream prediction_ss;
        prediction_ss << classifier.predict(cv::Point(0, 0), histogram_vector[j]);
        cv::putText(image_clone, prediction_ss.str(), square_roi_vector[j].tl(), cv::FONT_HERSHEY_SIMPLEX,
                    0.75, cv::Scalar(255, 255, 255), prediction > 0 ? 2 : 1, CV_AA);
      }
      cv::imshow("ADABOOST", image_clone);
      cv::waitKey(5);
      image_clone = image.clone();
    }
  }

  cv::destroyAllWindows();
  cv::waitKey(0);

  return 0;
}
