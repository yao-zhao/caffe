#ifdef USE_OPENCV
#include <vector>

#include "gtest/gtest.h"

#include "caffe/image_transformations.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageTransformationsTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageTransformationsTest()
      : seed_(1701) {}

  CompareTwoImages(const cv::Mat img1, const cv::Mat img2) {
    CHECK_EQ(img1.rows(), img2.rows());
    CHECK_EQ(img1.cols(), img2.cols());
    CHECK_EQ(img1.channels(), img2.channels());
    for (h = 0; h < img1.rows(); ++h) {
      uchar* ptr1 = img1.ptr<uchar>(h);
      uchar* ptr2 = img2.ptr<uchar>(h);
      for (int w = 0; w < w_dst; ++w) {
        for (int c = 0; c < num_channels; ++c) {
        }
      }
    }
  }

  int seed_;
};

TYPED_TEST_CASE(ImageTransformationsTest);

TYPED_TEST_CASE(ImageTransformationsTest, TestNothing) {
}

TYPED_TEST_CASE(ImageTransformationsTest, TestResizeImagePeriodic) {
  cv::Mat img_src = cv::Mat::eye(3, 3, cv::CV_8UC3);
  cv::Mat img_dst = cv::Mat::zeros(5, 6, cv::CV_8UC3);
  cv::Mat img_exp;
  ResizeImagePeriodic(img_src, 0, 0, img_dst);
  img_exp = (cv::Mat_<double>(5,6) << 1, 0, 0, 1, 0, 0,
                                      0, 1, 0, 0, 1, 0,
                                      0, 0, 1, 0, 0, 1,
                                      1, 0, 0, 1, 0, 0,
                                      0, 1, 0, 0, 1, 0);
}

}

#endif  // USE_OPENCV