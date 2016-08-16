#ifdef USE_OPENCV
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/image_transformations.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ImageTransformationsTest : public ::testing::Test {
 protected:
  ImageTransformationsTest()
      : seed_(1701) {}

  void CompareTwoImages(const cv::Mat& img_expected,
      const cv::Mat& img_actual) {
    // EXPECT_EQ(img_expected.rows, img_actual.rows);
    // EXPECT_EQ(img_expected.cols, img_actual.cols);
    // EXPECT_EQ(img_expected.channels(), img_actual.channels());
    for (int h = 0; h < img_expected.rows; ++h) {
      const uchar* ptr1 = img_expected.ptr<uchar>(h);
      const uchar* ptr2 = img_actual.ptr<uchar>(h);
      int index = 0;
      for (int w = 0; w < img_expected.cols; ++w) {
        for (int c = 0; c < img_expected.channels(); ++c) {
          int value_expected = ptr1[index];
          int value_actual = ptr2[index++];
          EXPECT_EQ(value_expected, value_actual)
              << "matrices don't agree at"
              << " h=" << h << " w=" << w << " c=" << c << "\n";
        }
      }
    }
  }

  int seed_;
};

TYPED_TEST_CASE(ImageTransformationsTest, TestDtypes);

TYPED_TEST(ImageTransformationsTest, TestNothing) {
}

TYPED_TEST(ImageTransformationsTest, TestResizeImagePeriodic) {
  cv::Mat img_src = cv::Mat::eye(3, 3, CV_8UC1);
  cv::Mat img_dst = cv::Mat::zeros(5, 6, CV_8UC1);
  cv::Mat img_exp;
  ResizeImagePeriodic(img_src, 0, 0, &img_dst);
  img_exp = (cv::Mat_<uint8_t>(5, 6) << 1, 0, 0, 1, 0, 0,
                                        0, 1, 0, 0, 1, 0,
                                        0, 0, 1, 0, 0, 1,
                                        1, 0, 0, 1, 0, 0,
                                        0, 1, 0, 0, 1, 0);
  this->CompareTwoImages(img_dst, img_exp);
}

TYPED_TEST(ImageTransformationsTest, TestResizeImagePeriodic2) {
  cv::Mat img_src = cv::Mat::eye(3, 3, CV_8UC1);
  cv::Mat img_dst = cv::Mat::zeros(5, 6, CV_8UC1);
  cv::Mat img_exp;
  ResizeImagePeriodic(img_src, 1, 2, &img_dst);
  img_exp = (cv::Mat_<uint8_t>(5, 6) << 0, 1, 0, 0, 1, 0,
                                        0, 0, 1, 0, 0, 1,
                                        1, 0, 0, 1, 0, 0,
                                        0, 1, 0, 0, 1, 0,
                                        0, 0, 1, 0, 0, 1);
  this->CompareTwoImages(img_dst, img_exp);
}

TYPED_TEST(ImageTransformationsTest, TestResizeImagePeriodic3) {
  cv::Mat img_src = cv::Mat::eye(3, 3, CV_8UC3);
  cv::Mat img_dst = cv::Mat::zeros(5, 4, CV_8UC3);
  cv::Mat img_exp;
  ResizeImagePeriodic(img_src, 1, 2, &img_dst);
  img_exp = (cv::Mat_<uint8_t>(5, 12) <<
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0);
  this->CompareTwoImages(img_dst, img_exp);
}

}  // namespace caffe

#endif  // USE_OPENCV
