#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_interpolation_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SoftmaxInterpolationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxInterpolationLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(2, 3, 2, 2)),
        blob_bottom_data2_(new Blob<Dtype>(*(new vector<int>(1, 3)))),
        blob_top_data_(new Blob<Dtype>()) {
    int index = 0;
    for (int i = 0; i < blob_bottom_data_->shape(0); ++i) {
      for (int c = 0; c < blob_bottom_data_->shape(1); ++c) {
        for (int j = 0; j < blob_bottom_data_->count(2); ++j) {
          blob_bottom_data_->mutable_cpu_data()[index] = Dtype(c+1) / Dtype(6);
          index += 1;
        }
      }
    }
    for (int i = 0; i < blob_bottom_data2_->count(); ++i) {
      blob_bottom_data2_->mutable_cpu_data()[i] = i;
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data2_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~SoftmaxInterpolationLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_data2_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_data2_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxInterpolationLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxInterpolationLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxInterpolationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(SoftmaxInterpolationLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxInterpolationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  int index = 0;
  for (int i = 0; i < this->blob_bottom_data_->shape(0); ++i) {
    for (int j = 0; j < this->blob_bottom_data_->count(2); ++j) {
      EXPECT_NEAR(this->blob_top_data_->cpu_data()[index],
          Dtype(4) / Dtype(3), 1e-6);
      index += 1;
  }
  }
}

}  // namespace caffe
