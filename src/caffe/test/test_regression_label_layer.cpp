#include <algorithm>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/regression_label_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RegressionLabelLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RegressionLabelLayerTest()
    : blob_bottom_(new Blob<Dtype>(4, 1, 1, 1)),
    blob_top_0_(new Blob<Dtype>()),
    blob_top_1_(new Blob<Dtype>()) {
    blob_top_vec_.push_back(blob_top_0_);
    blob_top_vec_.push_back(blob_top_1_);
    blob_bottom_vec_.push_back(blob_bottom_);
    // fill the values
    Dtype input[] = {.1, .2, .3, .4};
    // assign the values
    for (int i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
      this->blob_bottom_vec_[0]->mutable_cpu_data()[i] = input[i];
    }
  }
  virtual ~RegressionLabelLayerTest() { delete blob_bottom_;
      delete blob_top_0_; delete blob_top_1_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RegressionLabelLayerTest, TestDtypesAndDevices);

TYPED_TEST(RegressionLabelLayerTest, TestSetup) {
  this->SetUp();
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RegressionLabelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(),
      this->blob_bottom_vec_[0]->shape());
  EXPECT_EQ(this->blob_top_vec_[1]->shape(),
      this->blob_bottom_vec_[0]->shape());
}

TYPED_TEST(RegressionLabelLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_regression_label_param()->
    set_lower_bound(0.0);
  layer_param.mutable_regression_label_param()->
    set_upper_bound(1.0);
  RegressionLabelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype expected_data_0[] = {.1, .2, .3, .4};
  Dtype expected_data_1[] = {.9, .8, .7, .6};
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    EXPECT_EQ(this->blob_top_vec_[0]->
      cpu_data()[i], expected_data_0[i]);
    EXPECT_EQ(this->blob_top_vec_[1]->
      cpu_data()[i], expected_data_1[i]);
  }
}

TYPED_TEST(RegressionLabelLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_regression_label_param()->
    set_lower_bound(0.0);
  layer_param.mutable_regression_label_param()->
    set_upper_bound(1.0);
  RegressionLabelLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
