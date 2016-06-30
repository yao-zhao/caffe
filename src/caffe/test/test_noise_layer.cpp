#include <algorithm>
#include <vector>
#include <math.h>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NoiseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  // constructor
  NoiseLayerTest()
  : blob_bottom_(new Blob<Dtype>()),
    blob_top_(new Blob<Dtype>()),
    blob_tmp_(new Blob<Dtype>()) {}
  // set up
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2,3,4,4);
    // fill the values
    FillerParameter filler_param;
    // filler_param.set_type("constant");
    filler_param.set_type("gaussian");
    // filler_param.set_value(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  // destructor
  virtual ~NoiseLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_tmp_;
  }
  // members
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Blob<Dtype>* blob_tmp_;
  // test foward
  void TestForward() {
    LayerParameter layer_param;
    NoiseParameter* noise_param = layer_param.mutable_noise_param();
    noise_param->set_sigma(0.1);
    NoiseLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_vec_[0]->num(), this->blob_bottom_vec_[0]->num());
    EXPECT_EQ(this->blob_top_vec_[0]->channels(), this->blob_bottom_vec_[0]->channels());
    EXPECT_EQ(this->blob_top_vec_[0]->height(), this->blob_bottom_vec_[0]->height());
    EXPECT_EQ(this->blob_top_vec_[0]->width(), this->blob_bottom_vec_[0]->width());
    // copy to bottom
    blob_tmp_->ReshapeLike(*blob_bottom_vec_[0]);
    caffe_copy(blob_tmp_->count(), 
      blob_bottom_vec_[0]->cpu_data(), blob_tmp_->mutable_cpu_data());
    // forward
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // count value
    int count = this->blob_bottom_->count();
    int count_lesser = 0;
    int count_greater = 0;
    Dtype diff=0;
    Dtype var=0;
    const Dtype* bottom_data = this->blob_tmp_->cpu_data();
    const Dtype* top_data = this->blob_top_vec_[0]->cpu_data();
    
    for (int i=0; i<count; ++i) {
      diff = top_data[i] - bottom_data[i];
      var += (diff*diff)/count;
      EXPECT_NEAR(diff, 0.0, 1.0);
      if ( diff < 0 ) ++count_lesser;
      else ++count_greater;
    }
    EXPECT_LE(abs(count_greater-count_lesser),4*sqrt(count));
    EXPECT_LE(sqrt(var), 0.12);
    EXPECT_GE(sqrt(var), 0.08);
  }
};

TYPED_TEST_CASE(NoiseLayerTest, TestDtypesAndDevices);

TYPED_TEST(NoiseLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  // setup
  LayerParameter layer_param;
  NoiseParameter* noise_param = layer_param.mutable_noise_param();
  noise_param->set_sigma(0.1);
  NoiseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(), this->blob_bottom_vec_[0]->num());
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), this->blob_bottom_vec_[0]->channels());
  EXPECT_EQ(this->blob_top_vec_[0]->height(), this->blob_bottom_vec_[0]->height());
  EXPECT_EQ(this->blob_top_vec_[0]->width(), this->blob_bottom_vec_[0]->width());
}

TYPED_TEST(NoiseLayerTest, TestForward) {
  this->blob_top_vec_[0] = this->blob_top_;
  this->TestForward();
}

TYPED_TEST(NoiseLayerTest, TestForwardInPlace) {
  this->blob_top_vec_[0] = this->blob_bottom_;
  this->TestForward();
}

TYPED_TEST(NoiseLayerTest, TestGradientEltwise) {
  typedef typename TypeParam::Dtype Dtype;
   // setup
  this->blob_top_vec_[0] = this->blob_top_;
  LayerParameter layer_param;
  NoiseParameter* noise_param = layer_param.mutable_noise_param();
  noise_param->set_sigma(0.1);
  NoiseLayer<Dtype> layer(layer_param);
  // check gradient
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_);
}

// this will not work, because in-place calculation changes the bottom data
// TYPED_TEST(NoiseLayerTest, TestGradientEltwiseInPlace) {
//   typedef typename TypeParam::Dtype Dtype;
//    // setup
//   this->blob_top_vec_[0] = this->blob_bottom_;
//   LayerParameter layer_param;
//   NoiseParameter* noise_param = layer_param.mutable_noise_param();
//   noise_param->set_sigma(0.1);
//   // check gradient
//   NoiseLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-2);
//   checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
//     this->blob_top_vec_);
// }
}

