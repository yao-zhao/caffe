#include <vector>
#include <math.h>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/noise_layer.hpp"


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
    blob_top_(new Blob<Dtype>()) {}
    // set up
    virtual void SetUp() {
      Caffe::set_random_seed(1701);
      blob_bottom_->Reshape(64,3,256,256);
      // fill the values
      FillerParameter filler_param;
      // filler_param.set_type("constant");
      filler_param.set_type("gaussian");
      // filler_param.set_value(10);
      Filler<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec.push_back(blob_bottom_);
      blob_top_vec.push_back(blob_top_);
    }
    // destructor
    virtual ~NoiseLayerTest() {
      delete blob_bottom_;
      delete blob_top_;
    }
    // members
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    // test foward
    void TestForward() {
      typedef typename TypeParam::Dtype Dtype;
      LayerParameter layer_param;
      NoiseParameter* noise_param = layer_param.mutable_noise_param();
      noise_param->set_sigma(1);
      NoiseLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
      EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
      EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
      EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
      layer.Foward(blob_bottom_vec_, blob_top_vec_);
      // count value
      int count = this->blob_bottom_->count();
      int count_lesser = 0;
      int count_greater = 0;
      Dtype diff;
      Dtype var;
      const Dtype* bottom_data = this->blob_bottom_->cpu_data();
      const Dtype* top_data = this->blob_top_->cpu_data();
      for (int i=0; i<count; ++i) {
        diff = top_data[i] - bottom_data[i];
        var += (diff*diff)/count;
        EXPECT_NEAR(diff, 0.0, 2.0);
        if ( diff < 0 ) ++count_lesser;
        else ++count_greater;
      }
      EXPECT_LE(abs(count_greater-count_lesser),2*sqrt(count));
      EXPECT_LE(count, 1.2);
      EXPECT_GE(count, 0.8);
    }
  }

  TYPED_TEST_CASE(NoiseLayerTest, TestDtypesAndDevices);

  TYPED_TEST(NoiseLayerTest, TestSetup) {
    typedef typename TypeParam::Dtype Dtype;
    // setup
    LayerParameter layer_param;
    NoiseParameter* noise_param = layer_param.mutable_noise_param();
    noise_param->set_sigma(1);
    NoiseLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
    EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
    EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
    EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
  }

  TYPED_TEST(PoolingLayerTest, TestForward) {
    this->blob_top_vec_[0] = this->blob_top_;
    this->TestForward();
  }

  TYPED_TEST(PoolingLayerTest, TestForwardInPlace) {
    this->blob_top_vec_[0] = this->blob_bottom_;
    this->TestForward();
  }

  TYPED_TEST(PoolingLayerTest, TestGradientEltwise) {
     // setup
    this->blob_top_vec_[0] = this->blob_top_;
    LayerParameter layer_param;
    NoiseParameter* noise_param = layer_param.mutable_noise_param();
    noise_param->set_sigma(1);
    NoiseLayer<Dtype> layer(layer_param);
    // check gradient
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
  }

    TYPED_TEST(PoolingLayerTest, TestGradientEltwiseInPlace) {
     // setup
    this->blob_top_vec_[0] = this->blob_bottom_;
    LayerParameter layer_param;
    NoiseParameter* noise_param = layer_param.mutable_noise_param();
    noise_param->set_sigma(1);
    // check gradient
    NoiseLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
  }
}

