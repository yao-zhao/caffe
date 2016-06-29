#include <algorithm>
#include <vector>
#include <math.h>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/ladder_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include <iostream>

namespace caffe {

template <typename TypeParam>
class LadderLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  // constructor
  LadderLossLayerTest()
  : blob_bottom_clean_(new Blob<Dtype>()),
  	blob_bottom_recon_(new Blob<Dtype>()),
  	blob_bottom_param_(new Blob<Dtype>()),
    blob_top_(new Blob<Dtype>()) {}
  // set up
  virtual void SetUp() {
    int index=0;
    blob_bottom_clean_->Reshape(16,4,5,6);
    blob_bottom_recon_->Reshape(16,4,5,6);
    vector<int> sz;
    sz.push_back(2);
    sz.push_back(4);
    blob_bottom_param_->Reshape(sz);
    vector<int> shape = blob_bottom_clean_->shape();
    Dtype* blob_bottom_clean_data = blob_bottom_clean_->mutable_cpu_data();
    Dtype* blob_bottom_recon_data = blob_bottom_recon_->mutable_cpu_data();
    Dtype* blob_bottom_param_data = blob_bottom_param_->mutable_cpu_data();
    // set clean path and reconstruct path
    for (int k=0; k<shape[0]; ++k) {
    	for (int c=0; c<shape[1]; ++c) {
    		for (int h=0; h<shape[2]; ++h) {
    			for (int w=0; w<shape[3]; ++w) {
    				index = w+shape[3]*(h+shape[2]*(c+shape[1]*k));
    				blob_bottom_clean_data[index] = index;
    				blob_bottom_recon_data[index] = 
    					(10+c)*(index + 2*(index%2)-1) + (15+c);
    			}
    		}
    	}
    }
    // set batch norm param
    for (int i=0; i<shape[1]; ++i) {
    	blob_bottom_param_data[i] = (15+i);
    	blob_bottom_param_data[i+shape[1]] = 10+i;
    }
    // push to vec 
    blob_bottom_vec_.push_back(blob_bottom_clean_);
    blob_bottom_vec_.push_back(blob_bottom_recon_);
    blob_bottom_vec_.push_back(blob_bottom_param_);
    blob_top_vec_.push_back(blob_top_);
  }
  // destructor
  virtual ~LadderLossLayerTest() {
    delete blob_bottom_clean_;
    delete blob_bottom_recon_;
    delete blob_bottom_param_;
    delete blob_top_;
  }
  // members
  Blob<Dtype>* const blob_bottom_clean_;
  Blob<Dtype>* const blob_bottom_recon_;
  Blob<Dtype>* const blob_bottom_param_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LadderLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(LadderLossLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LadderLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->count(),1);
}

TYPED_TEST(LadderLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LadderLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype loss = layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_NEAR(loss, 0.5, 1e-12);
}

TYPED_TEST(LadderLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
   // setup
  LayerParameter layer_param;
  LadderLossLayer<Dtype> layer(layer_param);
  // check gradient, only test agains bottom 0 and 1
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  	this->blob_top_vec_, 1);
}

}