#include <algorithm>
#include <vector>
#include <math.h>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/vanilla_ladder_combinator_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include <iostream>

namespace caffe {
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename TypeParam>
class VanillaLadderCombinatorLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  // constructor
  VanillaLadderCombinatorLayerTest()
  : blob_bottom_z_(new Blob<Dtype>()),
  	blob_bottom_u_(new Blob<Dtype>()),
    blob_top_(new Blob<Dtype>()) {}
    // blob_tmp_(new Blob<Dtype>()),
  // set up
  virtual void SetUp() {
    blob_bottom_z_->Reshape(4,3,5,6);
    blob_bottom_u_->Reshape(4,3,5,6);
    Dtype* blob_bottom_z_data = blob_bottom_z_->mutable_cpu_data();
    Dtype* blob_bottom_u_data = blob_bottom_u_->mutable_cpu_data();
    // set clean path and reconstruct path
    for (int i=0; i<blob_bottom_z_->count(); ++i) {
    	blob_bottom_z_data[i]= Dtype(i)/100 + 0.2;
      blob_bottom_u_data[i]= Dtype(i)/100 + 0.25;
    }
    // push to vec 
    blob_bottom_vec_.push_back(blob_bottom_z_);
    blob_bottom_vec_.push_back(blob_bottom_u_);
    blob_top_vec_.push_back(blob_top_);
  }
  // destructor
  virtual ~VanillaLadderCombinatorLayerTest() {
    delete blob_bottom_z_;
    delete blob_bottom_u_;
    // delete blob_tmp_;
    delete blob_top_;
  }
  // members
  Blob<Dtype>* const blob_bottom_z_;
  Blob<Dtype>* const blob_bottom_u_;
  // Blob<Dtype>* const blob_tmp_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(VanillaLadderCombinatorLayerTest, TestDtypesAndDevices);

TYPED_TEST(VanillaLadderCombinatorLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  VanillaLadderCombinatorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(), this->blob_bottom_vec_[0]->num());
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), this->blob_bottom_vec_[0]->channels());
  EXPECT_EQ(this->blob_top_vec_[0]->height(), this->blob_bottom_vec_[0]->height());
  EXPECT_EQ(this->blob_top_vec_[0]->width(), this->blob_bottom_vec_[0]->width());
}

TYPED_TEST(VanillaLadderCombinatorLayerTest, TestForwardDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  VanillaLadderCombinatorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_vec_[0]->cpu_data();
  Dtype z;
  Dtype u;
  for (int i=0; i< this->blob_top_vec_[0]->count(); ++i) {
    z= Dtype(i)/100 + 0.2;
    EXPECT_NEAR(top_data[i], z+sigmoid(z), 1e-5);
  }
}

TYPED_TEST(VanillaLadderCombinatorLayerTest, TestForwardRandom) {
  typedef typename TypeParam::Dtype Dtype;
  // setup 
  LayerParameter layer_param;
  VanillaLadderCombinatorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // filler
  vector<shared_ptr<Blob<Dtype> > > blobs = layer.blobs();
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i=0; i<9; ++i) {
    filler.Fill(&*blobs[i]);
  }
  // forward
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // check result
  const Dtype* top_data = this->blob_top_vec_[0]->cpu_data();
  Dtype z, u, p;
  int iw;
  int inner_num = this->blob_top_vec_[0]->count(1);
  for (int i=0; i< this->blob_top_vec_[0]->count(); ++i) {
    z= Dtype(i)/100 + 0.2;
    u= Dtype(i)/100 + 0.25;
    iw = i % inner_num;
    p =  blobs[0]->cpu_data()[iw] + blobs[1]->cpu_data()[iw]*z + 
      blobs[2]->cpu_data()[iw]*u + blobs[3]->cpu_data()[iw]*u*z + 
      blobs[4]->cpu_data()[iw] * sigmoid(
        blobs[5]->cpu_data()[iw] + blobs[6]->cpu_data()[iw]*z +
        blobs[7]->cpu_data()[iw]*u + blobs[8]->cpu_data()[iw]*u*z) ;
    EXPECT_NEAR(top_data[i], p, 1e-5);
  }
}


// TYPED_TEST(VanillaLadderCombinatorLayerTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//    // setup
//   LayerParameter layer_param;
//   LadderLossLayer<Dtype> layer(layer_param);
//   // check gradient, only test agains bottom 0 and 1
//   GradientChecker<Dtype> checker(1e-2, 1e-2);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//     this->blob_top_vec_, 0);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//   	this->blob_top_vec_, 1);
// }

}