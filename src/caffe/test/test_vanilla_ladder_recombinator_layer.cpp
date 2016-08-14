#include <math.h>

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/vanilla_ladder_combinator_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

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
    blob_top_(new Blob<Dtype>()),
    gradient_computed_(new Blob<Dtype>()),
    gradient_estimated_(new Blob<Dtype>()),
    objective_positive_(new Blob<Dtype>()),
    objective_negative_(new Blob<Dtype>()) {}
  // set up
  virtual void SetUp() {
    blob_bottom_z_->Reshape(4, 3, 5, 6);
    blob_bottom_u_->Reshape(4, 3, 5, 6);
    gradient_computed_->ReshapeLike(*blob_bottom_z_);
    gradient_estimated_->ReshapeLike(*blob_bottom_z_);
    objective_positive_->ReshapeLike(*blob_bottom_z_);
    objective_negative_->ReshapeLike(*blob_bottom_z_);
    Dtype* blob_bottom_z_data = blob_bottom_z_->mutable_cpu_data();
    Dtype* blob_bottom_u_data = blob_bottom_u_->mutable_cpu_data();
    // set clean path and reconstruct path
    for (int i = 0; i < blob_bottom_z_->count(); ++i) {
      blob_bottom_z_data[i] = Dtype(i)/100+0.2;
      blob_bottom_u_data[i] = Dtype(i)/100+0.25;
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
    delete blob_top_;
    delete gradient_computed_;
    delete gradient_estimated_;
    delete objective_positive_;
    delete objective_negative_;
  }
  // members
  Blob<Dtype>* const blob_bottom_z_;
  Blob<Dtype>* const blob_bottom_u_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const gradient_computed_;
  Blob<Dtype>* const gradient_estimated_;
  Blob<Dtype>* const objective_positive_;
  Blob<Dtype>* const objective_negative_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(VanillaLadderCombinatorLayerTest, TestDtypesAndDevices);

TYPED_TEST(VanillaLadderCombinatorLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  VanillaLadderCombinatorLayer<Dtype> layer(layer_param);
  this->blob_top_vec_[0] = this->blob_top_;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->num(),
    this->blob_bottom_vec_[0]->num());
  EXPECT_EQ(this->blob_top_vec_[0]->channels(),
    this->blob_bottom_vec_[0]->channels());
  EXPECT_EQ(this->blob_top_vec_[0]->height(),
    this->blob_bottom_vec_[0]->height());
  EXPECT_EQ(this->blob_top_vec_[0]->width(),
    this->blob_bottom_vec_[0]->width());
}

TYPED_TEST(VanillaLadderCombinatorLayerTest, TestForwardDefault) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  VanillaLadderCombinatorLayer<Dtype> layer(layer_param);
  this->blob_top_vec_[0] = this->blob_top_;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_vec_[0]->cpu_data();
  Dtype z;
  Dtype u;
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    z = Dtype(i)/100 + 0.2;
    EXPECT_NEAR(top_data[i], z+sigmoid(z), 1e-5);
  }
}

TYPED_TEST(VanillaLadderCombinatorLayerTest, TestForwardRandom) {
  typedef typename TypeParam::Dtype Dtype;
  // setup
  LayerParameter layer_param;
  VanillaLadderCombinatorLayer<Dtype> layer(layer_param);
  this->blob_top_vec_[0] = this->blob_top_;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // filler
  vector<shared_ptr<Blob<Dtype> > > blobs = layer.blobs();
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  Caffe::set_random_seed(1701);
  for (int i = 0; i < 9; ++i) {
    filler.Fill(blobs[i].get());
  }
  // forward
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // check result
  const Dtype* top_data = this->blob_top_vec_[0]->cpu_data();
  Dtype z, u, p;
  int iw;
  int inner_num = this->blob_top_vec_[0]->count(1);
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    z = Dtype(i)/100 + 0.2;
    u = Dtype(i)/100 + 0.25;
    iw = i % inner_num;
    p =  blobs[0]->cpu_data()[iw] + blobs[1]->cpu_data()[iw]*z +
      blobs[2]->cpu_data()[iw]*u + blobs[3]->cpu_data()[iw]*u*z +
      blobs[4]->cpu_data()[iw] * sigmoid(
        blobs[5]->cpu_data()[iw] + blobs[6]->cpu_data()[iw]*z +
        blobs[7]->cpu_data()[iw]*u + blobs[8]->cpu_data()[iw]*u*z);
    EXPECT_NEAR(top_data[i], p, 1e-5);
  }
}

TYPED_TEST(VanillaLadderCombinatorLayerTest, TestBottomGradient) {
  typedef typename TypeParam::Dtype Dtype;
  // setup
  LayerParameter layer_param;
  VanillaLadderCombinatorLayer<Dtype> layer(layer_param);
  this->blob_top_vec_[0] = this->blob_top_;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // filler
  vector<shared_ptr<Blob<Dtype> > > blobs = layer.blobs();
  FillerParameter filler_param;
  GaussianFiller<Dtype> gaussianfiller(filler_param);
  Caffe::set_random_seed(1701);
  for (int i = 0; i < 9; ++i) {
    gaussianfiller.Fill(blobs[i].get());
  }
  // check gradient, only test agains bottom 0 and 1
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> propagate_down(2, true);
  int count = this->blob_bottom_z_->count();
  caffe_set<Dtype>(count, Dtype(1), this->blob_top_vec_[0]->mutable_cpu_diff());
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  const Dtype* gradient_computed_z = this->blob_bottom_z_->cpu_diff();
  const Dtype* gradient_computed_u = this->blob_bottom_u_->cpu_diff();

  // estimated backward
  Dtype step_size = 1e-2;
  Dtype scale;
  Dtype threshold = 1e-2;
  for (int i = 0; i < 2; ++i) {
    // increase bot
    caffe_add_scalar(count, step_size,
      this->blob_bottom_vec_[i]->mutable_cpu_data());
    this->blob_top_vec_[0] = this->objective_positive_;
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // decrease bot
    caffe_add_scalar(count, step_size*Dtype(-2),
      this->blob_bottom_vec_[i]->mutable_cpu_data());
    this->blob_top_vec_[0] = this->objective_negative_;
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // reset input
    caffe_add_scalar(count, step_size,
      this->blob_bottom_vec_[i]->mutable_cpu_data());
    // estimated gradient
    Dtype* gradient_estimated_data =
      this->gradient_estimated_->mutable_cpu_data();
    caffe_sub(count, this->objective_positive_->cpu_data(),
      this->objective_negative_->cpu_data(), gradient_estimated_data);
    caffe_scal(count, Dtype(0.5)/step_size, gradient_estimated_data);
    // computed gradient
    const Dtype* gradient_computed_data;
    if (i == 0) {
      gradient_computed_data = gradient_computed_z;
    } else {
      gradient_computed_data = gradient_computed_u;
    }
    // check estimated and computed
    for (int j = 0; j < count; ++j) {
      scale = std::max<Dtype>(Dtype(0.1), std::max(
        fabs(gradient_estimated_data[j]), fabs(gradient_computed_data[j])));
      EXPECT_NEAR(gradient_estimated_data[j], gradient_computed_data[j],
        threshold * scale);
    }
  }
  this->blob_top_vec_[0] = this->blob_top_;
}

TYPED_TEST(VanillaLadderCombinatorLayerTest, TestWeightGradient) {
  typedef typename TypeParam::Dtype Dtype;
  // setup
  LayerParameter layer_param;
  VanillaLadderCombinatorLayer<Dtype> layer(layer_param);
  this->blob_top_vec_[0] = this->blob_top_;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  int count = this->blob_bottom_z_->count();
  // filler
  vector<shared_ptr<Blob<Dtype> > > blobs = layer.blobs();
  FillerParameter filler_param;
  GaussianFiller<Dtype> gaussianfiller(filler_param);
  Caffe::set_random_seed(1701);
  for (int i = 0; i < 9; ++i) {
    gaussianfiller.Fill(blobs[i].get());
  }
  // set top diff
  // calculate top forward first, need top data and
  // also need to set intermedia variables for backward
  // because backward depends on forward
  this->blob_top_vec_[0] = this->blob_top_;
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // set top diff
  caffe_cpu_scale(count, Dtype(2), this->blob_top_->cpu_data(),
    this->blob_top_->mutable_cpu_diff());
  // reset weight diff
  // blobs[iblob]->mutable_cpu_diff()[j] = 0;
  // finally do back prop
  vector<bool> propagate_down(2, true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  // computed gradient, have to recalculate blobs at each iter
  Dtype step_size = 1e-2;
  Dtype scale;
  Dtype threshold = 1e-2;
  Dtype obj_positive, obj_negative;
  for (int iblob = 0; iblob < 9; ++iblob) {
    for (int j = 0; j < blobs[iblob]->count(); ++j) {
      // increase
      blobs[iblob]->mutable_cpu_data()[j] += step_size;
      this->blob_top_vec_[0] = this->objective_positive_;
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      // decrease
      blobs[iblob]->mutable_cpu_data()[j] -= 2*step_size;
      this->blob_top_vec_[0] = this->objective_negative_;
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      // restore
      blobs[iblob]->mutable_cpu_data()[j] += step_size;
      // dont use dot, will cause large error
      // calculate estimated gradient
      // Dtype gradient_estimated_data = caffe_cpu_dot(count,
      //   this->objective_positive_->cpu_data(),
      //     this->objective_positive_->cpu_data()) -
      // caffe_cpu_dot(count, this->objective_negative_->cpu_data(),
      //   this->objective_negative_->cpu_data()) / (2 * step_size);
      // caffe_sub(count, this->objective_positive_->cpu_data(),
      // this->objective_negative_->cpu_data(),
      //   this->objective_negative_->mutable_cpu_data());
      // Dtype gradient_estimated_data = caffe_cpu_dot(count,
      //   this->objective_positive_->cpu_data(),
      //     this->objective_negative_->cpu_data())/
      //    (2 * step_size);
      Dtype gradient_estimated_data = 0;
      const Dtype* positive_data = this->objective_positive_->cpu_data();
      const Dtype* negative_data = this->objective_negative_->cpu_data();
      for (int k = 0; k < count; ++k) {
        gradient_estimated_data += positive_data[k]*positive_data[k] -
        negative_data[k]*negative_data[k];
      }
      gradient_estimated_data /= 2*step_size;
      Dtype gradient_computed_data = blobs[iblob]->cpu_diff()[j];
      // compare
      scale = std::max<Dtype>(Dtype(0.1), std::max(
        fabs(gradient_estimated_data), fabs(gradient_computed_data)));
      EXPECT_NEAR(gradient_estimated_data, gradient_computed_data,
        threshold*scale);
    }
  }
}

}  // namespace caffe
