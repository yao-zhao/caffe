#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_decay_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxWithDecayLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxWithDecayLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(2, 3, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(2, 1, 2, 3)),
        blob_top_data_(new Blob<Dtype>(2, 3, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(2);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = 1;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    blob_top_vec_2_.push_back(blob_top_data_);
  }
  virtual ~SoftmaxWithDecayLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_vec_2_;
};

TYPED_TEST_CASE(SoftmaxWithDecayLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxWithDecayLossLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithDecayLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}


TYPED_TEST(SoftmaxWithDecayLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxWithDecayLossLayer<Dtype> layer(layer_param);
  FillerParameter filler_param;
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype computed_loss = this->blob_top_loss_->cpu_data()[0];
  // SoftmaxWithDecayLossLayer<Dtype> layer2(layer_param);
  // layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_2_);
  // layer2.Forward(this->blob_bottom_vec_, this->blob_top_vec_2_);
  // const Dtype* label_data = this->blob_bottom_data_->cpu_data();
  // const Dtype* softmax_data = this->blob_top_data_->cpu_data();
  // Dtype accum_loss = 0;
  // int inner_dim = this->blob_bottom_data_->shape(2) *
  //     this->blob_bottom_data_->shape(3);
  // int softmax_dim = this->blob_bottom_data_->shape(1);
  // for (int i = 0; i < this->blob_bottom_data_->count(); ++i) {
  //   const int batch_index = i / (inner_dim * softmax_dim);
  //   const int label_index = (i / inner_dim) % softmax_dim;
  //   const int spatial_index = i % inner_dim;
  //   const Dtype diff = Dtype(label_index)
  //     - label_data[batch_index * inner_dim + spatial_index];
  //   accum_loss += 1/Dtype(softmax_dim) *
  //       (- log(std::max(softmax_data[i], Dtype(FLT_MIN))));
  // }
  // accum_loss /= this->blob_bottom_data_->count()/softmax_dim;
  Dtype accum_loss = - log(1./3.);
  EXPECT_NEAR(computed_loss, accum_loss, 1e-6);
}


TYPED_TEST(SoftmaxWithDecayLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithDecayLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
