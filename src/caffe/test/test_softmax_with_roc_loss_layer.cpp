#include <cmath>
#include <vector>
#include <iostream>
using namespace std;

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_roc_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxWithROCLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxWithROCLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(64, 2, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(64, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  void SetAllPositive() {
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = 1;
    }
  }
  virtual ~SoftmaxWithROCLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxWithROCLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxWithROCLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Dtype eps = 0.2;
  layer_param.mutable_softmax_roc_loss_param()->set_eps(eps);
  SoftmaxWithROCLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<Dtype> positive, negative;
  int count = this->blob_bottom_label_->count();
  const Dtype* label_data = this->blob_bottom_label_->cpu_data();
  const Dtype* prob_data = this->blob_bottom_data_->cpu_data();
  for (int i = 0; i < count; ++i) {
    Dtype softmax_pos = exp(prob_data[2*i+1])/
        (exp(prob_data[2*i+1])+exp(prob_data[2*i]));
    Dtype softmax_neg = exp(prob_data[2*i])/
        (exp(prob_data[2*i+1])+exp(prob_data[2*i]));
    if (label_data[i] == 1) {
      positive.push_back(softmax_pos);
    } else if (label_data[i] == 0) {
      negative.push_back(softmax_neg);
    }
  }
  Dtype auc = 0;
  for (int i = 0; i < positive.size(); ++i) {
    for (int j = 0; j < negative.size(); ++j) {
      Dtype diff = positive[i]-negative[j];
      if (diff > eps) {
        auc += 1;
      } else if (diff > -eps) {
        auc += 0.5 + 0.5 * diff / eps;
      }
    }
  }
  Dtype loss = 1 - auc / positive.size() / negative.size();
  EXPECT_NEAR(this->blob_top_vec_[0]->cpu_data()[0], loss, 1e-7);
}

TYPED_TEST(SoftmaxWithROCLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  layer_param.mutable_softmax_roc_loss_param()->set_eps(0.2);
  SoftmaxWithROCLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithROCLossLayerTest, TestGradientAllPositive) {
  typedef typename TypeParam::Dtype Dtype;
  this->SetAllPositive();
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  layer_param.mutable_softmax_roc_loss_param()->set_eps(0.2);
  SoftmaxWithROCLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[0], 0.5);
  GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
