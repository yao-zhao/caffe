#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
// #include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lorentzian_prob_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class LorentzianProbLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LorentzianProbLossLayerTest()
      : blob_bottom_mean_(new Blob<Dtype>(4, 3, 2, 2)),
        blob_bottom_var_(new Blob<Dtype>(4, 3, 2, 2)),
        blob_bottom_label_(new Blob<Dtype>(4, 3, 2, 2)),
        blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    blob_bottom_vec_.push_back(blob_bottom_mean_);
    blob_bottom_vec_.push_back(blob_bottom_var_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    eps_ = 0.1;
  }

  void SetUpEasy() {
    this->blob_bottom_mean_->Reshape(4, 1, 1, 1);
    this->blob_bottom_var_->Reshape(4, 1, 1, 1);
    this->blob_bottom_label_->Reshape(4, 1, 1, 1);
    Dtype mean[] = {-0.2, 0.1, 0.2, 0.3 };
    Dtype label[] = {-0.1, 0.2, 0.3, 0.2 };
    Dtype var[] = {0.1, 0.1, 0.1, 0.1};
    for (int i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
      this->blob_bottom_vec_[0]->mutable_cpu_data()[i] = mean[i];
      this->blob_bottom_vec_[1]->mutable_cpu_data()[i] = var[i];
      this->blob_bottom_vec_[2]->mutable_cpu_data()[i] = label[i];
    }
  }

  void SetUpRandom() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_mean_);
    FillerParameter filler_param2;
    filler_param.set_min(0);
    filler_param.set_max(0.4);
    UniformFiller<Dtype> filler2(filler_param);
    filler2.Fill(this->blob_bottom_var_);
    filler.Fill(this->blob_bottom_label_);
  }

  virtual Dtype getLoss() {
    const Dtype* mean = this->blob_bottom_vec_[0]->cpu_data();
    const Dtype* var = this->blob_bottom_vec_[1]->cpu_data();
    const Dtype* label = this->blob_bottom_vec_[2]->cpu_data();
    const int count =this->blob_bottom_vec_[0]->count();
    Dtype loss = 0;
    Dtype tmp = 0;
    Dtype tmp2 = 0;
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(var[i], 0);
      tmp = mean[i]-label[i];
      tmp = tmp * tmp;
      tmp2 = Dtype(this->eps_)+var[i];
      EXPECT_GT(var[i], 0);
      loss += log(tmp/tmp2+1)+log(tmp2)/2;
    }
    loss /= Dtype(count);
    loss += Dtype(log(3.1415926));
    return loss;
  }

  virtual ~LorentzianProbLossLayerTest() {
    delete blob_bottom_mean_;
    delete blob_bottom_var_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_mean_;
  Blob<Dtype>* const blob_bottom_var_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  float eps_;
};

TYPED_TEST_CASE(LorentzianProbLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(LorentzianProbLossLayerTest, TestSetUpEasy) {
  typedef typename TypeParam::Dtype Dtype;
  this->SetUpEasy();
  LayerParameter layer_param;
  LorentzianProbLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->count(), 1);
}

TYPED_TEST(LorentzianProbLossLayerTest, TestSetUpRandom) {
  typedef typename TypeParam::Dtype Dtype;
  this->SetUpRandom();
  LayerParameter layer_param;
  LorentzianProbLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->count(), 1);
}

TYPED_TEST(LorentzianProbLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->SetUpEasy();
  LayerParameter layer_param;
  layer_param.mutable_lorentzian_prob_loss_param()->set_eps(this->eps_);
  LorentzianProbLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype loss = this->getLoss();
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_NEAR(loss, 0.3888, 1e-4);
  EXPECT_NEAR(this->blob_top_vec_[0]->cpu_data()[0], loss, 1e-5);
}

TYPED_TEST(LorentzianProbLossLayerTest, TestForward2) {
  typedef typename TypeParam::Dtype Dtype;
  this->SetUpRandom();
  LayerParameter layer_param;
  LorentzianProbLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->eps_ = layer_param.mutable_lorentzian_prob_loss_param()->eps();
  Dtype loss = this->getLoss();
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_NEAR(this->blob_top_vec_[0]->cpu_data()[0],
    loss, 1e-5);
}

TYPED_TEST(LorentzianProbLossLayerTest, TestGradientEasy) {
  typedef typename TypeParam::Dtype Dtype;
  this->SetUpEasy();
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LorentzianProbLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-4, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LorentzianProbLossLayerTest, TestGradientRandom) {
  typedef typename TypeParam::Dtype Dtype;
  this->SetUpRandom();
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LorentzianProbLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


}  // namespace caffe
