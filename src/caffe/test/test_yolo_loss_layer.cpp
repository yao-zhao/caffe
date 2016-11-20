#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/yolo_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class YoloLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  YoloLossLayerTest()
      : blob_bottom_box_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_bottom_class_(new Blob<Dtype>()),
        blob_top_loss_(new Blob<Dtype>()) {
    // for (int i = 0; i < blob_bottom_label_->count(); ++i) {
    //   blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    // }
    // for (int i = 0; i < blob_bottom_label_->count(); ++i) {
    //   blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    // }
    // blob_bottom_vec_.push_back(blob_bottom_box_);
    // blob_bottom_vec_.push_back(blob_bottom_label_);
    // blob_top_vec_.push_back(blob_top_loss_);
  }

  void ManualSetup() {
    int box_dim[] = {1, 3, 3, 1, 5};
    blob_bottom_box_->Reshape(*(new vector<int>(box_dim, box_dim+5)));
    Dtype input1[] = {.5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                     };
    for (int i = 0; i < blob_bottom_box_->count(); ++i) {
      blob_bottom_box_->mutable_cpu_data()[i] = input1[i];
    }
    int label_dim[] = {1, 2, 5};
    blob_bottom_label_->Reshape(*(new vector<int>(label_dim, label_dim+3)));
    Dtype input2[] = {2/3., 2/3., 2/3., 2/3., 0,
                      .5, .5, .5, .5, -1,
                     };
    blob_bottom_vec_.clear();
    for (int i = 0; i < blob_bottom_box_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = input2[i];
    }
    blob_bottom_vec_.push_back(blob_bottom_box_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.clear();
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~YoloLossLayerTest() {
    delete blob_bottom_box_;
    delete blob_bottom_label_;
    delete blob_bottom_class_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_box_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_class_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(YoloLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(YoloLossLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  YoloLossLayer<Dtype> layer(layer_param);
  this->ManualSetup();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

// TYPED_TEST(YoloLossLayerTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   SoftmaxWithLossLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_, 0);
// }

// TYPED_TEST(YoloLossLayerTest, TestForwardIgnoreLabel) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   layer_param.mutable_loss_param()->set_normalize(false);
//   // First, compute the loss with all labels
//   scoped_ptr<SoftmaxWithLossLayer<Dtype> > layer(
//       new SoftmaxWithLossLayer<Dtype>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
//   // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
//   Dtype accum_loss = 0;
//   for (int label = 0; label < 5; ++label) {
//     layer_param.mutable_loss_param()->set_ignore_label(label);
//     layer.reset(new SoftmaxWithLossLayer<Dtype>(layer_param));
//     layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//     layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//     accum_loss += this->blob_top_loss_->cpu_data()[0];
//   }
//   // Check that each label was included all but once.
//   EXPECT_NEAR(4 * full_loss, accum_loss, 1e-4);
// }

// TYPED_TEST(YoloLossLayerTest, TestGradientIgnoreLabel) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
//   layer_param.mutable_loss_param()->set_ignore_label(0);
//   SoftmaxWithLossLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_, 0);
// }

// TYPED_TEST(YoloLossLayerTest, TestGradientUnnormalized) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   layer_param.mutable_loss_param()->set_normalize(false);
//   SoftmaxWithLossLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_, 0);
// }

}  // namespace caffe
