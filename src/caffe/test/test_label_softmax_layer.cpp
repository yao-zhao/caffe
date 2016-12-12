#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/label_softmax_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class LabelSoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LabelSoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 1, 2, 1)),
        blob_top_(new Blob<Dtype>()) {
    for (int i = 0; i < 4; ++i) {
      blob_bottom_->mutable_cpu_data()[i] = i % 3;
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LabelSoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LabelSoftmaxLayerTest, TestDtypesAndDevices);

TYPED_TEST(LabelSoftmaxLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_label_softmax_param()->set_axis(1);
  layer_param.mutable_label_softmax_param()->set_num_classes(3);
  LabelSoftmaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1),
      layer_param.label_softmax_param().num_classes());
  EXPECT_EQ(this->blob_top_->shape(2), this->blob_bottom_->shape(2));
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}


TYPED_TEST(LabelSoftmaxLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_label_softmax_param()->set_axis(1);
  layer_param.mutable_label_softmax_param()->set_num_classes(3);
  LabelSoftmaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  EXPECT_EQ(top_data[0], Dtype(1));
  EXPECT_EQ(top_data[1], Dtype(0));
  EXPECT_EQ(top_data[2], Dtype(0));
  EXPECT_EQ(top_data[3], Dtype(1));
  EXPECT_EQ(top_data[4], Dtype(0));
  EXPECT_EQ(top_data[5], Dtype(0));
  EXPECT_EQ(top_data[6], Dtype(0));
  EXPECT_EQ(top_data[7], Dtype(1));
  EXPECT_EQ(top_data[8], Dtype(0));
  EXPECT_EQ(top_data[9], Dtype(0));
  EXPECT_EQ(top_data[10], Dtype(1));
  EXPECT_EQ(top_data[11], Dtype(0));
}

}  // namespace caffe
