#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/image_box_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageBoxDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageBoxDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg "
        << EXAMPLES_SOURCE_DIR "bounding_box/label0.txt"<< std::endl;
    outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg "
        << EXAMPLES_SOURCE_DIR "bounding_box/label1.txt"<< std::endl;
    outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg "
        << EXAMPLES_SOURCE_DIR "bounding_box/label2.txt"<< std::endl;
    outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg "
        << EXAMPLES_SOURCE_DIR "bounding_box/label3.txt"<< std::endl;
    outfile.close();
  }

  virtual ~ImageBoxDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageBoxDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageBoxDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageBoxDataParameter* image_box_data_param =
      param.mutable_image_box_data_param();
  image_box_data_param->set_batch_size(4);
  image_box_data_param->set_source(this->filename_.c_str());
  image_box_data_param->set_shuffle(false);
  image_box_data_param->set_max_num_box(2);
  ImageBoxDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 4);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 4);
  EXPECT_EQ(this->blob_top_label_->channels(), 2);
  EXPECT_EQ(this->blob_top_label_->height(), 5);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* label_data = this->blob_top_label_->cpu_data();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (j < i) {
        EXPECT_NEAR(label_data[0], Dtype(i)*0.1, 1e-6);
        EXPECT_NEAR(label_data[1], Dtype(i)*0.1, 1e-6);
        EXPECT_NEAR(label_data[2], Dtype(i)*0.2, 1e-6);
        EXPECT_NEAR(label_data[3], Dtype(i)*0.2, 1e-6);
        EXPECT_EQ(label_data[4], Dtype(j));
      } else {
        EXPECT_EQ(label_data[4], Dtype(-1));
      }
      label_data += 5;
    }
  }
};

}  // namespace caffe
#endif  // USE_OPENCV
