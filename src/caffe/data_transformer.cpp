#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <math.h>

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/image_transformations.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check crop parameter
  CHECK(param_.crop_size() == 0 ||
        (param_.crop_h() == 0 && param_.crop_w() == 0))
        << "Crop size is crop_size OR crop_h and crop_w; not both";
  CHECK((param_.crop_h() != 0) == (param_.crop_w() != 0))
        << "For non-square crops both crop_h and crop_w are required.";
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();

  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_uint8 = data.size() > 0;
  const Dtype scale = GetScale();

  // use advanced transformation if USE_OPENCV
  bool convert_to_cv = false;
#ifdef USE_OPENCV
  cv::Mat cv_img = DatumToCVMat(datum);
  PeriodicResize(&cv_img, &datum_height, &datum_width);
  GeometricalTransform(&cv_img);
  convert_to_cv = true;
#endif  // USE_OPENCV
#ifndef USE_OPENCV
  NoAdvancedTransformations();
#endif  // USE_OPENCV

  // get post crop height and width
  int height, width;
  GetPostCropSize(&height, &width, datum_height, datum_width);
  CHECK_GT(datum_channels, 0);

  // get mean
  GetMean(datum_channels, datum_height, datum_width);

  // get offset
  int h_off, w_off;
  GetOffset(&h_off, &w_off, height, width, datum_height, datum_width);

  // assign values
  const Dtype* mean = param_.has_mean_file() ?
      data_mean_.mutable_cpu_data() : NULL;
  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c*datum_height+h_off+h)*datum_width+w_off+w;
        if (do_mirror) {
          top_index = (c*height+h)*width+(width-1-w);
        } else {
          top_index = (c*height+h)*width+w;
        }
        if (has_uint8) {
          if (convert_to_cv) {
#ifdef USE_OPENCV
            switch (datum_channels) {
              case 1:
                datum_element =
                static_cast<Dtype>(cv_img.at<uchar>(h+h_off, w+w_off));
                break;
              case 3:
                datum_element =
                static_cast<Dtype>(cv_img.at<cv::Vec3b>(h+h_off, w+w_off)[c]);
                break;
              default:
                CHECK(0) << "wrong number of channels";
            }
#endif  // USE_OPENCV
          } else {
            datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
          }
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (param_.has_mean_file()) {
          transformed_data[top_index] =
            (datum_element-mean[data_index])*scale;
        } else {
          if (mean_values_.size() > 0) {
            transformed_data[top_index] =
              (datum_element-mean_values_[c])*scale;
          } else {
            transformed_data[top_index] = datum_element*scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  // get crop
  int crop_h, crop_w;
  GetPostCropSize(&crop_h, &crop_w, datum_height, datum_width);
  CHECK_GT(datum_channels, 0);
  CHECK_EQ(crop_h, height);
  CHECK_EQ(crop_w, width);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data()+offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data()+offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = GetScale();
  const bool do_mirror = param_.mirror() && Rand(2);

  // advanced transformations
  cv::Mat transformed_img = cv_img;
  PeriodicResize(&transformed_img, &img_height, &img_width);
  GeometricalTransform(&transformed_img);

  // get crop
  int crop_h, crop_w;
  bool has_crop = GetPostCropSize(&crop_h, &crop_w, img_height, img_width);
  CHECK_GT(img_channels, 0);

  // get mean
  GetMean(img_channels, img_height, img_width);

  // get offset
  int h_off, w_off;
  GetOffset(&h_off, &w_off, crop_h, crop_w, img_height, img_width);
  CHECK_EQ(crop_h, height);
  CHECK_EQ(crop_w, width);
  if (has_crop) {
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    transformed_img = cv_img(roi);
  }

  CHECK(transformed_img.data);

  const Dtype* mean = param_.has_mean_file() ?
      data_mean_.mutable_cpu_data() : NULL;
  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = transformed_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c*height+h)*width+(width-1-w);
        } else {
          top_index = (c*height+h)*width+w;
        }
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (param_.has_mean_file()) {
          int mean_index = (c*img_height+h_off+h)*img_width+w_off+w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (mean_values_.size() > 0) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel*scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob_img,
                                       const cv::Mat& cv_lab,
                                       Blob<Dtype>* transformed_blob_lab) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  const int lab_channels =  cv_lab.channels();

  NoAdvancedTransformations();

  CHECK_EQ(cv_lab.channels(), 1) << "Can only handle grayscale label images";
  CHECK(cv_lab.rows == img_height && cv_lab.cols == img_width) <<
    "Input and label image heights and widths must match";
  // Check dimensions.
  const int channels = transformed_blob_img->channels();
  const int height = transformed_blob_img->height();
  const int width = transformed_blob_img->width();
  const int num = transformed_blob_img->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  const Dtype scale = GetScale();
  const bool do_mirror = param_.mirror() && Rand(2);

  // get crop
  int crop_h, crop_w;
  bool has_crop = GetPostCropSize(&crop_h, &crop_w, img_height, img_width);
  CHECK_GT(img_channels, 0);

  // get mean
  GetMean(img_channels, img_height, img_width);

  // get offset
  int h_off, w_off;
  GetOffset(&h_off, &w_off, crop_h, crop_w, img_height, img_width);
  CHECK_EQ(crop_h, height);
  CHECK_EQ(crop_w, width);
  cv::Mat cv_cropped_img = cv_img;
  cv::Mat cv_cropped_label = cv_lab;
  if (has_crop) {
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_img(roi);
    cv_cropped_label = cv_lab(roi);
  }

  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_label.data);

  const Dtype* mean = param_.has_mean_file() ?
      data_mean_.mutable_cpu_data() : NULL;
  Dtype* transformed_data_img = transformed_blob_img->mutable_cpu_data();
  Dtype* transformed_data_lab = transformed_blob_lab->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr_img = cv_cropped_img.ptr<uchar>(h);
    const uchar* ptr_lab = cv_cropped_label.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c*height+h)*width+(width-1-w);
        } else {
          top_index = (c*height+h)*width+w;
        }
        Dtype pixel_img = static_cast<Dtype>(ptr_img[img_index]);
        img_index++;
        if (param_.has_mean_file()) {
          int mean_index = (c*img_height+h_off+h)*img_width+w_off+w;
          transformed_data_img[top_index] =
          (pixel_img-mean[mean_index])*scale;
        } else {
          if (mean_values_.size() > 0) {
            transformed_data_img[top_index] =
            (pixel_img-mean_values_[c])*scale;
          } else {
            transformed_data_img[top_index] = pixel_img*scale;
          }
        }
      }
    }
    img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < lab_channels; ++c) {
        if (do_mirror) {
          top_index = (c*height+h)*width+(width-1-w);
        } else {
          top_index = (c*height+h)*width+w;
        }
        Dtype pixel_lab = static_cast<Dtype>(ptr_lab[img_index]);
        img_index++;
        transformed_data_lab[top_index] = pixel_lab;
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  NoAdvancedTransformations();

  int crop_h, crop_w;
  GetPostCropSize(&crop_h, &crop_w, input_height, input_width);
  CHECK_GT(input_channels, 0);

  // Build BlobShape.
  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    transformed_blob->Reshape(
      input_num, input_channels, crop_h, crop_w);
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);

  const Dtype scale = GetScale();
  const bool do_mirror = param_.mirror() && Rand(2);

  // get offset
  int h_off, w_off;
  GetOffset(&h_off, &w_off, crop_h, crop_w, input_height, input_width);
  CHECK_EQ(crop_h, height);
  CHECK_EQ(crop_w, width);


  Dtype* input_data = input_blob->mutable_cpu_data();
  if (param_.has_mean_file()) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (mean_values_.size() > 0) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  CHECK_GT(datum_channels, 0);
  int crop_h, crop_w;
  GetPostCropSize(&crop_h, &crop_w, datum_height, datum_width);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = crop_h;
  shape[3] = crop_w;

  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  CHECK_GT(img_channels, 0);
  int crop_h, crop_w;
  GetPostCropSize(&crop_h, &crop_w, img_height, img_width);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = crop_h;
  shape[3] = crop_w;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  // const bool needs_crop = param_.crop_size() > 0 ||
  //     param_.crop_h() > 0 || param_.crop_w() > 0;
  // const bool needs_rand = param_.mirror() ||
  //     (phase_ == TRAIN && needs_crop);
  // if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  // } else {
  //   rng_.reset();
  // }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void DataTransformer<Dtype>::NoAdvancedTransformations() {
  CHECK_EQ(param_.rotation_range(), 0) << "does not support rotation";
  CHECK_EQ(param_.perspective_transformation_border(), 0) <<
      "does not support perspective transformations";
  CHECK_EQ(param_.scale_jitter_range(), 0) <<
      "does not support scale jitter";
  CHECK_EQ(param_.contrast_jitter_range(), 0) <<
      "does not support contrast jitter";
  CHECK_EQ(param_.periodic_resize(),
      TransformationParameter_PeriodicResizeMode_NONE) <<
      "does not support periodic resize";
  CHECK_EQ(param_.periodic_resize_h(), 0) <<
      "does not support periodic resize";
  CHECK_EQ(param_.periodic_resize_w(), 0) <<
      "does not support periodic resize";
}

template <typename Dtype>
bool DataTransformer<Dtype>::GetPostCropSize(int* crop_h, int* crop_w,
    const int height, const int width) {
  const int crop_size = param_.crop_size();
  *crop_h = param_.crop_h();
  *crop_w = param_.crop_w();
  if (crop_size > 0) {
    *crop_h = *crop_w = crop_size;
  }
  CHECK_GE(height, *crop_h);
  CHECK_GE(width, *crop_w);
  bool has_crop = (*crop_h > 0 || *crop_w > 0);
  if (*crop_h == 0) {
    *crop_h = height;
  }
  if (*crop_w == 0) {
    *crop_w = width;
  }
  return has_crop;
}

template <typename Dtype>
void DataTransformer<Dtype>::GetMean(const int channels,
      const int height, const int width) {
  if (param_.has_mean_file()) {
    CHECK_EQ(channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
  }
  if (mean_values_.size() > 0) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::GetOffset(int* h_off, int* w_off,
    const int crop_h, const int crop_w, const int height, const int width) {
  // We only do random crop when we do training.or have randome crop test
  CHECK_GE(height, crop_h);
  CHECK_GE(width, crop_w);
  if (phase_ == TRAIN || param_.random_crop_test()) {
    *h_off = Rand(height-crop_h+1);
    *w_off = Rand(width-crop_w+1);
  } else {
    *h_off = (height-crop_h)/2;
    *w_off = (width-crop_w)/2;
  }
}

template <typename Dtype>
Dtype DataTransformer<Dtype>::GetScale() {
  Dtype scale = param_.scale();
  float contrast_jitter_range = param_.contrast_jitter_range();
  CHECK_GE(contrast_jitter_range, 0);
  if (contrast_jitter_range > 0) {
    scale = scale*exp(contrast_jitter_range*
      static_cast<float>(Rand(201)-100)/200.0);
  }
  return scale;
}

#ifdef USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::GeometricalTransform(cv::Mat* cv_img) {
  const bool has_rotation = param_.rotation_range() > 0;
  const bool has_perspective_transformation =
    param_.perspective_transformation_border() > 0;
  const bool has_scale_jitter = param_.scale_jitter_range() > 0;
  if ((has_perspective_transformation || has_rotation || has_scale_jitter)) {
    // if has rotation or scale jitter
    if (has_rotation || has_scale_jitter) {
      const int rotation_range = param_.rotation_range();
      const float scale_jitter = exp(param_.scale_jitter_range()*
        static_cast<float>(Rand(201)-100)/200.0);
      RandomRotateImage(*cv_img, rotation_range, scale_jitter, cv_img);
    }
    // perspective transform
    if (has_perspective_transformation) {
      const int perspective_transformation_border =
          param_.perspective_transformation_border();
      RandomPerspectiveTransformImage(*cv_img,
          perspective_transformation_border, cv_img);
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::PeriodicResize(cv::Mat* cv_img,
    int* input_height, int* input_width) {
  if (param_.periodic_resize() !=
      TransformationParameter_PeriodicResizeMode_NONE) {
    int height = param_.periodic_resize_h();
    int width = param_.periodic_resize_w();
    CHECK_GT(width, 0) <<
        "preodic resize width has to be larger than zero";
    CHECK_GT(height, 0) <<
        "preodic resize height has to be larger than zero";
    switch (param_.periodic_resize()) {
      case TransformationParameter_PeriodicResizeMode_CENTER:
        ResizeImagePeriodic(*cv_img, height/2 - cv_img->rows/2,
            width/2 - cv_img->cols/2, cv_img);
        break;
      case TransformationParameter_PeriodicResizeMode_RANDOM:
        ResizeImagePeriodic(*cv_img, Rand(height), Rand(width), cv_img);
        break;
      case TransformationParameter_PeriodicResizeMode_ZERO:
        ResizeImagePeriodic(*cv_img, 0, 0, cv_img);
        break;
      default:
        LOG(FATAL) << "Unknown periodic resize method.";
    }
    *input_height = height;
    *input_width = width;
  }
}
#endif  // USE_OPENCV

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
