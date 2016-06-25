#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>
#include <math.h>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
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
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;
  const bool has_rotation = param_.rotation_range()>0;
  const bool has_perspective_transformation = param_.perspective_transformation_border()>0;
  const bool random_crop_test = param_.random_crop_test();
  const bool has_scale_jitter = param_.scale_jitter_range()>0;
  const bool has_contrast_jitter = param_.contrast_jitter_range()>0;
  
  // Datum *newdatum = NULL;
  cv::Mat cv_img;
  bool use_new_data =false;

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size > 0) {
    crop_h = crop_w = crop_size;
  }

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
  if (has_contrast_jitter) {
    float contrast_jitter_range = param_.contrast_jitter_range();
    CHECK_GE(contrast_jitter_range,0);
    scale = scale*exp(contrast_jitter_range*(float)(Rand(21)-10)/20.0);
  }

  if ((has_perspective_transformation || has_rotation || has_scale_jitter) ) {
    // load the image
    cv_img=cv::Mat(datum_height,datum_width,CV_8UC3);
    for (int h = 0; h < datum_height; ++h) {
      uchar* ptr = cv_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < datum_width; ++w) {
        for (int c = 0; c < datum_channels; ++c) {
          int datum_index = (c * datum_height + h) * datum_width + w;
          ptr[img_index++]=static_cast<uint8_t>(data[datum_index]);
        }
      }
    }
    // if has rotation or scale jitter
    if (has_rotation || has_scale_jitter) {
      const int rotation_range = param_.rotation_range();
      const float scale_jitter = exp(param_.scale_jitter_range()*(float)(Rand(21)-10)/20.0);
      CHECK_GE(scale_jitter,0);
      CHECK_GE(rotation_range,0);
      CHECK_LE(rotation_range,180);
      // rotate the image
      cv::Point2f pt(cv_img.cols/2,cv_img.rows/2);
      cv::Mat r;
      if (has_rotation) {
        r = cv::getRotationMatrix2D(pt, Rand(rotation_range)-rotation_range/2, scale_jitter);
      } else {
        r = cv::getRotationMatrix2D(pt, 0, scale_jitter);
      }
      // cv::Mat r = cv::getRotationMatrix2D(pt, 50.0, 0.10);
      cv::warpAffine(cv_img,cv_img,r,cv::Size(datum_width, datum_height));
    }
    //perspective transform
    if (has_perspective_transformation) {
      const int perspective_transformation_border = param_.perspective_transformation_border();
      CHECK_GE(perspective_transformation_border,0);
      CHECK_LE(perspective_transformation_border,datum_height/2);
      CHECK_LE(perspective_transformation_border,datum_width/2);
      // get transformation matrix
      cv::Point2f src_shape[4];
      src_shape[0]=cv::Point2f(0+Rand(perspective_transformation_border),0+Rand(perspective_transformation_border));
      src_shape[1]=cv::Point2f(0+Rand(perspective_transformation_border),cv_img.rows-Rand(perspective_transformation_border));
      src_shape[2]=cv::Point2f(cv_img.cols-Rand(perspective_transformation_border),cv_img.rows-Rand(perspective_transformation_border));
      src_shape[3]=cv::Point2f(cv_img.cols-Rand(perspective_transformation_border),0+Rand(perspective_transformation_border));
      cv::Point2f dst_shape[4];
      dst_shape[0]=cv::Point2f(0,0);
      dst_shape[1]=cv::Point2f(0,cv_img.rows);
      dst_shape[2]=cv::Point2f(cv_img.cols,cv_img.rows);
      dst_shape[3]=cv::Point2f(cv_img.cols,0);
      cv::Mat ptmatrix = cv::getPerspectiveTransform(src_shape,dst_shape);
      cv::warpPerspective(cv_img,cv_img,ptmatrix,cv::Size(datum_width,datum_height),cv::INTER_LINEAR,cv::BORDER_CONSTANT);
    }
    // copy back to datum
    // CVMatToDatum(cv_img, newdatum);
    use_new_data = true;
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_h > 0 || crop_w > 0) {
    height = crop_h;
    width = crop_w;
    // We only do random crop when we do training.
    if (phase_ == TRAIN || random_crop_test) {
      h_off = Rand(datum_height - crop_h + 1);
      w_off = Rand(datum_width - crop_w + 1);
    } else {
      h_off = (datum_height - crop_h) / 2;
      w_off = (datum_width - crop_w) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          if (use_new_data) {
            datum_element =
              static_cast<Dtype>(cv_img.at<cv::Vec3b>(h+h_off,w+w_off)[c]);
          } else {
            datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
          }
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
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

  const int crop_size = param_.crop_size();
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

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size > 0) {
    crop_h = crop_w = crop_size;
  }
  if (crop_h > 0 || crop_w > 0) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

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
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
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
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

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

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;
  if (crop_size > 0) {
    crop_h = crop_w = crop_size;
  }

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_h > 0 || crop_w > 0) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
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

  CHECK(cv_lab.channels() == 1) << "Can only handle grayscale label images";
  CHECK(cv_lab.rows == img_height && cv_lab.cols == img_width) << "Input and label "
      << "image heights and widths must match";
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
  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  // add crop h and w 
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size > 0) {
    crop_h = crop_w = crop_size;
  }

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  Dtype* mean = NULL;
  if (has_mean_file ) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;  
  cv::Mat cv_cropped_img = cv_img;
  cv::Mat cv_cropped_label = cv_lab;
  if (crop_h > 0 || crop_w > 0) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_img(roi);
    cv_cropped_label = cv_lab(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_label.data);

  Dtype* transformed_data_img = transformed_blob_img->mutable_cpu_data();
  Dtype* transformed_data_lab = transformed_blob_lab->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr_img = cv_cropped_img.ptr<uchar>(h);
    const uchar* ptr_lab = cv_cropped_label.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        CHECK(img_channels==1)<<"only support single channel images for not, fix it for color images";
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel_img = static_cast<Dtype>(ptr_img[img_index]);
        Dtype pixel_lab = static_cast<Dtype>(ptr_lab[img_index]);
        img_index++;
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data_img[top_index] =
            (pixel_img - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data_img[top_index] =
              (pixel_img - mean_values_[c]) * scale;
          } else {
            transformed_data_img[top_index] = pixel_img * scale;
          }
        }
        transformed_data_lab[top_index] = pixel_lab;
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  CHECK(param_.crop_size() == 0 ||
        (param_.crop_h() == 0 && param_.crop_w() == 0))
        << "Crop size is crop_size OR crop_h and crop_w; not both";
  CHECK((param_.crop_h() != 0) == (param_.crop_w() != 0))
        << "For non-square crops both crop_h and crop_w are required.";
  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else if (crop_h>0 && crop_w>0){
      transformed_blob->Reshape(input_num, input_channels,
                                crop_h, crop_w);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
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


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;
  if (crop_size > 0) {
    crop_h = crop_w = crop_size;
  }

  int h_off = 0;
  int w_off = 0;
  if (crop_h > 0 || crop_w > 0) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_h + 1);
      w_off = Rand(input_width - crop_w + 1);
    } else {
      h_off = (input_height - crop_h) / 2;
      w_off = (input_width - crop_w) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
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
  const int crop_size = param_.crop_size();
  const int crop_h = param_.crop_h();
  const int crop_w = param_.crop_w();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  CHECK(param_.crop_size() == 0 ||
        (param_.crop_h() == 0 && param_.crop_w() == 0))
        << "Crop size is crop_size OR crop_h and crop_w; not both";
  CHECK((param_.crop_h() != 0) == (param_.crop_w() != 0))
        << "For non-square crops both crop_h and crop_w are required.";
  if (crop_size) {
    shape[2]=crop_size;
    shape[3]=crop_size;
  } else if (crop_h>0 && crop_w>0){
    shape[2]=crop_h;
    shape[3]=crop_w;
  } else {
    shape[2] = datum_height;
    shape[3] = datum_width;
  }
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
  const int crop_size = param_.crop_size();
  const int crop_h = param_.crop_h();  
  const int crop_w = param_.crop_w();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);  
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  CHECK(param_.crop_size() == 0 ||
        (param_.crop_h() == 0 && param_.crop_w() == 0))
        << "Crop size is crop_size OR crop_h and crop_w; not both";
  CHECK((param_.crop_h() != 0) == (param_.crop_w() != 0))
        << "For non-square crops both crop_h and crop_w are required.";
  if (crop_size) {
    shape[2]=crop_size;
    shape[3]=crop_size;
  } else if (crop_h>0 && crop_w>0){
    shape[2]=crop_h;
    shape[3]=crop_w;
  } else {
    shape[2] = img_height;
    shape[3] = img_width;
  }
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

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
