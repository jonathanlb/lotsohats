# Lotsohats
Deface your image or stream of images with cephalic accessories.

## How to run
Download [deep neural network classifier readable by GoCV](https://godoc.org/gocv.io/x/gocv#ReadNet) to detect faces, e.g. Caffe or TensorFlow, or some other image feature.
```bash
cd data
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
```

Identify an image to place on top of your detected features.  The image format must be one readable by OpenCV, e.g. png or jpeg.  The portion of the image representing the background must be black.  OpenCV does not reliably handle png transparency.

Create a JSON configuration file resembling the following
```json
{
  "HatFilenames" : ["crown.png"],
  "ClassifierModelFilename" : "res10_300x300_ssd_iter_140000.caffemodel",
  "ClassifierNetworkFilename" : "deploy.proto",
  "ScaleX": 1.35,
  "TranslateYPct" : 0.75,
}
```

```bash
go run main.go data/multi.json data/sense8.png 
```
