// Based on gocv dnn-detection, scan an image for faces and place a hat on them.
// 
// Example model file from:
// https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
//
// Example network file from:
// https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
// 
// go run main.go data/donald.jpg data/jester-hat.png data/res10_300x300_ssd_iter_140000.caffemodel data/deploy.prototxt
//
package main

import (
	"errors"
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"log"
	"os"
	"path/filepath"
)

const hatScaleX = 1.35
const hatTranslateY = 0.75

// Take the results from Net.Forward() and the source image and return the 
// interesting boundary.
func GetInterestingRect(prob gocv.Mat, img gocv.Mat, i int) image.Rectangle {
	left := int(prob.GetFloatAt(0, i+3) * float32(img.Cols()))
	top := int(prob.GetFloatAt(0, i+4) * float32(img.Rows()))
	right := int(prob.GetFloatAt(0, i+5) * float32(img.Cols()))
	bottom := int(prob.GetFloatAt(0, i+6) * float32(img.Rows()))
	return image.Rect(left, top, right, bottom)
}

func ImageFFile(filename string) gocv.Mat {
	log.Printf("Reading image from %v", filename)
	img := gocv.IMRead(filename, gocv.IMReadColor)
	if img.Empty() {
		log.Fatal(errors.New(fmt.Sprintf("Error reading image from: %v", filename)))
	}
	return img
}

func PasteHat(hat gocv.Mat, roi image.Rectangle, target *gocv.Mat) {
	// shift the hat destination up and scale to fit hat width
	dy := int(float64(roi.Dy()) * hatTranslateY)
	roiCX := roi.Min.X + int(0.5 * float64(roi.Dx()))
	halfHatWidth := int(0.5 * float64(hat.Cols()))

	underHatRect := image.Rect(
		roiCX - halfHatWidth, // left
		roi.Max.Y - dy - hat.Rows(), // top
		roiCX + halfHatWidth, // right
		roi.Max.Y - dy) // bottom

	// clip to stay in target
	clipRect := image.Rect(0, 0, hat.Cols() - 1, hat.Rows() - 1)
	if underHatRect.Min.X < 0 {
		log.Printf("clipping min x")
		clipRect.Min.X = -underHatRect.Min.X
		underHatRect.Min.X = 0
	}
	if underHatRect.Max.X >= target.Cols() {
		log.Printf("clipping max x")
		clipRect.Max.X -= underHatRect.Max.X - target.Cols() + 1
		underHatRect.Max.X = target.Cols() - 1
	}
	// rounding...
	if clipRect.Dx() < underHatRect.Dx() {
		if clipRect.Min.X > 0 {
			clipRect.Min.X--
		} else {
			clipRect.Max.X++
		}
	}

	if underHatRect.Min.Y < 0 {
		log.Printf("clipping min y")
		clipRect.Min.Y = -underHatRect.Min.Y
		underHatRect.Min.Y = 0
	}
	if underHatRect.Max.Y >= target.Rows() {
		log.Printf("clipping max y")
		clipRect.Max.Y -= underHatRect.Max.Y - target.Rows() + 1
		underHatRect.Max.Y = target.Rows() - 1
	}
	// rounding...
	if clipRect.Dy() < underHatRect.Dy() {
		if clipRect.Min.Y > 0 {
			clipRect.Min.Y--
		} else {
			clipRect.Max.Y++
		}
	}
	clippedHat := hat.Region(clipRect)
	underHat := target.Region(underHatRect)
	log.Printf("Hat %dx%d", clippedHat.Cols(), clippedHat.Rows())
	log.Printf("Target %dx%d @ (%d,%d)->(%d,%d)", underHatRect.Dx(), underHatRect.Dy(), underHatRect.Min.X, underHatRect.Max.Y, underHatRect.Max.X, underHatRect.Min.Y)
	gocv.AddWeighted(clippedHat, 1.0, underHat, 1.0, 0.0, &underHat)
}

// Take an image and scale it to the width of the roi with an additional margin.
func ScaleImageToRegion(img gocv.Mat, roi image.Rectangle) gocv.Mat {
	scaledImg := gocv.NewMat()
	scaledWidth := int(float64(roi.Dx()) * hatScaleX)
	scaledHeight := img.Rows() * scaledWidth / img.Cols()
	gocv.Resize(img, &scaledImg, image.Pt(scaledWidth, scaledHeight),
		0, 0, gocv.InterpolationCubic)
	return scaledImg
}


func main() {
	if len(os.Args) < 5 {
		log.Println("How to run:\n\tgo run main.go [img-file] [hat-file] [model-file] [network-config-file]")
		return
	}

	window := gocv.NewWindow("Lots-o-hats")
	defer window.Close()

	target := ImageFFile(os.Args[1])
	defer target.Close()

	hat := ImageFFile(os.Args[2])
	defer hat.Close()

	modelFilename := os.Args[3]
	netConfigFilename := os.Args[4]
	netBackend := gocv.NetBackendDefault
	netTargetCPU := gocv.NetTargetCPU
	net := gocv.ReadNet(modelFilename, netConfigFilename)
	if net.Empty() {
		log.Fatal(errors.New(fmt.Sprintf(
			"Error reading network from model: %v network: %v", modelFilename, netConfigFilename)))
	}
	defer net.Close()
	net.SetPreferableBackend(gocv.NetBackendType(netBackend))
	net.SetPreferableTarget(gocv.NetTargetType(netTargetCPU))

	var ratio float64
  var mean gocv.Scalar
  var swapRGB bool

	// Caffe vs TF magic scaling factors
  if filepath.Ext(modelFilename) == ".caffemodel" {
    ratio = 1.0
    mean = gocv.NewScalar(104, 177, 123, 0)
    swapRGB = false
  } else {
    ratio = 1.0 / 127.5
    mean = gocv.NewScalar(127.5, 127.5, 127.5, 0)
    swapRGB = true
  }

	window.IMShow(target)
	// locate heads
	blob := gocv.BlobFromImage(target, ratio, image.Pt(300, 300), mean, swapRGB, false)
	net.SetInput(blob, "")
	prob := net.Forward("")

	const confidenceThresh = 0.5
	for i := 0; i < prob.Total(); i += 7 {
		confidence := prob.GetFloatAt(0, i+2)
		if confidence > confidenceThresh {
			roiRect := GetInterestingRect(prob, target, i)
			scaledHat := ScaleImageToRegion(hat, roiRect)
			defer scaledHat.Close()
			PasteHat(scaledHat, roiRect, &target)
			window.IMShow(target)
		}
	}

	prob.Close()
	blob.Close()


	for {
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
