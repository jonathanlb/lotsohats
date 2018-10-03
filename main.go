// Based on gocv dnn-detection, scan an image for faces and place a hat on them.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"log"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

type HatsConfig struct {
	HatFilenames []string
	ClassifierModelFilename string
	ClassifierNetworkFilename string
	ScaleX float64
	TranslateYPct float64
}

type NeuralConfig struct {
	NetBackend gocv.NetBackendType
	NetTargetCPU gocv.NetTargetType
	Network gocv.Net
	Ratio float64
	Mean gocv.Scalar
	SwapRGB bool
}

func ConfigFFile(filename string) HatsConfig {
	log.Printf("config at: %v", filename)
	var config HatsConfig
	file, e := ioutil.ReadFile(filename)
	if e != nil {
		log.Fatal(errors.New(fmt.Sprintf(
			"Error reading configuration from: %v\n%v", filename, e)))
	}
	json.Unmarshal(file, &config)

	// Make all fields relative to configuration file location
	fileSepIdx := strings.LastIndex(filename, "/")
	var prefix = "./"
	if fileSepIdx >= 0 {
		prefix = filename[0:fileSepIdx]
	}
	fields := [...]*string{
		&config.ClassifierModelFilename,
		&config.ClassifierNetworkFilename }
	for _, elt := range fields {
		*elt = fmt.Sprintf("%v/%v", prefix, *elt)
	}
	for i, elt := range config.HatFilenames {
		config.HatFilenames[i] = fmt.Sprintf("%v/%v", prefix, elt)
	}

	return config
}

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

func LocateHeads(target gocv.Mat, config NeuralConfig) (gocv.Mat, gocv.Mat) {
	blob := gocv.BlobFromImage(
		target, config.Ratio, image.Pt(300, 300), config.Mean,
		config.SwapRGB, false)
	config.Network.SetInput(blob, "")
	prob := config.Network.Forward("")

	return prob, blob
}

func NeuralFConfig(config HatsConfig) NeuralConfig {
	var result NeuralConfig

	result.NetBackend = gocv.NetBackendDefault
	result.NetTargetCPU = gocv.NetTargetCPU
	result.Network = gocv.ReadNet(
		config.ClassifierNetworkFilename,
		config.ClassifierModelFilename)
	if result.Network.Empty() {
		log.Fatal(errors.New(fmt.Sprintf(
			"Error reading network from model: %v network: %v",
			config.ClassifierModelFilename,
			config.ClassifierNetworkFilename)))
	}
	result.Network.SetPreferableBackend(
		gocv.NetBackendType(result.NetBackend))
	result.Network.SetPreferableTarget(
		gocv.NetTargetType(result.NetTargetCPU))

	// Caffe vs TF magic scaling factors
  if filepath.Ext(config.ClassifierModelFilename) == ".caffemodel" {
    result.Ratio = 1.0
    result.Mean = gocv.NewScalar(104, 177, 123, 0)
    result.SwapRGB = false
  } else {
    result.Ratio = 1.0 / 127.5
    result.Mean = gocv.NewScalar(127.5, 127.5, 127.5, 0)
    result.SwapRGB = true
  }

	return result
}

func PasteHat(hat gocv.Mat, roi image.Rectangle, target *gocv.Mat, config HatsConfig) {
	// shift the hat destination up and scale to fit hat width
	dy := int(float64(roi.Dy()) * config.TranslateYPct)
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
func ScaleImageToRegion(img gocv.Mat, roi image.Rectangle, config HatsConfig) gocv.Mat {
	scaledImg := gocv.NewMat()
	scaledWidth := int(float64(roi.Dx()) * config.ScaleX)
	scaledHeight := img.Rows() * scaledWidth / img.Cols()
	gocv.Resize(img, &scaledImg, image.Pt(scaledWidth, scaledHeight),
		0, 0, gocv.InterpolationCubic)
	return scaledImg
}


func main() {
	if len(os.Args) < 2 {
		log.Println("How to run:\n\tgo run main.go [img-file] [config-json]")
		return
	}

	window := gocv.NewWindow("Lots-o-hats")
	defer window.Close()

	target := ImageFFile(os.Args[1])
	defer target.Close()

	config := ConfigFFile(os.Args[2])

	hat := ImageFFile(config.HatFilenames[0])
	defer hat.Close()

	neuralConfig := NeuralFConfig(config)
	defer neuralConfig.Network.Close()

	window.IMShow(target)
	prob, blob := LocateHeads(target, neuralConfig)

	const confidenceThresh = 0.5
	for i := 0; i < prob.Total(); i += 7 {
		confidence := prob.GetFloatAt(0, i+2)
		if confidence > confidenceThresh {
			roiRect := GetInterestingRect(prob, target, i)
			scaledHat := ScaleImageToRegion(hat, roiRect, config)
			defer scaledHat.Close()
			PasteHat(scaledHat, roiRect, &target, config)
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
