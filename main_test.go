package main

import (
	"gocv.io/x/gocv"
	"image"
	"math"
	"testing"
)

func TestReadConfig(t *testing.T) {
	config := ConfigFFile("data/crown.json")

	if len(config.HatFilenames) != 1 {
		t.Errorf(`Cannot parse hat filenames as array, read: %q`,
			config.HatFilenames)
	} else if config.HatFilenames[0] != "data/crown.png" {
		t.Errorf(`Expected hat filename data/crown.png, found: %q`,
			config.HatFilenames[0])
	}

	if config.ClassifierModelFilename != "data/res10_300x300_ssd_iter_140000.caffemodel" {
		t.Errorf(`Maligned classifier model filename, found: %q`,
			config.ClassifierModelFilename)
	}
	if config.ClassifierNetworkFilename != "data/deploy.prototxt" {
		t.Errorf(`Maligned classifier network filename, found: %q`,
			config.ClassifierNetworkFilename)
	}

	if math.Abs(config.ScaleX[0] - 1.5) > 0.001 {
		t.Errorf(`Cannot read config width scaling, found %f`, config.ScaleX[0])
	}

  if len(config.TranslateYPct) != 1 {
		t.Errorf(`Unexpected vertical shifting, found %v`, config.TranslateYPct)
  } else if math.Abs(config.TranslateYPct[0] - 0.75) > 0.001 {
		t.Errorf(`Cannot read config vertical shifting, found %f`,
			config.TranslateYPct[0])
	}

  if len(config.TranslateXPct) != 1 {
		t.Errorf(`Unexpected horizontal shifting, found %v`, config.TranslateXPct)
  } else if math.Abs(config.TranslateXPct[0] - 0.0) > 0.001 {
		t.Errorf(`Cannot read config horizontal shifting, found %f`,
			config.TranslateXPct[0])
	}

}

func TestReadImage(t *testing.T) {
	img := ImageFFile("data/crown.png")
	defer img.Close()
	if img.Cols() != 1809 || img.Rows() != 1076 {
		t.Errorf(`Expected image size x , but found %dx%d`, img.Cols(), img.Rows())
	}
}

func TestPasteHat(t *testing.T) {
	config := ConfigFFile("data/crown.json")

	hat := gocv.NewMatWithSize(160, 100, gocv.MatTypeCV8U)
	for i := hat.Rows()-1; i >= 0; i-- {
		for j := hat.Cols()-1; j >= 0; j-- {
			hat.SetUCharAt(i, j, 0xff)
		}
	}

	target := gocv.NewMatWithSize(1000, 1000, gocv.MatTypeCV8U)
	roi := image.Rect(200, 200, 300, 300)
	PasteHat(hat, roi, &target, config, 0)

	if 0xff != target.GetUCharAt(200, 250) {
		t.Errorf(`Did not paste pixel at 250x200 (%d)`,
			target.GetUCharAt(200, 250))
	}
}

func TestPasteHatOffCorner(t *testing.T) {
	config := ConfigFFile("data/crown.json")

	hat := gocv.NewMatWithSize(160, 100, gocv.MatTypeCV8U)
	for i := hat.Rows()-1; i >= 0; i-- {
		for j := hat.Cols()-1; j >= 0; j-- {
			hat.SetUCharAt(i, j, 0xff)
		}
	}

	target := gocv.NewMatWithSize(1000, 1000, gocv.MatTypeCV8U)
	roi := image.Rect(900, 0, 990, 100)
	PasteHat(hat, roi, &target, config, 0)

	if 0xff != target.GetUCharAt(10, 945) {
		t.Errorf(`Did not paste pixel at 10x945 (%d)`,
			target.GetUCharAt(10, 945))
	}
}

func TestScaleImage(t *testing.T) {
	config := ConfigFFile("data/crown.json")
	img := gocv.NewMatWithSize(160, 100, gocv.MatTypeCV8U)
	defer img.Close()
	roi := image.Rect(2000, 1000, 2100, 1100)

	scaledImg := ScaleImageToRegion(img, roi, config, 0)
	defer scaledImg.Close()

	if scaledImg.Cols() != 150 || scaledImg.Rows() != 240 {
		t.Errorf(`Expected 150x240 scaled image but got %dx%d`,
			scaledImg.Cols(), scaledImg.Rows())
	}
}

func TestWebcamMatch(t *testing.T) {
  if IsDeviceId("data/foo.jpg") {
    t.Errorf(`Expected "data/foo.jpg" to match as file`)
  }
  if !IsDeviceId("0") {
    t.Errorf(`Expected "0" to match as device`)
  }
}
