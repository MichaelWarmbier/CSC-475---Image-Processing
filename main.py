from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from scipy.stats import norm
import math, random, statistics
import numpy as np
import asyncio

## NOISE ESTIMATION ON SLIDES 49
## GUASSIAN ON 50
# using np.random.normal(var, mean) 
# mean = sum(histogram) / 256
# you got this

####################

def createHistogram(image, title, show=False):
  width, height = image.size;
  pixel = image.load();

  histogram = [];

  for intensity in range(256):
    histogram.append(0);

  for x in range(width):
    for y in range(height):
      histogram[pixel[x, y]] += 1;

  plt.plot(range(256), histogram);    
  plt.xlabel('Luminosity')
  plt.ylabel('Frequency')
  plt.title(title)
  if (show): plt.show();

  return histogram;

def createGaussianNoise(w, h, variance, mean, mode='L'):
  guassianNoise = Image.new('L', (w, h), 0);
  guassianPx = guassianNoise.load();

  for x in range(w):
    for y in range(h):
       guassianPx[x, y] = round(np.random.normal(mean, variance));
  return guassianNoise;

def addImages(img1, img2):
  sW, sH = img1.size;
  p1 = img1.load();
  p2 = img2.load();

  for x in range(sW):
    for y in range(sH):
      p1[x, y] += p2[x, y];

  return img1;
  
####################

def main():
  imageTitle = ['GuassianResult90.png']
  imageSave = ['Output.jpg']
  
  Input = Image.open(imageTitle[0]).convert('L');
  pW, pH = Input.size;
  pixel = Input.load();

  histogram = createHistogram(Input, "Gaussian Result Var=90", True);
    
  #result.save(imageSave[0]);
  print("DONE!");
main();