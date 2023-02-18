#################### External Data

from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import math, random
import numpy as np

#################### Image Processing Methods

def CreateHistogram(image, title, show=False):
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

########## Utility Methods

def splitChannels(image): 
  sW, sH = image.size;
  pX = image.load();
  Channel = [
    Image.new('L', (sW, sH), 0),
    Image.new('L', (sW, sH), 0),
    Image.new('L', (sW, sH), 0)
  ];
  CPx = [];
  for v in range(3): CPx.append(Channel[v].load());

  for index in range(3):
    for x in range(sW):
      for y in range(sH):
        CPx[index][x, y] = pX[x, y][index];
  return Channel;

def L_To_RGB(image, color):
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("RGB", (sW, sH), 0);
  rPx = Result.load();
  for x in range(sW):
    for y in range(sH):   
      if (color.lower()  == 'r'): rPx[x, y] = (pX[x, y], 0, 0);
      if (color.lower()  == 'g'): rPx[x, y] = (0, pX[x, y], 0);
      if (color.lower()  == 'b'): rPx[x, y] = (0, 0, pX[x, y]);
  return Result;

########## Image Arithmetic

def AddImages(image1, image2, alpha = 1):
  if (alpha > 1): alpha = 1;
  if (alpha < 0): alpha = 0;
  sW, sH = image2.size;
  p1 = image1.load();
  p2 = image2.load();

  for x in range(sW):
    for y in range(sH):
      p1[x, y] = round(p1[x, y] + (p2[x, y] * alpha));
  return image1;

########## Image Filters

def GammaCorrect(image, gamma, modifier=1):
  sW, sH = image.size;
  pX = image.load();
  for  x in range(sW):
    for y in range(sH):
      pX[x, y] = round(modifier * pow(pX[x, y], gamma));
  return iage;

########## Noise Generations

def CreateGaussian(image, std):
  w, h = image.size;
  guassianNoise = Image.new('L', (w, h), 0);
  guassianPx = guassianNoise.load();
  mean = 256/2;

  for x in range(w):
    for y in range(h):
       guassianPx[x, y] = round(np.random.normal(mean, std));
  return guassianNoise;


def CreateImpulse(image, chance):
  w, h = image.size;
  noiseResult = Image.new('L', (w, h), 0);
  noisePx = noiseResult.load();

  for x in range(w):
    for y in range(h):
      rand = random.randint(1, 100);
      if (rand < chance):
        if (rand < 50): 
          noisePx[x, y] = 255;
        else: noisePx[x, y] = 0;
  return noiseResult;

########## Noise Filters

def MedianNoiseFilter(image, size=3):
  if (size % 3 == 0): size += 1;
  pW, pH = image.size;
  pixel = image.load();
  Result = Image.new('L', (pW, pH), 0);
  ResultPx = Result.load();
  offset = math.floor(size / 2);

  localMean = [];
  for x in range(1, pW - 1):
    for y in range(1, pH - 1):
      for subX in range(-1 * offset, offset):
        for subY in range(-1 * offset, offset):
          localMean.append(pixel[x + subX, y + subY]);

      localMean.sort();
      ResultPx[x, y] = localMean[round(len(localMean) / 2)];
      localMean = [];
  return Result;

def MeanNoiseFilter(img, size=3):
  if (size % 3 == 0): size += 1;
  pW, pH = img.size;
  pixel = img.load();
  Result = Image.new('L', (pW, pH), 0);
  ResultPx = Result.load();
  offset = math.floor(size / 2);

  localMean = 0;
  for x in range(1, pW - 1):
    for y in range(1, pH - 1):
      for subX in range(-1 *offset, offset):
        for subY in range(-1 *offset, offset):
          localMean += pixel[x + subX, y + subY];

      ResultPx[x, y] = round(localMean / pow(size, 2));
      localMean = 0;
  return Result;
