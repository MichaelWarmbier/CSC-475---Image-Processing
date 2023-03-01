#################### External Data

from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import math, random, statistics
import numpy as np

#################### Image Processing Methods

########## Utility Methods

def CreateHistogram(image, title="", show=False):
  width, height = image.size;
  pixel = image.load();

  histogram = [];

  for intensity in range(256):
    histogram.append(0);

  for x in range(width):
    for y in range(height):
      histogram[pixel[x, y]] += 1;

  plt.bar(range(256), histogram);    
  plt.xlabel('Luminosity')
  plt.ylabel('Frequency')
  plt.title(title)
  if (show): plt.show();

  return histogram;

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

def toGrayScale(image):
  return ImageOps.grayscale(image);

def GlobalThreshold(image, OR=0):
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("1", (sW, sH), 0);
  rX = Result.load();
  LumValues = CreateHistogram(image);
  maxL, minL = max(LumValues), min(LumValues);
  for V in range(256):
    if (LumValues[V] == maxL): maxL = V;
    if (LumValues[V] == minL): minL = V;
  T = (round((maxL + minL) / 2) * (not OR)) + OR;
  print(T);
  
  for x in range(sW):
    for y in range(sH):
      if (pX[x, y] <= T): rX[x, y] = 0;
      else: rX[x, y] = 1;

  return Result;

def LocalThreshold(image, split=3):
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("1", (sW, sH), 0);
  rX = Result.load();
  wSegL = round(sW / split);
  hSegL = round(sH / split);

  segData = [];
  segMax, segMin = 0, 0;
  segT = 0;
  
  for wSeg in range(split):
    for hSeg in range(split):
      for currPass in range(2):
        for x in range(wSeg * wSegL, (wSeg + 1) * wSegL):
          for y in range(hSeg * hSegL, (hSeg + 1) * hSegL):
            if (x >= sW or y >= sH): continue;
            if (not currPass): segData.append(pX[x, y]);
            else: 
              if (pX[x, y] <= segT): rX[x, y] = 0;
              else: rX[x, y] = 1;
        if (not currPass): segMax = max(segData);
        if (not currPass): segMin = min(segData);
        segT = round((segMax + segMin) / 2);
        segData = [];
  return Result;

def NiblackThreshold(image, split=3, k=-.2):
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("1", (sW, sH), 0);
  rX = Result.load();
  wSegL = round(sW / split);
  hSegL = round(sH / split);

  segData = [];
  segMax, segMin = 0, 0;
  segT = 0;
  
  for wSeg in range(split):
    for hSeg in range(split):
      for currPass in range(2):
        for x in range(wSeg * wSegL, (wSeg + 1) * wSegL):
          for y in range(hSeg * hSegL, (hSeg + 1) * hSegL):
            if (x >= sW or y >= sH): continue;
            if (not currPass): segData.append(pX[x, y]);
            else: 
              if (pX[x, y] <= segT): rX[x, y] = 0;
              else: rX[x, y] = 1;
        if (not currPass): 
          segMax = max(segData);
          segMin = min(segData);
          segT = round((segMax + segMin) / 2) + (k * statistics.stdev(segData))
        segData = [];
  return Result;

def SauvolaThreshold(image, split=3, k=.5, R=128):
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("1", (sW, sH), 0);
  rX = Result.load();
  wSegL = round(sW / split);
  hSegL = round(sH / split);

  segData = [];
  segMax, segMin = 0, 0;
  segT = 0;
  
  for wSeg in range(split):
    for hSeg in range(split):
      for currPass in range(2):
        for x in range(wSeg * wSegL, (wSeg + 1) * wSegL):
          for y in range(hSeg * hSegL, (hSeg + 1) * hSegL):
            if (x >= sW or y >= sH): continue;
            if (not currPass): segData.append(pX[x, y]);
            else: 
              if (pX[x, y] <= segT): rX[x, y] = 0;
              else: rX[x, y] = 1;
        if (not currPass): 
          segMax = max(segData);
          segMin = min(segData);
          segT = round((segMax + segMin) / 2) * (1 + k * ((statistics.stdev(segData)/R) - 1))
        segData = [];
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
      p1[x, y] = round(p1[x, y] + p2[x, y] * alpha);
  return image1;

def SubImages(image1, image2, alpha = 1):
  if (alpha > 1): alpha = 1;
  if (alpha < 0): alpha = 0;
  sW, sH = image2.size;
  p1 = image1.load();
  p2 = image2.load();
  Result = Image.new('L', (sW, sH), 0);
  ResultPx = Result.load();

  for x in range(sW):
    for y in range(sH):
      ResultPx[x, y] = round(p1[x, y] - p2[x, y] * alpha);
  return Result;

########## Image Filters

def AlphaTrim(image, alpha=0, size=3):
  offset = math.floor(size / 2);
  
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new('L', (sW, sH), 0);
  ResultPx = Result.load();
  
  Neighborhood = [];
  AlphaTrim = [];
  
  for x in range(offset, sW - offset):
      for y in range(offset, sH - offset):
        
        for subX in range(-1 * offset, offset + 1):
          for subY in range(-1 * offset, offset + 1):
            Neighborhood.append(pX[x + subX, y + subY]);
        Neighborhood.sort();
        for V in range(0 + alpha, len(Neighborhood) - alpha):
          AlphaTrim.append(Neighborhood[V]);
        ResultPx[x, y] = round(sum(AlphaTrim) / len(AlphaTrim));
        Neighborhood = [];
        AlphaTrim = [];
  return Result;


def ContrastStretch(image): 
  sW, sH = image.size;
  pX = image.load();
  data = CreateHistogram(image);
  
  for x in range(sW):
    for y in range(sH):
      pX[x, y] = round(((pX[x, y] - min(data)) / (max(data) - min(data))) * 255);
  return image;

def Equalize(image):
  return ImageOps.equalize(image);

def GammaCorrect(image, gamma, modifier=1):
  sW, sH = image.size;
  pX = image.load();
  for  x in range(sW):
    for y in range(sH):
      pX[x, y] = round(modifier * pow(pX[x, y], gamma));
  return image;

def SharpenImage(image, size=3):
  image = toGrayScale(image);
  Smooth = SmoothImage(image, size);
  Detail = SubImages(image, Smooth);
  image = AddImages(image, Detail);
  return image;

def SmoothImage(image, size=3):
  if (size % 2 == 0): size += 1;
  offset = math.floor(size / 2);
  localMatrix = [];
  
  sW, sH = image.size;
  pX = image.load();

  Output = Image.new('L', (sW, sH), 0);
  oX = Output.load();

  for x in range(offset, sW - offset):
    for y in range(offset, sH - offset):
      for subX in range(-1 * offset, offset + 1):
        for subY in range(-1 * offset, offset + 1):
          localMatrix.append(pX[x + subX, y + subY]);
      oX[x, y] = round(1/9 * sum(localMatrix));
      localMatrix = [];
  return Output;

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
        if (random.randint(0, 9) < 4): 
          noisePx[x, y] = 255;
        else: noisePx[x, y] = 0;
  return noiseResult;

########## Noise Filters

def MinFilter(image, size=3):
  return image.filter(ImageFilter.MinFilter(size));

def MaxFilter(image, size=3):
  return image.filter(ImageFilter.MaxFilter(size));

def MedianFilter(image, size=3):
  if (size % 2 == 0): size += 1;
  pW, pH = image.size;
  pixel = image.load();
  Result = Image.new('L', (pW, pH), 0);
  ResultPx = Result.load();
  offset = math.floor(size / 2);

  localMean = [];
  for x in range(offset, pW - offset):
    for y in range(offset, pH - offset):
      for subX in range(-1 * offset, offset):
        for subY in range(-1 * offset, offset):
          localMean.append(pixel[x + subX, y + subY]);

      localMean.sort();
      ResultPx[x, y] = localMean[round(len(localMean) / 2)];
      localMean = [];
  return Result;

def WeightedMeanFilter(image, Weight):
  size = math.sqrt(len(Weight));
  if (size % 2 == 0): size += 1;
  pW, pH = image.size;
  pixel = image.load();
  Result = Image.new('L', (pW, pH), 0);
  ResultPx = Result.load();
  offset = math.floor(size / 2);

  localMatrix = [];
  Numerator = 0;
  for x in range(offset, pW - offset):
    for y in range(offset, pH - offset):
      for subX in range(-1 *offset, offset + 1):
        for subY in range(-1 *offset, offset + 1):
          localMatrix.append(pixel[x + subX, y + subY]);
      for V in range(round(size * size)): 
        Numerator += localMatrix[V] * Weight[V];
      ResultPx[x, y] = round(Numerator / sum(Weight));
      localMatrix = [];
      Numerator = 0;
  return Result;

def MeanFilter(img, size=3):
  if (size % 2 == 0): size += 1;
  pW, pH = img.size;
  pixel = img.load();
  Result = Image.new('L', (pW, pH), 0);
  ResultPx = Result.load();
  offset = math.floor(size / 2);

  localMean = 0;
  for x in range(offset, pW - offset):
    for y in range(offset, pH - offset):
      for subX in range(-1 *offset, offset):
        for subY in range(-1 *offset, offset):
          localMean += pixel[x + subX, y + subY];

      ResultPx[x, y] = round(localMean / pow(size, 2));
      localMean = 0;
  return Result;
