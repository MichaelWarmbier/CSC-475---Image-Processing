#################### External Data

from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import math, random, statistics, sys
import numpy as np
import time
import cv2


#################### Image Processing Methods

########## Utility Methods

def getTime():
  return time.time();

def getLuminosityRange(img):
  Hist = CreateHistogram(img, "L", "", False);
  minIndex = -1;
  maxIndex = 0;
  for v in range(len(Hist)):
    if (Hist[v]):
      if (v > maxIndex): maxIndex = v;
      if (minIndex == -1): minIndex = v;

  return maxIndex - minIndex;

def BinZeroPad(image, size=1, BinColor = 0):
  sW, sH = image.size;
  pX = image.load();
  for x in range(sW):
    for y in range(sH):
      if (x < size or y < size or x > (sW - size - 1) or y > (sH - size - 1)):
        pX[x, y] = 0;
  return image;

def BitSplit(image):
  pX = image.load();
  sW, sH = image.size;
  Layers = [];
  BinPx = [];
  for im in range(8): 
    Layers.append(Image.new('1', (sW, sH), 0));
    BinPx.append(Layers[im].load());
  for x in range(sW):
    for y in range(sH):
      binData = bin(pX[x, y]).replace("0b", "")[::-1];
      for im in range(len(binData)): BinPx[im][x, y] = int(binData[im]);

  return Layers;

def IsolateObjects(threshPath, originalPath, separate=False):
  original = cv2.imread(originalPath, cv2.IMREAD_UNCHANGED);
  img = cv2.imread(threshPath, cv2.IMREAD_UNCHANGED);
  cnt, grgb = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

  isolatedObjects = [];
  for contourIndex in range(len(cnt)):
    isolatedObjects.append(np.ones(img.shape[:2], dtype="uint8") * 255);
    mask = isolatedObjects[contourIndex];
    cv2.drawContours(mask, cnt, contourIndex, 0, -1);
    img = cv2.bitwise_and(img, img, mask=mask);
    if (separate): 
      isolatedObjects[contourIndex] = cv2.bitwise_and(original, cv2.bitwise_not(isolatedObjects[contourIndex]));
  print("Total Objects:", len(isolatedObjects));
  return isolatedObjects;

def ThreshAndPrep(img):
  Struct = [
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    1, 1, 1, 1, 1,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0
  ]

  Struct2 = [
    0, 1, 0,
    1, 1, 1,
    0, 1, 0,
  ]


  Input = toGrayScale(img);
  Input = SharpenImage(img);
  Input = GlobalThreshold(Input, 210);
  Input = Erode(Input, Struct);
  Input = Erode(Input, Struct2);
  Input = Dilate(Input, Struct);
  Input = Dilate(Input, Struct2);
  Input = BinInvert(Input);
  Input = BinZeroPad(Input, 2);

  return Input;

def Circularity(imgPath):
  Input = cv2.imread(imgPath);
  Input = cv2.cvtColor(Input, cv2.COLOR_BGR2GRAY);

  cnt, grgb = cv2.findContours(Input, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
  cnt = cnt[0];

  area = cv2.contourArea(cnt);
  perimeter = cv2.arcLength(cnt, True);
  circularity = 4 * np.pi * (area / (perimeter * perimeter));

  return circularity;

def Dilate(image, struct):
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("1", (sW, sH), 0);
  rX = Result.load();
  isOdd = (len(struct) % 2 != 0);
  offset = math.floor(np.sqrt(len(struct))/2);
  LocalArea = [];
  Values = []

  for x in range(offset, sW - offset):
    for y in range(offset, sH - offset):
      for subX in range(-1 * offset, offset + (isOdd)):
        for subY in range(-1 * offset, offset + (isOdd)):
          LocalArea.append(pX[x + subX, y + subY]);
      for v in range(len(LocalArea)):
        if (struct[v]): Values.append(LocalArea[v]);
      rX[x, y] = max(Values);
      LocalArea = [];
      Values = [];
  return Result;

def Erode(image, struct):
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("1", (sW, sH), 0);
  rX = Result.load();
  isOdd = (len(struct) % 2 != 0);
  offset = math.floor(np.sqrt(len(struct))/2);
  LocalArea = [];
  Values = []

  for x in range(offset, sW - offset):
    for y in range(offset, sH - offset):
      for subX in range(-1 * offset, offset + (isOdd)):
        for subY in range(-1 * offset, offset + (isOdd)):
          LocalArea.append(pX[x + subX, y + subY]);
      for v in range(len(LocalArea)):
        if (struct[v]): Values.append(LocalArea[v]);
      rX[x, y] = min(Values);
      LocalArea = [];
      Values = [];

  return Result;

def DetectObject(image, object):
  start_time = time.time();
  Struct = [
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    1, 1, 1, 1, 1,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0
  ]

  object = toGrayScale(object);
  Original = toGrayScale(image);
  
  Sharp = SharpenImage(Original);
  print("Applied Enhancement", (time.time() - start_time));
  Median = MedianFilter(Sharp, 5);
  print("Applied Noise Redcution", (time.time() - start_time))
  Segmented =  LocalThreshold(Median, 5);
  print("Applied Thresholding", (time.time() - start_time))
  BinNoiseRed = Erode(Segmented, Struct);
  BinNoiseRed = Dilate(BinNoiseRed, Struct);
  print("Applied Morphology", (time.time() - start_time))
  Multiply = MultiplyImages(BinNoiseRed, Original);
  Isolated = FindObject(Multiply, object);
  print("Object Located", (time.time() - start_time));
  
  return Isolated;

def FindObject(image1, image2):
  sW, sH = image1.size;
  mW, mH = image2.size;
  pX = image1.load();
  mX = image2.load();
  Output = Image.new('1', (sW, sH), 0);
  oX = Output.load();

  modValues = 0;
  origValues = 0;
  smallestSectionIndex = -1;
  smallestSectionValues = 1_000_000

  for x in range(mW):
    for y in range(mH):
      modValues+= mX[x, y];


  index = 0;
  completion = 10;
  for x in range(0, sW - mW):
    for y in range(0, sH - mH):
      for subX in range(mW):
        for subY in range(mH):
          if (not origValues): continue;
          origValues += pX[x + subX, y + subY];
      origValues = abs((modValues - origValues) / (origValues + 1));
      if (origValues < smallestSectionValues): 
        smallestSectionIndex = index;
        smallestSectionValues = origValues;
      index += 1;

  Location = smallestSectionIndex;
  
  index = 0;
  for x in range(0, sW - mW):
    for y in range(0, sH - mH):
      if (index == smallestSectionIndex):
        for subX in range(mW):
          for subY in range(mH):
            oX[x + subX, y + subY] = 1;
        return Output;
      index += 1;

def smallestIndex(values):
  minV = min(values);
  index = values.index(minV);
  return index;

def GenerateMagnitude(*image):
  sW, sH = image[0].size;
  pX = [];
  for arg in image: pX.append(arg.load());

  values = []
    
  for x in range(sW):
    for y in range(sH):
      for v in range(len(image)): values.append(pow(pX[v][x, y], 2));
      pX[0][x, y] = round(np.sqrt(sum(values)));
      values = [];

  return image[0];

def GenerateMaxResult(*image):
  sW, sH = image[0].size;
  pX = [];
  for arg in image: pX.append(arg.load());

  values = []
    
  for x in range(sW):
    for y in range(sH):
      for v in range(len(image)): values.append(pX[v][x, y]);
      pX[0][x, y] = max(values);
      values = [];

  return image[0];

def GenerateDirection(image1, image2):
  sW, sH = image1.size;
  pX1 = image1.load();
  pX2 = image2.load();
    
  for x in range(sW):
    for y in range(sH):
      pX1[x, y] = round(np.degrees(np.arctan2(pX1[x, y], (pX2[x, y] + 1))));

  return image1;

def CreateHistogram(image, mode="L", title="", show=False):
  width, height = image.size;
  pixel = image.load();

  histogram = [];

  if (mode == '1'): mode = 2;
  else: mode = 256;

  
  for intensity in range(mode):
    histogram.append(0);

  for x in range(width):
    for y in range(height):
      if (mode == 256): histogram[pixel[x, y]] += 1;
      else: histogram[(pixel[x, y] >= 1)] += 1;

  if (show):
    plt.bar(range(mode), histogram);    
    plt.xlabel('Luminosity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show();

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
      if (color.lower() == 'r'): rPx[x, y] = (pX[x, y], 0, 0);
      if (color.lower() == 'g'): rPx[x, y] = (0, pX[x, y], 0);
      if (color.lower() == 'b'): rPx[x, y] = (0, 0, pX[x, y]);
  return Result;

def toGrayScale(image):
  return ImageOps.grayscale(image);

def GaussianSmoothing(image):
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("L", (sW, sH), 0);
  rX = Result.load();

  Kernal = [
    1,  4,  7,  4,  1,
    4, 16, 26, 16,  4,
    7, 26, 41, 26,  7,
    4, 16, 26, 16,  4,
    1,  4,  7,  4,  1
  ]
  LocalArea = []

  for x in range(1, sW - 3):
    for y in range(1, sH - 3):
      for subX in range(-2, 3):
        for subY in range(-2, 3):
          LocalArea.append(pX[x + subX, y + subY]);
      for V in range(len(LocalArea)):
        LocalArea[V] *= Kernal[V];
      rX[x, y] = round((1/273) * sum(LocalArea));
      LocalArea = [];

  return Result;

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
  
  for x in range(sW):
    for y in range(sH):
      if (pX[x, y] <= T): rX[x, y] = 0;
      else: rX[x, y] = 1;

  return Result;

def GlobalThreshold_Grayscale(image, OR=0):
  sW, sH = image.size;
  pX = image.load();
  R1, R2 = Image.new("L", (sW, sH), 0), Image.new("L", (sW, sH), 0);
  rX1, rX2 = R1.load(), R2.load();
  LumValues = CreateHistogram(image);
  maxL, minL = max(LumValues), min(LumValues);
  for V in range(256):
    if (LumValues[V] == maxL): maxL = V;
    if (LumValues[V] == minL): minL = V;
  T = (round((maxL + minL) / 2) * (not OR)) + OR;
  
  for x in range(sW):
    for y in range(sH):
      if (pX[x, y] <= T): rX1[x, y] = pX[x, y];
      else: rX2[x, y] = pX[x, y];

  return [R1, R2];

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

def NiblackThreshold(image, size=3, k=-.1):
  image = toGrayScale(image);
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("1", (sW, sH), 0);
  rX = Result.load();
  offset = int(np.floor(size/2));

  segData = [];
  segMax, segMin = 0, 0;
  segT = 0;
  
  for x in range(offset, sW - offset):
    for y in range(offset, sH - offset):
      for currPass in range(2):
        for subX in range(-1 * offset, offset + 1):
          for subY in range(-1 * offset, offset + 1):
            if (currPass == 0): segData.append(pX[x + subX, y + subY]);
            else: 
              if (pX[x + subX, y] <= segT): rX[x, y] = 0;
              else: rX[x, y] = 1;
        if (currPass == 0): 
          segMax = max(segData);
          segMin = min(segData);
          segT = round((segMax + segMin) / 2) + (k * statistics.stdev(segData));
        segData = [];
  return Result;

def SauvolaThreshold(image, size=3, k=.5, R=128):
  image = toGrayScale(image);
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new("1", (sW, sH), 0);
  rX = Result.load();
  offset = int(np.floor(size/2));

  segData = [];
  segMax, segMin = 0, 0;
  segT = 0;
  
  for x in range(offset, sW - offset):
    for y in range(offset, sH - offset):
      for currPass in range(2):
        for subX in range(-1 * offset, offset + 1):
          for subY in range(-1 * offset, offset + 1):
            if (currPass == 0): segData.append(pX[x + subX, y + subY]);
            else: 
              if (pX[x + subX, y] <= segT): rX[x, y] = 0;
              else: rX[x, y] = 1;
        if (currPass == 0): 
          segMax = max(segData);
          segMin = min(segData);
          segT = round((segMax + segMin) / 2) * (1 + k * ((statistics.stdev(segData)/R) - 1))
        segData = [];
  return Result;

def BernsenThreshold(image, size=3, cMin=15):
  offset = math.floor(size / 2);
  
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new('1', (sW, sH), 0);
  rX = Result.load();
  
  Neighborhood = [];
  T, C, lMax, lMin = 0, 0, 0, 0;
  
  for x in range(offset, sW - offset):
    for y in range(offset, sH - offset):
      for subX in range(-1 * offset, offset + 1):
        for subY in range(-1 * offset, offset + 1):
          Neighborhood.append(pX[x + subX, y + subY]);
      lMax = max(Neighborhood);
      lMin = min(Neighborhood);
      T = round((lMax - lMin) / 2);
      C = lMax - lMin;
      if (pX[x, y] <= T or C < cMin): rX[x, y] = 1;
      else: rX[x, y] = 0;
      Neighborhood = [];

  return Result;

def Invert(image):
  sW, sH = image.size;
  pX = image.load();
  for x in range(sW):
    for y in range(sH):
      pX[x, y] = 255 - pX[x, y];
  return image;

def BinInvert(image):
  sW, sH = image.size;
  pX = image.load();
  for x in range(sW):
    for y in range(sH):
      pX[x, y] = not pX[x, y];
  return image;

def ApplyMask(image, mask):
  M = len(mask);
  offset = math.floor(round(np.sqrt(M)) / 2);
  sW, sH = image.size;
  pX = image.load();
  Result = Image.new('L', (sW, sH), 0);
  rX = Result.load();
  Local = [];

  for x in range(offset, sW - offset):
    for y in range(offset, sH - offset):
      for subX in range(-1 * offset, offset + 1):
        for subY in range(-1 * offset, offset + 1):
          Local.append(pX[x + subX, y + subY]);
      for V in range(M):
        Local[V] *= mask[V];
        rX[x, y] = round(sum(Local)); 
      Local = [];

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

def MultiplyImages(image1, image2, alpha=1):
  if (alpha > 1): alpha = 1;
  if (alpha < 0): alpha = 0;
  sW, sH = image2.size;
  p1 = image1.load();
  p2 = image2.load();
  Result = Image.new('L', (sW, sH), 0); 
  rX = Result.load();
  for x in range(sW):
    for y in range(sH):
      rX[x, y] = p1[x, y] * p2[x, y] * alpha;
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

def GammaCorrect(image, gamma, modifier=1):
  sW, sH = image.size;
  pX = image.load();
  for x in range(sW):
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

#################### Internal Data

Sobel_Right = [
  -1, 0, 1,
  -2, 0, 2,
  -1, 0, 1
]

Sobel_Left = [
  1, 0, -1,
  2, 0, -2,
  1, 0, -1
]

Sobel_Up = [
   1,  2,  1,
   0,  0,  0,
  -1, -2, -1
]

Sobel_Down = [
 -1, -2, -1,
  0,  0,  0,
  1,  2,  1
]

FreiChen_Right = [
  -1, 0, 1,
  -1 * np.sqrt(2), 0, np.sqrt(2),
  -1, 0, 1
]

FreiChen_Left = [
  1, 0, -1,
  np.sqrt(2), 0, -1 * np.sqrt(2),
  1, 0, -1
]

FreiChen_Down = [
   -1, -1 * np.sqrt(2),  -1,
   0,  0,  0,
   1, np.sqrt(2), 1
]

FreiChen_Up = [
   1,  np.sqrt(2),  1,
   0,  0,  0,
  -1, -1 * np.sqrt(2), -1
]

Robinson_0 = [
  -1,  0,  1,
  -2,  0,  2,
  -1,  0,  1
]

Robinson_45 = [
   0,  1,  2,
  -1,  0,  1,
  -2, -1,  0
]

Robinson_90 = [
   1,  2,  1,
   0,  0,  0,
  -1, -2, -1
]

Robinson_135 = [
   2,  1,  0,
   1,  0, -1,
   0, -1, -2
]

Robinson_180 = [
   1,  0, -1,
   2,  0, -2,
   1,  0, -1
]

Robinson_225 = [
   0, -1, -2,
   1,  0, -1,
   2,  1,  0
]

Robinson_270 = [
   1,  2,  1,
   0,  0,  0,
  -1, -2, -1
]

Robinson_315 = [
  -2, -1,  0,
  -1,  0,  1,
   0,  1,  2
]

Kirsch_0 = [
  -3, -3,  5,
  -3,  0,  5,
  -3, -3,  5
]

Kirsch_45 = [
  -3,  5,  5,
  -3,  0,  5,
  -3, -3, -3
]

Kirsch_90 = [
   5,  5,  5,
  -3,  0, -3,
  -3, -3, -3
]

Kirsch_135 = [
   5,  5,  -3,
   5,  0, -3,
  -3, -3, -3
]

Kirsch_180 = [
   5, -3, -3,
   5,  0, -3,
   5, -3, -3
]

Kirsch_225 = [
  -3, -3, -3,
   5,  0, -3,
   5,  5, -3
]

Kirsch_270 = [
  -3, -3,  -3,
  -3,  0,  -3,
   5,  5,   5
]

Kirsch_315 = [
  -3, -3,  5,
  -3,  0,  5,
  -3, -3,  5
]


NevatiaBabu_0 = [
  -100, -100, 0, 100, 100,
  -100, -100, 0, 100, 100,
  -100, -100, 0, 100, 100,
  -100, -100, 0, 100, 100,
  -100, -100, 0, 100, 100
]

NevatiaBabu_30 = [
  -100,  -32,  100, 100, 100,
  -100,  -78,   92, 100, 100,
  -100, -100,    0, 100, 100,
  -100, -100,  -92,  78, 100,
  -100, -100, -100,  32, 100
]

NevatiaBabu_60 = [
   100,  100,  100, 100,  100,
   -32,   78,  100, 100,  100,
  -100,  -92,    0,  92,  100,
  -100, -100, -100, -78,   32,
  -100, -100, -100, -100, -100
]

NevatiaBabu_90 = [
  100,  100,  100,  100,  100,
  100,  100,  100,  100,  100,
    0,    0,    0,    0,    0,
 -100, -100, -100, -100, -100,
 -100, -100, -100, -100, -100
]

NevatiaBabu_120 = [
  100,  100,  100,  100,  100,
  100,  100,  100,   78,  -32,
  100,   92,    0,  -92, -100,
   32,  -78, -100, -100, -100,
 -100, -100, -100, -100, -100
]

NevatiaBabu_150 = [
  100,  100,  100,   32, -100,
  100,  100,   92,  -78, -100,
  100,  100,    0, -100, -100,
  100,   78,  -92, -100, -100,
  100,  -32, -100, -100, -100
]
