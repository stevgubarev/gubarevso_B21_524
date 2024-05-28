# -*- coding: utf-8 -*-
"""лаб 1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_oTnqy8Do5Cb7clKVdpIBmkRvABPZ4eP

Лабораторная работа 1.
Передискретизация изображений.
"""

from PIL import Image
import numpy as np
import math

def interpolate(image, K):
    data = image.load()
    width, height = image.size
    new_width, new_height = tuple(size * K for size in image.size)

    result = Image.new('RGB', (new_width, new_height))
    new_data = result.load()

    for x in range(width):
        for y in range(height):
            for offset_x in range(K):
                for offset_y in range(K):
                    new_x = x * K + offset_x
                    new_y = y * K + offset_y
                    new_data[new_x, new_y] = data[x, y]

    return result

def decimate(image, K):
    data = image.load()
    dim_original = image.size
    dim_result = tuple(math.floor(size / K) for size in dim_original)

    result = Image.new('RGB', dim_result)
    new_data = result.load()

    new_pixel = lambda x, y: data[x * K, y * K]

    for x in range(result.size[0]):
        for y in range(result.size[1]):
            new_data[x, y] = new_pixel(x, y)

    return result

def resample_image_2pass(image, in_K, de_K):
  return decimate(interpolate(image, in_K), de_K)

def resample_image_1pass(image, in_K, de_K):
    data = image.load()
    dim_original = image.size
    dim_result = tuple(int(size * in_K / de_K) for size in dim_original)

    result = Image.new('RGB', dim_result)
    new_data = result.load()

    for x in range(result.size[0]):
        for y in range(result.size[1]):
            orig_x = math.floor(x * de_K / in_K)
            orig_y = math.floor(y * de_K / in_K)
            new_data[x, y] = data[orig_x, orig_y]

    return result

with Image.open('/content/baran.png').convert('RGB') as img:
  ONEpassR_img = resample_image_1pass(img, 4, 7)
  interpolateBARAN = interpolate(img, 4)
  decimateBARAN = decimate(img, 8)
  TWOpassR_img = resample_image_2pass(img, 4, 7)

TWOpassR_img

ONEpassR_img

decimateBARAN

interpolateBARAN

decimateBARAN

decimateBARAN.save("/content/decimateBARAN.png")

res=Image.open("/content/decimateBARAN.png").convert('RGB')

res