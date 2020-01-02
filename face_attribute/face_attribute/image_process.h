#pragma once
typedef unsigned char* buff;

// method:
// bilinear: 1

// 11101001:不支持的图像变换方法
// 1       :执行正确
int image_resize(buff imagedata, int src_width, int src_height, buff output, int dst_width, int dst_height, int depth, int method);

int image_scaling_bilinear(buff imagedata, int src_width, int src_height, buff output, int dst_width, int dst_height, int depth);