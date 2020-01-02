#include <iostream>
#include "image_process.h"

template <typename T>
inline T min(T a, T b){	return a < b ? a : b;}

template <typename T>
inline T max(T a, T b){	return a > b ? a : b;}

// 11101001:不支持的图像变换方法
// 1 ： 执行正确

int image_scaling_bilinear(buff imagedata, int src_width, int src_height, buff output, int dst_width, int dst_height, int depth)
{
    // 输入数据行列应该是整数倍
	float ExpScalValue_h, ExpScalValue_w; 
	ExpScalValue_h = (dst_height-1)/(float)(src_height-1);
	ExpScalValue_w = (dst_width-1) / (float)(src_width-1);
	// 期望的缩放倍数（允许小数）
	// 图像每一行的字节数必须是4的整数倍
    // 加3的目的是识别向上取整的效果
	int lineByte = (src_width * depth + 3) / 4 * 4; 
	int lineByte2 = (dst_width * depth + 3) / 4 * 4;
	/*******************图像处理部分******************/
	/*******************双线性插值******************/
	for (int i = 0; i < dst_height; i++)
		for (int j = 0; j < dst_width; j++)
		{
			float d_original_img_hnum = i / ExpScalValue_h;
			float d_original_img_wnum = j / ExpScalValue_w;
			int i_original_img_hnum = d_original_img_hnum;
			int i_original_img_wnum = d_original_img_wnum;
			float distance_to_a_x = d_original_img_wnum - i_original_img_wnum;//在原图像中与a点的水平距离    
			float distance_to_a_y = d_original_img_hnum - i_original_img_hnum;//在原图像中与a点的垂直距离    

			int original_point_a = i_original_img_hnum * lineByte + i_original_img_wnum * depth;
			//数组位置偏移量，对应于图像的各像素点RGB的起点,相当于点A      
			int original_point_b = i_original_img_hnum * lineByte + (i_original_img_wnum + 1) * depth;
			//数组位置偏移量，对应于图像的各像素点RGB的起点,相当于点B    
			int original_point_c = (i_original_img_hnum + 1) * lineByte + i_original_img_wnum * depth;
			//数组位置偏移量，对应于图像的各像素点RGB的起点,相当于点C     
			int original_point_d = (i_original_img_hnum + 1) * lineByte + (i_original_img_wnum + 1) * depth;
			//数组位置偏移量，对应于图像的各像素点RGB的起点,相当于点D 

			//边缘情况处理
			if (i_original_img_hnum == src_height - 1)
			{
				original_point_c = original_point_a;
				original_point_d = original_point_b;
			}
			if (i_original_img_wnum == src_width - 1)
			{
				original_point_b = original_point_a;
				original_point_d = original_point_c;
			}

			int pixel_point = i * lineByte2 + j * depth;//映射尺度变换图像数组位置偏移量    
			for (int k = 0; k < depth; k++)
			{
				output[pixel_point + k] =
					imagedata[original_point_a + k] * (1 - distance_to_a_x) * (1 - distance_to_a_y) +
					imagedata[original_point_b + k] * distance_to_a_x * (1 - distance_to_a_y) +
					imagedata[original_point_c + k] * distance_to_a_y * (1 - distance_to_a_x) +
					imagedata[original_point_d + k] * distance_to_a_y * distance_to_a_x;
			}

		}
	return 1;
}


int image_resize(buff imagedata, int src_width, int src_height, buff output, int dst_width, int dst_height, int depth, int method){
    if(method == 1)
    {
        return image_scaling_bilinear(imagedata, src_width, src_height, output, dst_width, dst_height, depth);
    }
    // method异常时 如何处理
    return 11101001; // 
}