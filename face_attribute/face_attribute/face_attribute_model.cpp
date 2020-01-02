#include "face_attribute_model.h"
#include <iostream>
#include "myutil.h"
#include "faceAttributeModel.h"

// 如何实现输入字符串的判断
int faceAttrInit(char *aModelFile, handle* faceAttrHandle)
{
    // judge model file full path
    if(myAccess(aModelFile, 4) == -1)
    {
        return 11100001;
    }
    faceAttribute* faceAttr = new faceAttribute;
    int ret = faceAttr->faceAttrInit(aModelFile, 48, 48, 1);
    if(ret != 1)
    {
        delete faceAttr;
        faceAttr = NULL;
    }
    *faceAttrHandle = faceAttr;
    return ret;    
}

int faceAttrPredict(unsigned char* buffer, int width, int height, std::string& output, handle* faceAttrHandle)
{
    if(*faceAttrHandle == NULL)
    {
        return 1111020; // 1111020 预测时模型指针为空
    }
    faceAttribute* faceAttr = (faceAttribute*)(*faceAttrHandle);  //change the void*  to faceAttribute*
    int ret = faceAttr->faceAttrPredict(buffer, width, height, output);
    return ret;
}

int faceAttrFree(handle* faceAttrHandle)
{
    faceAttribute* faceAttr = (faceAttribute*)(*faceAttrHandle);
    faceAttr->faceAttrRelease();
}
