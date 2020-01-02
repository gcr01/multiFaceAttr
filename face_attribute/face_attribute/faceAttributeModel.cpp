#include "faceAttributeModel.h"

faceAttribute::faceAttribute()
{
    faceAttr = NULL;
}

int faceAttribute::faceAttrInit(char *aModelFile, int width, int height, int depth)
{
    faceAttr = new tf_faceAttrModel();
    int ret = faceAttr->initModel(aModelFile, width, height, depth);
    if (ret != 1)
    {
        delete faceAttr;
        faceAttr = NULL;
    }
    return ret;
}

int faceAttribute::faceAttrPredict(unsigned char *buffer, int width, int height, std::string &predict_ouput)
{
    if (faceAttr == NULL)
    {
        return 11101001;
    }
    Attr_Data predict_result;
    int ret = faceAttr->predict(buffer, predict_result, width, height);
    if (ret != 1)
    {
        return ret;
    }
    transfer_predict(predict_result, predict_ouput);
    return ret;
}

int faceAttribute::faceAttrRelease()
{
    if (faceAttr == NULL)
    {
        return 11100001; // 11100001 模型为空
    }
    delete faceAttr;
    return 1;
}

int faceAttribute::transfer_predict(const Attr_Data predict, std::string& predict_format)
{
    predict_format.clear();
    predict_format += "smile: ";
    std::vector<std::vector<float>> smile = predict.smile;
    std::vector<std::vector<float>> gender = predict.gender;
    std::vector<std::vector<float>> glasses = predict.glasses;
    std::vector<std::vector<float>> ethnic = predict.ethnic;

    for (int i = 0; i < predict.smile.size(); i++)
    {
        for (int j = 0; j < smile[i].size(); j++)
        {
            predict_format += std::to_string(smile[i][j]) + ",";
        }
    }
    predict_format += "gender: ";
    for (int i = 0; i < gender.size(); i++)
    {
        for (int j = 0; j < gender[i].size(); j++)
        {
            predict_format += std::to_string(gender[i][j]) + ",";
        }
    }

    predict_format += "glasses: ";
    for (int i = 0; i < glasses.size(); i++)
    {
        for (int j = 0; j < glasses[i].size(); j++)
        {
            predict_format += std::to_string(glasses[i][j]) + ",";
        }
    }

    predict_format += "ethnic: ";
    for (int i = 0; i < ethnic.size(); i++)
    {
        for (int j = 0; j < ethnic[i].size(); j++)
        {
            predict_format += std::to_string(ethnic[i][j]) + ",";
        }
    }
    return 1;
}