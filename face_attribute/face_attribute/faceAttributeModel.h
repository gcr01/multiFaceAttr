#include "tensorflow_faceAttrModel.h"

class faceAttribute
{
    private:
        tf_faceAttrModel* faceAttr;
    public:
        faceAttribute(); //constrcut 
    private:
        int transfer_predict(const Attr_Data predict, std::string& predict_format);
    public:
        int faceAttrInit(char* aModelFile, int width, int height, int depth);
        int faceAttrPredict(unsigned char* buffer, int width, int height, std::string& predict_ouput);
        int faceAttrRelease();
};