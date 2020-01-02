#include <iostream>
#include "myutil.h"
#include "face_attribute_model.h"

using std::cout;
using std::endl;

int main(int argc, char** argv){
    if(argc<4){
        // to-do
        std::cout << "Usage: demo image_filename w h" << endl;
        return 1;
    }
    cout << "Usage： " << argv[0] << endl
         << "filename: " << argv[1] <<endl
         << "width: " << argv[2] <<endl
         << "Height: " << argv[3] <<endl;
    handle faceAttrModel;
    char* model_file = "/home/leilei/tensorflow-build/4takssModel/face_model.pb";
    int ret = faceAttrInit(model_file, &faceAttrModel);
    if(ret != 1)
    {
        cout<<"faceInit error with code: "<<ret<<endl;
    }
    // char* im_file = argv[1];
    int w = std::atoi(argv[2]);
    int h = std::atoi(argv[3]);
    unsigned char* im_buff = readFile2UChar(argv[1], w,h, 1);
    std::string predict_result;
    ret = faceAttrPredict(im_buff, w, h, predict_result, &faceAttrModel);
    if(ret != 1)
    {
        cout << "facePredict error with code: "<<ret<<endl;
    }
    cout << predict_result << endl;
    ret = faceAttrFree(&faceAttrModel);
    if(ret != 1)
    {
        cout<<"faceAttrFree error with code: " << ret << endl;
    }
    delete [] im_buff;
}