// define Handle
#include <iostream>
typedef void * handle;

int faceAttrInit(char *aModelFile, handle* faceAttrHandle);
int faceAttrPredict(unsigned char* buffer, int width, int height, std::string& output, handle* faceAttrHandle);
int faceAttrFree(handle* faceAttrHandle);