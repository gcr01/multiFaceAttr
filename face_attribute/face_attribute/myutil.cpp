#include "myutil.h"
#if defined (WINDOWS)
#include <io.h>
#elif defined(unix) || defined(__unix__) || defined(__unix)
#include <unistd.h>
#endif
#include <string>
#include <fstream>

int myAccess(const char *name, int type)
{
    return access(name, type);
}

unsigned char* readFile2UChar(char* filename, int width, int height, int depth)
  {
    int width_bit = (width*depth+3)/4*4; //bytes per line
    unsigned char* txt = new unsigned char [width_bit*height];
    std::ifstream in_file(filename, std::ios::binary);
    if (!in_file.bad()) {
      in_file.read(reinterpret_cast<char*>(txt), width*height*depth);
    }
    return txt;
  }
