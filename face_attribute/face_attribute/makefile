# This Makefile compiles a example for Tensorflow deploy model.
# First need is to run Tensorflow build Program \
# `tensorflow/contrib/makefile/build_all_linux.sh`
# 
# Just setup for linux

# Setting up compiler
CC := gcc
AR := ar
# Setting up Tensorflow Path
ROOT_TENSORFLOW = /home/leilei/tensorflow-build/tensorflow-1.12.2-2/
# 编译其他文件
WORKSPACE_PATH=/home/leilei/tensorflow-build/tensorflow-1.12.2-2/example/face_attribute/

INCLUDE_PATH := \
-I. \
-I$(ROOT_TENSORFLOW) \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/downloads/ \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/downloads/eigen \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/downloads/gemmlowp \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/downloads/nsync/public \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/downloads/fft2d \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/downloads/double_conversion \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/downloads/absl \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/gen/proto/ \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/gen/proto_text/ \
-I$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/gen/protobuf-host/include \
-I/usr/local/include

CXXFLAGS=--std=c++11 -DIS_SLIM_BUILD -fno-exceptions -DNDEBUG -O3 -march=native -fPIC
ARFLAGS=crv

STATIC_LIBRARY :=\
-Wl,--allow-multiple-definition -Wl,--whole-archive  $(ROOT_TENSORFLOW)tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a \
-Wl,--no-whole-archive $(ROOT_TENSORFLOW)tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/nsync.a \
-Wl,--whole-archive $(ROOT_TENSORFLOW)tensorflow/contrib/makefile/gen/protobuf-host/lib/libprotobuf.a

LIBRARY_PATH=-L$(ROOT_TENSORFLOW)tensorflow/contrib/makefile/gen/protobuf-host/lib

DYNAMIC_LIBRARY=\
-lstdc++ \
-lz \
-lm \
-ldl \
-lpthread 


STATIC_FILE := \
$(WORKSPACE_PATH)face_attribute_model.cpp \
$(WORKSPACE_PATH)faceAttributeModel.cpp \
$(WORKSPACE_PATH)image_process.cpp \
$(WORKSPACE_PATH)myutil.cpp \
$(WORKSPACE_PATH)tensorflow_faceAttrModel.cpp


STATIC_OBJ := $(STATIC_FILE:%.cpp=%.o)

LIB_NAME := libfaceAttrTest.a

static: LIB_NAME

LIB_NAME: $(STATIC_FILE)
	$(CC) $(CXXFLAGS) $(INCLUDE_PATH) \
	-c $(STATIC_FILE) 
	$(AR) $(ARFLAGS) $(LIB_NAME) $(STATIC_OBJ) 
	rm *.o

test: demo.cpp LIB_NAME
	$(CC) $(CXXFLAGS) $(INCLUDE_PATH) \
	-o demo \
	$(WORKSPACE_PATH)demo.cpp \
	$(WORKSPACE_PATH)$(LIB_NAME) \
	$(STATIC_LIBRARY) 
	# \
	# $(DYNAMIC_LIBRARY)


.PHONEY:clean clean_all

clean:
	rm *.a