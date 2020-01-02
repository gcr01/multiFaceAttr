#include <string.h>
#include <vector>
#include "tensorflow/core/public/session.h"

// 注意多线程使用的问题，建立多个对象还是建立
struct Attr_Data
    {
      std::vector< std::vector<float>> smile;
      std::vector< std::vector<float>> gender;
      std::vector< std::vector<float>> glasses;
      std::vector< std::vector<float>> ethnic;
    };

class tf_faceAttrModel{
  private:
    std::unique_ptr<tensorflow::Session> session;
    std::string graph_path;
    std::string input_layer;
    std::vector<std::string> input_layers;
    std::string output_layer;
    std::vector<std::string> output_layers;
    tensorflow::Tensor dropout;
    tensorflow::Tensor b_train;
    int input_width;
    int input_height;
    int input_depth;
  private:
    tensorflow::Tensor read2Tensor(unsigned char * image_data, int src_width, int src_height, int new_width, int new_height, int depth);
    void normalTensor(tensorflow::Tensor& input_tensor);
    std::vector< std::vector< std::vector<float> > > D3Tensor2Vector(tensorflow::Tensor& tensor);
    std::vector< std::vector<float> > D2Tensor2Vector(tensorflow::Tensor& tensor);
    std::vector< std::vector<int> > D2Tensor2VectorINT(tensorflow::Tensor& tensor);
    tensorflow::Status LoadGraph(const std::string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session);
    int load_model(std::string graph_path, std::unique_ptr<tensorflow::Session> &session);
    int PredictFromTensor(
        std::vector<std::string> input_layers,
        std::vector<tensorflow::Tensor> &input_tensors,
        std::vector<std::string> output_layer,
        std::vector<tensorflow::Tensor> &outputs,
        const std::unique_ptr<tensorflow::Session>& session);
    int transfer_output(std::vector<tensorflow::Tensor> tensor_output, Attr_Data& attr_data);
  public:
    tf_faceAttrModel();
  private:
    int SetGraph(std::string graph);
  public:
    int predict(unsigned char* bgr_buff, Attr_Data& output, int src_width, int src_height);
    int initModel(std::string graph, int width, int height, int depth);
};