#include <iostream>
#include <fstream>
#include <stdio.h>
#include "tensorflow_faceAttrModel.h"
#include "image_process.h"

using namespace tensorflow;
using std::cout;
using std::endl;
using std::vector;

//读入一(批)原图，返回resize后的图片的tensor
tensorflow::Tensor tf_faceAttrModel::read2Tensor(unsigned char *image_data, int src_width, int src_height, int new_width, int new_height, int depth)
{
    const int width_byte = (new_width * depth + 3) / 4 * 4;
    tensorflow::Tensor input_tensor(DT_FLOAT, TensorShape({1, new_height, new_width, depth})); // 构造Tensor类
    unsigned char *resize_image = new unsigned char[width_byte * new_height];
    // 没有检测返回值,可能出现bug，注意
    int b_result = image_resize(image_data, src_width, src_height, resize_image, new_width, new_height, depth, 1);

    auto tmap = input_tensor.tensor<float, 4>(); // 构建四维tensor

    for (int i = 0; i < new_width; i++)
    {
        for (int j = 0; j < new_height; j++)
        {
            for (int k = 0; k < depth; k++)
            {
                tmap(0, j, i, k) = (float)(resize_image[j * width_byte + i * depth + k]);
            }
        }
    }
    // delete [] resize_image;
    return input_tensor;
}

// 自写，需要再检查  ,用来归一化
void tf_faceAttrModel::normalTensor(tensorflow::Tensor &input_tensor)
{
    auto tmap = input_tensor.tensor<float, 4>();
    int dims = input_tensor.dims();
    assert(dims == 4);
    int shape[4] = {};
    for (int i = 0; i < dims; i++)
    {
        shape[i] = input_tensor.dim_size(i); //batch_size,width,height,depth
    }
    for (int i = 0; i < shape[0]; ++i)
    {
        for (int j = 0; j < shape[1]; ++j)
        {
            for (int k = 0; k < shape[2]; ++k)
            {
                for (int channel = 0; channel < shape[3]; ++channel)
                {
                    tmap(i, j, k, channel) -= 128.0;
                    tmap(i, j, k, channel) /= 255.0;
                }
            }
        }
    }
}

//三维tensor转vector
std::vector<vector<vector<float>>> tf_faceAttrModel::D3Tensor2Vector(tensorflow::Tensor &tensor)
{
    int dtype = tensor.dtype();
    std::vector<int> shape;
    for (int i = 0; i < 3; ++i)
    {
        int dim_size = tensor.dim_size(i);
        shape.push_back(dim_size);
    }
    auto tmap = tensor.tensor<float, 3>(); // 构建三维tensor 
    vector<vector<vector<float>>> batch;
    for (int i = 0; i < shape[0]; i++)
    {
        vector<vector<float>> sample;
        for (int j = 0; j < shape[1]; j++)
        {
            std::vector<float> channel;
            if (tmap(i, j, 0) < 0)
            {
                break;
            }
            for (int k = 0; k < shape[2]; k++)
            {
                if (tmap(i, j, k) < 0)
                {
                    break;
                }
                channel.push_back(tmap(i, j, k));
            }
            sample.push_back(channel);
        }
        batch.push_back(sample);
    }
    return batch;
}

std::vector<vector<float>> tf_faceAttrModel::D2Tensor2Vector(tensorflow::Tensor &tensor)
{
    int dtype = tensor.dtype();
    std::vector<int> shape;
    for (int i = 0; i < 2; ++i)
    {
        int dim_size = tensor.dim_size(i);
        shape.push_back(dim_size);
    }
    auto tmap = tensor.tensor<float, 2>(); // 构建二维tensor
    vector<vector<float>> batch;
    for (int i = 0; i < shape[0]; i++)
    {
        vector<float> sample;
        for (int j = 0; j < shape[1]; j++)
        {
            if (tmap(i, j) < 0)
            {
                break;
            }
            sample.push_back(tmap(i, j));
        }
        batch.push_back(sample);
    }
    return batch;
}

std::vector<vector<int>> tf_faceAttrModel::D2Tensor2VectorINT(tensorflow::Tensor &tensor)
{
    int dtype = tensor.dtype();
    std::vector<int> shape;
    for (int i = 0; i < 2; ++i)
    {
        int dim_size = tensor.dim_size(i);
        shape.push_back(dim_size);
    }
    auto tmap = tensor.tensor<int, 2>(); // 构建二维tensor
    vector<vector<int>> batch;
    for (int i = 0; i < shape[0]; i++)
    {
        vector<int> sample;
        for (int j = 0; j < shape[1]; j++)
        {
            if (tmap(i, j) < 0)
            {
                break;
            }
            sample.push_back(tmap(i, j));
        }
        batch.push_back(sample);
    }
    return batch;
}

Status tf_faceAttrModel::LoadGraph(const string &graph_file_name,
                                   std::unique_ptr<tensorflow::Session> *session)
{
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok())
    {
        load_graph_status = ReadTextProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    }

    if (!load_graph_status.ok())
    {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok())
    {
        return session_create_status;
    }
    return Status::OK();
}

int tf_faceAttrModel::load_model(std::string graph_path, std::unique_ptr<tensorflow::Session> &session)
{
    // string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok())
    {
        LOG(ERROR) << load_graph_status;
        return 11100101; // 模型载入失败
    }
    return 1;
}

int tf_faceAttrModel::PredictFromTensor(
    std::vector<std::string> input_layers,
    std::vector<Tensor> &input_tensors,
    std::vector<std::string> output_layer,
    std::vector<Tensor> &outputs,
    const std::unique_ptr<tensorflow::Session> &session)
{
    std::vector<std::pair<string, Tensor>> inputs;
    for (int i = 0; i < input_layers.size(); i++)
    {
        inputs.push_back({input_layers[i], input_tensors[i]});
    }
    Status run_status =
        session->Run(inputs, output_layer, {}, &outputs);
    if (!run_status.ok())
    {
        LOG(ERROR) << "Running model failed: " << run_status;
        return 11100103; // 模型运行错误
    }
    return 1;
}

tf_faceAttrModel::tf_faceAttrModel()
{
    input_layer = "Placeholder:0,Placeholder_3:0,Placeholder_4:0";  // x, phase_train, keep_prob
    output_layer =
        "smile_softmax/Softmax:0,gender_softmax/Softmax:0,glasses_softmax/Softmax:0,ethnic_softmax/Softmax:0";
}

int tf_faceAttrModel::SetGraph(std::string graph)
{
    graph_path = graph;
    return graph_path.empty();
}

int tf_faceAttrModel::initModel(std::string graph, int width, int height, int depth)
{
    tensorflow::Tensor dropout_init(DT_FLOAT, TensorShape());
    auto tmap = dropout_init.scalar<float>();
    tmap() = 1.0;
    dropout = dropout_init;

    tensorflow::Tensor train_init(DT_BOOL, TensorShape());
    auto tmap_2 = train_init.scalar<bool>();
    tmap_2() = false;
    b_train = train_init;

    if (SetGraph(graph))
    {
        return 11100102; // 11100102: graph路径未设置
    }
    output_layers = tensorflow::str_util::Split(output_layer, ',');
    input_layers = tensorflow::str_util::Split(input_layer, ',');

    int result = load_model(graph_path, session);
    if (result != 1)
    {
        return result;
    }
    input_width = width;
    input_height = height;
    input_depth = depth;
    return 1;
}

int tf_faceAttrModel::predict(unsigned char *bgr_buff, Attr_Data& output, int src_width, int src_height)
{
    tensorflow::Tensor image_tensor = read2Tensor(bgr_buff, src_width, src_height, input_width, input_height, input_depth); //read img from buff array to tensor
    normalTensor(image_tensor);  //normalize, (pxiel-128)/255 (for every pixel)
    std::vector<tensorflow::Tensor> tensor_outputs; 
    std::vector<tensorflow::Tensor> input_tensors;
    input_tensors.push_back(image_tensor);
    input_tensors.push_back(b_train);
    input_tensors.push_back(dropout);  //[img_tensor, phase_train_tensor, keep_prob_tensor]
    int run_status = PredictFromTensor(input_layers, input_tensors, 
                                       output_layers, tensor_outputs, session);

    if(run_status != 1)
    {
        return run_status;
    }
    run_status = transfer_output(tensor_outputs, output);
    return run_status;
}

int tf_faceAttrModel::transfer_output(std::vector<tensorflow::Tensor> tensor_output, Attr_Data& attr_data)
{
    vector<vector<float>> smile = D2Tensor2Vector(tensor_output[0]);
    vector<vector<float>> gender = D2Tensor2Vector(tensor_output[1]);
    vector<vector<float>> glasses = D2Tensor2Vector(tensor_output[2]);
    vector<vector<float>> ethnic = D2Tensor2Vector(tensor_output[3]);

    attr_data.smile = smile;
    attr_data.gender =gender;
    attr_data.glasses = glasses;
    attr_data.ethnic = ethnic;
    return 1;
}