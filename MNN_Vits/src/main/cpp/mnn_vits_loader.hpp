#ifndef VITS_LOADER_HPP
#define VITS_LOADER_HPP
#include <android/log.h>
#include <string>
#include <vector>
#define MNN_VITS_PRINT(format, ...)  __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
namespace MNN_VITS {

// 初始化 Vits 加载器，内部会创建全局资源（例如执行器）
    void init_vits_loader();

// 销毁 Vits 加载器，释放全局资源
    void destroy_vits_loader();

// 设置模型路径，内部会保存路径供其他方法使用
    void set_vits_model_path(
            const std::string &enc_model_path,
            const std::string &dec_model_path,
            const std::string &dp_model_path,
            const std::string &emb_model_path,
            const std::string &flow_model_path);

    void set_length_scale(float length_scale);

// 开始音频推理，返回推理结果（音频数据数组）
    std::vector<float> start_audio_infer(const std::vector<int>& input_seq, int spkid);
} // namespace vits

#endif // VITS_LOADER_HPP
