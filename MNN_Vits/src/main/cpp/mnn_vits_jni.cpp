#include <jni.h>
#include <string>
#include "mnn_vits_loader.hpp"

extern "C"
JNIEXPORT void JNICALL
Java_com_example_mnn_1vits_MnnVitsJni_initVitsLoader(JNIEnv *env, jobject thiz) {
    MNN_VITS::init_vits_loader();
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_mnn_1vits_MnnVitsJni_destroyVitsLoader(JNIEnv *env, jobject thiz) {
    MNN_VITS::destroy_vits_loader();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_mnn_1vits_MnnVitsJni_setVitsModelPath(JNIEnv *env, jobject thiz,
                                                       jstring encoder_path, jstring decoder_path,
                                                       jstring flow_path, jstring emb_path,
                                                       jstring dp_path) {
    const char* encPathCStr = env->GetStringUTFChars(encoder_path, nullptr);
    const char* decPathCStr = env->GetStringUTFChars(decoder_path, nullptr);
    const char* flowPathCStr = env->GetStringUTFChars(flow_path, nullptr);
    const char* embPathCStr = env->GetStringUTFChars(emb_path, nullptr);
    const char* dpPathCStr = env->GetStringUTFChars(dp_path, nullptr);
    MNN_VITS::set_vits_model_path(encPathCStr, decPathCStr, dpPathCStr, embPathCStr, flowPathCStr);
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_mnn_1vits_MnnVitsJni_startVitsInfer(JNIEnv *env, jobject thiz,
                                                     jintArray input_seq, jint spkid) {
    jsize length = env->GetArrayLength(input_seq);
    jint* input_seq_elements = env->GetIntArrayElements(input_seq, nullptr);
    std::vector<int> input_seq_vector(input_seq_elements, input_seq_elements + length);
    std::vector<float> audio = MNN_VITS::start_audio_infer(input_seq_vector, spkid);
    env->ReleaseIntArrayElements(input_seq, input_seq_elements, 0);
    jfloatArray jResult = env->NewFloatArray(static_cast<jsize>(audio.size()));
    if (jResult == nullptr) {
        return nullptr; // Out of memory error thrown
    }
    env->SetFloatArrayRegion(jResult, 0, static_cast<jsize>(audio.size()), audio.data());
    return jResult;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_mnn_1vits_MnnVitsJni_setAudioLengthScale(JNIEnv *env, jobject thiz,
                                                            jfloat length_scale) {
    MNN_VITS::set_length_scale(length_scale);
}