# VITS-MNN

> ✨ [VITS](https://github.com/Plachtaa/VITS-fast-fine-tuning) Android 版, 推理框架基于 [alibaba-MNN](https://github.com/alibaba/MNN).

---

## 🧠 简介

本工程提供了一个示例，基于 MNN 实现了离线推理版本的 VITS，目前仅适配了中文，在 [VITS-uma-genshin-honkai](https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/tree/main) 上进行了验证：

- 🏗 **MNN** ：基于 MNN 推理框架实现 VITS 的整个推理流程，推理参考自 [script](onnx_export/script) 内的 onnx 推理代码。
- 🧠 如果你希望用效果更好一些的模型，可以尝试 [Bert-VITS2-MNN](https://github.com/Voine/Bert-VITS2-MNN), 但本仓库版本模型体积会更小。

整个过程在 Android 端全程 **离线推理** 无需任何联网服务.


---


## 🎵 示例音频

此处提供一些中文音频示例:

| Text               | Character | Audio                                                                                      |
|--------------------|-----------|--------------------------------------------------------------------------------------------|
| 博士，当初在龙门，我不该放你走的。  | 陈         | 🔊 [Play](https://github.com/user-attachments/assets/a6fc4022-e473-41e3-89da-0f5c9741a4c4) |
| 旅行者，好久不见。          | 珐露珊       | 🔊 [Play](https://github.com/user-attachments/assets/60a96546-1e18-43b8-9a6a-3c9bfd5eca42) |
| 工作还没有做完，又要开始搬砖了。   | 甘雨        | 🔊 [Play](https://github.com/user-attachments/assets/7482e892-630f-47ee-829f-336ceb9525c4)                                                   |

---

## ⚡ 本地编译指南

### Clone with submodules

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules git@github.com:Voine/VITS-MNN.git

# for windows powershell
$env:GIT_LFS_SKIP_SMUDGE=1; git clone --recurse-submodules git@github.com:Voine/VITS-MNN.git

cd VITS-MNN
```

If already cloned:

```bash
git submodule update --init --recursive
```

### Build for Android

> 📦 建议使用 Android Studio 进行工程编译，用 IDE 打开根目录即可

```bash
# From project root
./gradlew assembleRelease
```

---

## 🛁 Git LFS

本工程的一些文件如 `.mnn` `.onnx` ，使用 lfs 进行存储，需要按照如下方式拉代码：

```bash
git lfs install
git lfs pull
```

To track files (if contributing):

```bash
git lfs track "*.mnn"
```

---

## 🛠️ Submodule 依赖

| Library      | Path                         |
|--------------|------------------------------|
| [MNN](https://github.com/alibaba/MNN)        | `third_party/MNN`            |


---


## 💡 关于 - 自制模型替换 -

1. 如果你需要替换自己的模型尝试验证，首先需要参考 [VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning) 内的说明进行训练得到桌面端模型, 作者基于 [VITS-uma-genshin-honkai](https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/tree/main) 模型进行了验证，其他的 VITS 仓库模型暂未验证
2. 将你的 pth 模型转换成 onnx, onnx 导出脚本在  [script](onnx_export/script)
3. 使用 [MNN Convert](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html) 将所有模块的 onnx 模型转成 mnn
4. 放到 assets/mnn 内，如果你的模型名字有变化，则需要修改 VoiceViewModel.kt 内关于模型路径加载的部分。

---

## 💡 关于 - third_party -

目前在 third_party 内的 MNN 仅是为了提供头文件

---

## 📋 工程大体结构

```
├── app/
├──── src/main/                 
│           ├── assets                    # mnn vits model
│           ├── java/ChineseCleaners      # Text preprocess code
├── MNN_Vits                              # VITS infer code
├── third_party                           # provide hpp

```

---

## 💡 类似仓库

- [VITS-Android-ncnn](https://github.com/weirdseed/Vits-Android-ncnn)
- [Sherpa-onnx-tts-android](https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxTts)

---

## 🙌 鸣谢

本工程基于以下前辈们的贡献做了一些微不足道的搬砖工作，也希望能为后续在端智能推理捣鼓的小伙伴提供一些参考。

- [VITS](https://github.com/jaywalnut310/vits)
- [VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)
- [VITS-uma-genshin-honkai](https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/tree/main)
- [MNN](https://github.com/alibaba/MNN)


---

## 免责声明
### 本项目仅供学习交流使用，禁止用于商业用途，作者纯为爱发电搞着玩的。

### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 严禁用于任何政治相关用途。

---
