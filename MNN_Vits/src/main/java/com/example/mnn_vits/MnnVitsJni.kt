package com.example.mnn_vits

object MnnVitsJni {

    @JvmStatic
    fun loadLibrary() {
        System.loadLibrary("MNN_Express")
        System.loadLibrary("MNN")
        System.loadLibrary("mnn_vits")
    }

    external fun initVitsLoader()

    external fun destroyVitsLoader()

    external fun setVitsModelPath(
        encoderPath: String,
        decoderPath: String,
        flowPath: String,
        embPath: String,
        dpPath: String,
    )

    external fun startVitsInfer(inputSeq: IntArray, sid: Int = 228): FloatArray
}