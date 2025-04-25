package com.example.mnnvits

import android.content.Context
import android.util.Log
import android.widget.Toast
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mnn_vits.MnnVitsJni
import com.example.mnnvits.utils.ChineseTextUtils
import com.example.mnnvits.utils.copyAssets2Local
import com.example.mnnvits.utils.saveWavFile
import com.google.gson.Gson
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.Continuation
import kotlin.coroutines.resume

/**
 * Author: Voine
 * Date: 2025/4/1
 * Description:
 */
class VoiceViewModel : ViewModel() {
    private val _uiState: MutableSharedFlow<UIState> = MutableSharedFlow(replay = 1, extraBufferCapacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    val uiState = _uiState.asSharedFlow()
    private val soundHandler: SoundPlayHandler by lazy {
        SoundPlayHandler()
    }
    private val vitsInferChannel by lazy {
        Channel<IntArray>(capacity = Int.MAX_VALUE, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    }

    private lateinit var cleaner: ChineseTextUtils
    private val symbolMap = hashMapOf<String, Int>()

    @Volatile
    private var currentSpkId = 228

    private lateinit var speakers: List<String>

    fun init(context: Context) {
        setLoading(true, "正在初始化...")
        viewModelScope.launch(Dispatchers.IO) {
            setLoading(true, "正在移动模型...")
            val absPath = initModel(context)

            setLoading(true, "正在初始化 VITS...")
            initVits(absPath)
            val configJson = File("${context.filesDir.absolutePath}/mnn/config.json")
            val config = Gson().fromJson(configJson.readText(), Config::class.java)
            val symbols = config.symbols
            symbols?.forEachIndexed { index, symbol ->
                symbolMap[symbol] = index
            }
            speakers = config.speakers ?: emptyList()
            initCharacters(speakers)
            setLoading(true, "正在初始化中文分词引擎...")
            cleaner =
                ChineseTextUtils(symbols = config.symbols!!, cleanerName = "chinese_cleaners", assetManager = context.assets)
            cleaner.initPinyinEngine()
            setDefaultState()
            startVoiceCheckLoop(context)
            setLoading(false)
        }
    }

    private fun startVoiceCheckLoop(context: Context) {
        viewModelScope.launch (Dispatchers.Main.immediate) {
            while (true) {
                val cleanedText = vitsInferChannel.receive()
                Log.d("runVits", "cleanedText start infer: ${cleanedText.toList()}")
                setLoading(true, "开始启动推理...")
                val startTime = System.currentTimeMillis()
                val result = withContext(Dispatchers.IO) {
                    MnnVitsJni.startVitsInfer(cleanedText, currentSpkId)
                }
                val endTime = System.currentTimeMillis()
                Toast.makeText(context, "推理耗时: ${endTime - startTime} ms", Toast.LENGTH_SHORT).show()
                Log.d("runVits", "result: ${result.joinToString(",", limit = 10)}")
                Log.d("runVits", "infer time: ${endTime - startTime} ms")
                setLoading(false)
                soundHandler.sendSound(result)
                if(BuildConfig.DEBUG){
                    launch(Dispatchers.IO) {
                        runCatching {
                            saveWavFile(
                                context,
                                context.filesDir.absolutePath,
                                result,
                                "output_${System.currentTimeMillis()}.wav"
                            )
                        }.onFailure {
                            Log.e("runVits", "saveWavFile error: ${it.message}")
                        }
                    }
                }
            }
        }
    }



    private suspend fun initModel(context: Context): String {
        return suspendCancellableCoroutine {
            context.copyAssets2Local(true, "mnn", context.filesDir.absolutePath) { isSuccess: Boolean, absPath: String ->
                Log.d("copyAssets2Local", "isSuccess: $isSuccess, absPath: $absPath")
                it.safeResume(absPath)
            }
        }
    }

    fun setDefaultState() {
        val sendState = _uiState.replayCache.firstOrNull() ?: UIState()
        Log.d("setDefaultState", "sendState: $sendState")
        _uiState.tryEmit(sendState.copy
            (inputText = "舰长，好久不见",
            selectedCharacter = speakers[228])
        )
    }


    fun initVits(absPath: String) {
        val encPath = File(absPath,"mnn/vits_uma_genshin_honkai_enc_fp16.mnn").absolutePath
        val decPath = File(absPath,"mnn/vits_uma_genshin_honkai_dec_fp16.mnn").absolutePath
        val flowPath = File(absPath,"mnn/vits_uma_genshin_honkai_flow_fp16.mnn").absolutePath
        val embPath = File(absPath,"mnn/vits_uma_genshin_honkai_emb_fp16.mnn").absolutePath
        val dpPath = File(absPath,"mnn/vits_uma_genshin_honkai_dp_fp16.mnn").absolutePath
        MnnVitsJni.loadLibrary()
        MnnVitsJni.initVitsLoader()
        MnnVitsJni.setVitsModelPath(
            encoderPath = encPath,
            decoderPath = decPath,
            flowPath = flowPath,
            embPath = embPath,
            dpPath = dpPath
        )
    }

    fun runVits(showText: String = "你好") {
        setLoading(true, "开始转义文本...")
        viewModelScope.launch(Dispatchers.IO) {
            val startTime = System.currentTimeMillis()
            val cleanedText = cleaner.convertText(showText)
            val endTime = System.currentTimeMillis()
            Log.d("runVits", "cleanedText time: ${endTime - startTime} ms")
            vitsInferChannel.trySend(cleanedText.firstOrNull()!!)
        }
    }

    fun initCharacters(speakers: List<String>?) {
        val sendState = _uiState.replayCache.firstOrNull() ?: UIState()
        _uiState.tryEmit(sendState.copy(characters = speakers ?: emptyList()))
    }

    fun updateInputText(string: String) {
        Log.d("updateInputText", "string: $string")
        val sendState = _uiState.replayCache.firstOrNull() ?: UIState()
        _uiState.tryEmit(sendState.copy(inputText = string))
    }

    fun startAudioInference(text: String) {
        runVits(text.trim())
    }

    fun selectCharacter(string: String) {
        val sendState = _uiState.replayCache.firstOrNull() ?: UIState()
        Log.d("selectCharacter", "string: $string")
        _uiState.tryEmit(sendState.copy(selectedCharacter = string))
        currentSpkId = speakers.indexOf(string)
    }

    fun setLoading(loading: Boolean, hint: String = "") {
        Log.d("setLoading", "loading: $loading, hint: $hint")
        val sendState = _uiState.replayCache.firstOrNull() ?: UIState()
        _uiState.tryEmit(sendState.copy(isLoading = loading, loadingHint = hint))
    }
}


data class UIState(
    val inputText: String = "",
    val selectedCharacter: String = "",
    val isLoading: Boolean = false,
    val loadingHint: String = "",
    val characters: List<String> = emptyList()
)

fun <T> CancellableContinuation<T>.safeResume(value: T) {
    if (this.isActive) {
        (this as? Continuation<T>)?.resume(value)
    }
}


data class Config(val data: MData?, val symbols: List<String>?, val speakers: List<String>?) {
    data class MData(
        val text_cleaners: List<String>?,
        val sampling_rate: Int?,
        val n_speakers: Int?,
    )
}
