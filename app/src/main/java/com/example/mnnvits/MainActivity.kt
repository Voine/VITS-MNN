package com.example.mnnvits

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.ViewModelProvider
import com.example.mnnvits.ui.theme.MNNVitsTheme

class MainActivity : ComponentActivity() {

    private val viewModel: VoiceViewModel by lazy {
        ViewModelProvider(this)[VoiceViewModel::class.java]
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            VoiceGenerationScreen(viewModel)
        }
        viewModel.init(this)
    }
}
