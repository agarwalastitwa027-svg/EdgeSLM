package com.edgeslm.app

import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

private sealed interface Screen {
    data object Picking : Screen
    data class Importing(val name: String, val bytesDone: Long, val totalBytes: Long) : Screen
    data object LoadingModel : Screen
    data class Chat(val modelName: String, val backend: String) : Screen
    data class Error(val message: String) : Screen
}

private data class ChatMessage(val isUser: Boolean, val text: String)

/** True if the tail of [text] is an exact back-to-back repeat of the block right before it. */
private fun isRepeatingLoop(text: String, blockSize: Int = 40): Boolean {
    if (text.length < blockSize * 2) return false
    val a = text.substring(text.length - blockSize)
    val b = text.substring(text.length - blockSize * 2, text.length - blockSize)
    return a == b
}

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        LlamaBridge.init(applicationInfo.nativeLibraryDir)

        setContent {
            val dark = isSystemInDarkTheme()
            val colorScheme = when {
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && dark -> dynamicDarkColorScheme(this)
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> dynamicLightColorScheme(this)
                dark -> darkColorScheme(primary = Color(0xFF9ECAFF))
                else -> lightColorScheme(primary = Color(0xFF3E6DC9))
            }
            MaterialTheme(colorScheme = colorScheme) {
                Surface(modifier = Modifier.fillMaxSize()) {
                    EdgeSlmApp()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        LlamaBridge.shutdown()
    }
}

@Composable
private fun EdgeSlmApp() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()

    var screen by remember { mutableStateOf<Screen>(Screen.Picking) }
    var localModels by remember { mutableStateOf(ModelManager.localModels(context)) }
    var downloadModels by remember { mutableStateOf(listOf<DetectedModel>()) }
    var scanning by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        downloadModels = withContext(Dispatchers.IO) { ModelManager.scanDownloadsForGguf(context) }
        scanning = false
    }

    fun loadModelFile(file: File) {
        screen = Screen.LoadingModel
        scope.launch {
            val ok = withContext(Dispatchers.Default) { LlamaBridge.loadModel(file.absolutePath) }
            screen = if (ok) {
                Screen.Chat(file.name, LlamaBridge.activeBackend())
            } else {
                Screen.Error("Failed to load ${file.name}. It may be corrupt or an unsupported GGUF format.")
            }
        }
    }

    fun importAndLoad(uri: Uri, suggestedName: String) {
        scope.launch {
            var total = 0L
            screen = Screen.Importing(suggestedName, 0, 0)
            val file = withContext(Dispatchers.IO) {
                ModelManager.importModel(context, uri, suggestedName) { done ->
                    total = done
                    screen = Screen.Importing(suggestedName, done, total)
                }
            }
            localModels = ModelManager.localModels(context)
            loadModelFile(file)
        }
    }

    val pickLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            val name = ModelManager.displayNameFromUri(context, uri)
            if (!name.endsWith(".gguf", ignoreCase = true)) {
                screen = Screen.Error("\"$name\" is not a .gguf model file.")
            } else {
                importAndLoad(uri, name)
            }
        }
    }

    Box(Modifier.fillMaxSize().safeDrawingPadding()) {
        when (val s = screen) {
            is Screen.Picking -> ModelPickerScreen(
                scanning = scanning,
                localModels = localModels,
                downloadModels = downloadModels,
                onPickLocal = { loadModelFile(it) },
                onPickDownload = { importAndLoad(it.sourceUri!!, it.displayName) },
                onBrowse = { pickLauncher.launch(arrayOf("*/*")) },
            )
            is Screen.Importing -> ProgressScreen(
                title = "Importing ${s.name}",
                subtitle = if (s.totalBytes > 0) "${s.bytesDone / (1 shl 20)} MiB copied" else "Copying...",
            )
            is Screen.LoadingModel -> ProgressScreen(title = "Loading model onto GPU", subtitle = "This can take a moment for large models")
            is Screen.Chat -> ChatScreen(
                modelName = s.modelName,
                backend = s.backend,
                onSwitchModel = {
                    LlamaBridge.unloadModel()
                    localModels = ModelManager.localModels(context)
                    screen = Screen.Picking
                },
            )
            is Screen.Error -> ErrorScreen(message = s.message, onDismiss = { screen = Screen.Picking })
        }
    }
}

@Composable
private fun ModelPickerScreen(
    scanning: Boolean,
    localModels: List<DetectedModel>,
    downloadModels: List<DetectedModel>,
    onPickLocal: (File) -> Unit,
    onPickDownload: (DetectedModel) -> Unit,
    onBrowse: () -> Unit,
) {
    Column(Modifier.fillMaxSize().padding(horizontal = 20.dp)) {
        Spacer(Modifier.height(24.dp))
        Text("EdgeSLM", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Text(
            "Runs small language models fully on-device via the Adreno GPU. Nothing is ever downloaded automatically - pick a GGUF model you already have.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 6.dp, bottom = 20.dp),
        )

        Button(
            onClick = onBrowse,
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(16.dp),
        ) {
            Icon(Icons.Filled.FolderOpen, contentDescription = null, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
            Text("Browse for a .gguf file")
        }

        Spacer(Modifier.height(24.dp))

        if (scanning) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Text("Scanning for models...", style = MaterialTheme.typography.bodyMedium)
            }
        }

        LazyColumn(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            if (localModels.isNotEmpty()) {
                item {
                    Text(
                        "Ready to run",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(top = 8.dp, bottom = 6.dp),
                    )
                }
                items(localModels) { m ->
                    ModelRow(m.displayName, m.sizeBytes, "Load", filled = true) { onPickLocal(m.localFile!!) }
                }
            }

            if (downloadModels.isNotEmpty()) {
                item {
                    Text(
                        "Found in Downloads",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(top = 16.dp, bottom = 6.dp),
                    )
                }
                items(downloadModels) { m ->
                    ModelRow(m.displayName, m.sizeBytes, "Import & load", filled = false) { onPickDownload(m) }
                }
            }

            if (!scanning && localModels.isEmpty() && downloadModels.isEmpty()) {
                item {
                    Text(
                        "No GGUF models detected yet. Download one from a source you trust (e.g. Hugging Face) and either save it to Downloads or browse to it above.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 24.dp),
                    )
                }
            }
        }
        Spacer(Modifier.height(12.dp))
    }
}

@Composable
private fun ModelRow(name: String, sizeBytes: Long, actionLabel: String, filled: Boolean, onClick: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (filled) MaterialTheme.colorScheme.secondaryContainer
            else MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Row(
            Modifier.fillMaxWidth().padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Filled.Memory,
                contentDescription = null,
                modifier = Modifier.size(28.dp),
                tint = MaterialTheme.colorScheme.primary,
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(name, fontWeight = FontWeight.Medium, maxLines = 1, style = MaterialTheme.typography.bodyLarge)
                Text(
                    "%.2f GiB".format(sizeBytes / 1024.0 / 1024.0 / 1024.0),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(Modifier.width(8.dp))
            Button(onClick = onClick, shape = RoundedCornerShape(12.dp)) { Text(actionLabel) }
        }
    }
}

@Composable
private fun ProgressScreen(title: String, subtitle: String) {
    Column(
        Modifier.fillMaxSize().padding(20.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(20.dp))
        Text(title, style = MaterialTheme.typography.titleMedium, textAlign = TextAlign.Center)
        Spacer(Modifier.height(4.dp))
        Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
    }
}

@Composable
private fun ErrorScreen(message: String, onDismiss: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(20.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Something went wrong", style = MaterialTheme.typography.titleMedium)
        Text(message, style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.Center, modifier = Modifier.padding(vertical = 12.dp))
        Button(onClick = onDismiss) { Text("Back") }
    }
}

@Composable
private fun ChatScreen(modelName: String, backend: String, onSwitchModel: () -> Unit) {
    val scope = rememberCoroutineScope()
    var input by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }
    // Qwen3 (and similar hybrid-reasoning models) honor a "/no_think" directive to skip the
    // <think> block entirely - default this off since extended thinking is what was circling.
    var thinkingEnabled by remember { mutableStateOf(false) }
    var tokensGenerated by remember { mutableStateOf(0) }
    val messages = remember { mutableStateListOf<ChatMessage>() }
    val listState = rememberLazyListState()

    fun send() {
        val prompt = input.trim()
        if (prompt.isEmpty() || generating) return
        input = ""
        val effectivePrompt = if (thinkingEnabled) prompt else "$prompt /no_think"
        messages.add(ChatMessage(isUser = true, text = prompt))
        messages.add(ChatMessage(isUser = false, text = ""))
        generating = true
        tokensGenerated = 0
        scope.launch {
            val started = withContext(Dispatchers.Default) { LlamaBridge.startGeneration(effectivePrompt) }
            if (!started) {
                messages[messages.lastIndex] = ChatMessage(isUser = false, text = "(generation failed to start)")
                generating = false
                return@launch
            }
            var repeatStreak = 0
            while (true) {
                val chunk = withContext(Dispatchers.Default) { LlamaBridge.nextToken() } ?: break
                tokensGenerated++
                if (chunk.isNotEmpty()) {
                    val current = messages[messages.lastIndex]
                    val newText = current.text + chunk
                    messages[messages.lastIndex] = current.copy(text = newText)
                    listState.animateScrollToItem(messages.lastIndex)

                    // Safety net: if the model starts emitting the same block of text back-to-back
                    // (a "circling" loop the sampler didn't catch), stop rather than hang forever -
                    // the same kind of guard coding assistants use against runaway completions.
                    if (isRepeatingLoop(newText)) {
                        repeatStreak++
                        if (repeatStreak >= 3) {
                            messages[messages.lastIndex] = current.copy(
                                text = newText + "\n\n_(stopped: detected a repeating loop)_"
                            )
                            break
                        }
                    } else {
                        repeatStreak = 0
                    }
                }
            }
            generating = false
        }
    }

    Column(Modifier.fillMaxSize()) {
        TopAppBar(
            title = {
                Column {
                    Text(modelName, style = MaterialTheme.typography.titleMedium, maxLines = 1)
                    Text(
                        if (generating) "Generating... ($tokensGenerated tokens)" else "GPU: $backend",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            },
            actions = {
                FilterChip(
                    selected = thinkingEnabled,
                    onClick = { thinkingEnabled = !thinkingEnabled },
                    label = { Text("Think") },
                    modifier = Modifier.padding(end = 4.dp),
                )
                IconButton(onClick = onSwitchModel) {
                    Icon(Icons.Filled.SwapHoriz, contentDescription = "Switch model")
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
        )

        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(messages) { msg -> MessageBubble(msg) }
        }

        HorizontalDivider()

        Surface(tonalElevation = 2.dp) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 10.dp)
                    .imePadding(),
                verticalAlignment = Alignment.Bottom,
            ) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Message") },
                    enabled = !generating,
                    shape = RoundedCornerShape(24.dp),
                    maxLines = 5,
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                    keyboardActions = KeyboardActions(onSend = { send() }),
                )
                Spacer(Modifier.width(8.dp))
                FilledIconButton(
                    onClick = { send() },
                    enabled = !generating && input.isNotBlank(),
                    modifier = Modifier.size(52.dp),
                ) {
                    if (generating) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    } else {
                        Icon(Icons.Filled.Send, contentDescription = "Send")
                    }
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(msg: ChatMessage) {
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val maxBubbleWidth = maxWidth * 0.82f
        Row(Modifier.fillMaxWidth(), horizontalArrangement = if (msg.isUser) Arrangement.End else Arrangement.Start) {
            Box(
                modifier = Modifier
                    .widthIn(max = maxBubbleWidth)
                    .clip(
                        RoundedCornerShape(
                            topStart = 18.dp,
                            topEnd = 18.dp,
                            bottomStart = if (msg.isUser) 18.dp else 4.dp,
                            bottomEnd = if (msg.isUser) 4.dp else 18.dp,
                        )
                    )
                    .background(
                        if (msg.isUser) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.surfaceVariant
                    )
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                contentAlignment = if (msg.isUser) Alignment.CenterEnd else Alignment.CenterStart,
            ) {
                Text(
                    text = msg.text.ifEmpty { "…" },
                    color = if (msg.isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }
    }
}
