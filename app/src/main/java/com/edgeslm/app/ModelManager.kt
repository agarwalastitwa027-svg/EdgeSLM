package com.edgeslm.app

import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import java.io.File

data class DetectedModel(
    val displayName: String,
    val sizeBytes: Long,
    /** Non-null when the model already sits at a real filesystem path llama.cpp can mmap directly. */
    val localFile: File?,
    /** Non-null when the model was found via a content:// Uri (SAF pick or Downloads scan) and still needs importing. */
    val sourceUri: Uri?,
)

/**
 * Never downloads anything itself. It only ever looks at what the user already has:
 *  1. Files the user previously imported, cached under the app's own external files dir.
 *  2. GGUF files already sitting in the public Downloads folder.
 *  3. A file the user just picked via the system file picker (Storage Access Framework).
 *
 * llama.cpp needs a real filesystem path to mmap, so anything reached via a content:// Uri
 * is copied once into the app's own storage; after that it is auto-detected on every launch.
 */
object ModelManager {

    private const val MODELS_SUBDIR = "models"

    fun modelsDir(context: Context): File =
        File(context.getExternalFilesDir(null), MODELS_SUBDIR).apply { mkdirs() }

    /** Models already imported and ready to load with no extra I/O. */
    fun localModels(context: Context): List<DetectedModel> =
        modelsDir(context).listFiles { f -> f.isFile && f.name.endsWith(".gguf", ignoreCase = true) }
            ?.sortedByDescending { it.lastModified() }
            ?.map { DetectedModel(it.name, it.length(), it, null) }
            ?: emptyList()

    /** Best-effort scan of the public Downloads collection for GGUF files not yet imported. */
    fun scanDownloadsForGguf(context: Context): List<DetectedModel> {
        val alreadyImported = localModels(context).map { it.displayName }.toSet()
        val results = mutableListOf<DetectedModel>()
        val projection = arrayOf(
            MediaStore.Downloads._ID,
            MediaStore.Downloads.DISPLAY_NAME,
            MediaStore.Downloads.SIZE,
        )
        runCatching {
            context.contentResolver.query(
                MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                projection,
                "${MediaStore.Downloads.DISPLAY_NAME} LIKE ?",
                arrayOf("%.gguf"),
                "${MediaStore.Downloads.DATE_ADDED} DESC",
            )?.use { cursor ->
                val idCol = cursor.getColumnIndexOrThrow(MediaStore.Downloads._ID)
                val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Downloads.DISPLAY_NAME)
                val sizeCol = cursor.getColumnIndexOrThrow(MediaStore.Downloads.SIZE)
                while (cursor.moveToNext()) {
                    val name = cursor.getString(nameCol) ?: continue
                    if (name in alreadyImported) continue
                    val id = cursor.getLong(idCol)
                    val uri = Uri.withAppendedPath(MediaStore.Downloads.EXTERNAL_CONTENT_URI, id.toString())
                    results += DetectedModel(name, cursor.getLong(sizeCol), null, uri)
                }
            }
        }
        return results
    }

    fun displayNameFromUri(context: Context, uri: Uri): String {
        var name = uri.lastPathSegment ?: "model.gguf"
        runCatching {
            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIdx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIdx >= 0 && cursor.moveToFirst()) {
                    name = cursor.getString(nameIdx)
                }
            }
        }
        return name
    }

    /**
     * Copies a content:// Uri into the app's own storage so llama.cpp can mmap it by path.
     * No-op (besides an existence check) if a file with the same name and size is already imported.
     */
    fun importModel(context: Context, uri: Uri, displayName: String, onProgress: (Long) -> Unit = {}): File {
        val dest = File(modelsDir(context), displayName)
        val resolver = context.contentResolver
        resolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Cannot open $uri" }
            dest.outputStream().use { output ->
                val buffer = ByteArray(1 shl 20) // 1 MiB
                var total = 0L
                var read: Int
                while (input.read(buffer).also { read = it } >= 0) {
                    output.write(buffer, 0, read)
                    total += read
                    onProgress(total)
                }
            }
        }
        return dest
    }
}
