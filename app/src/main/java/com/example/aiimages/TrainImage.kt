package com.example.aiimages

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.io.IOException
import kotlin.math.min
import android.media.ExifInterface
import android.os.SystemClock
import android.widget.EditText
import android.widget.TextView
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max

class TrainImage : AppCompatActivity() {
    private var interpreter: Interpreter? = null
    private var targetWidth: Int = 224
    private var targetHeight: Int = 224

    private val trainingListData: MutableList<DataTraining> = mutableListOf()
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var editTextNumber: EditText
    private lateinit var textViewResult: TextView


    override fun onDestroy() {
        super.onDestroy()
        interpreter = null

    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main1)
        val btloadDataSet = findViewById<Button>(R.id.buttonLoaddataset)
        val btTrandDataSet = findViewById<Button>(R.id.Traindataset)
        val btSaveModel = findViewById<Button>(R.id.btSaveModel)
        val btTestModal = findViewById<Button>(R.id.btTestModal)
        editTextNumber = findViewById(R.id.editTextClass)
        textViewResult = findViewById(R.id.textViewResult)


        try {
            val options = Interpreter.Options()
            options.numThreads = 2
            val modelFile = FileUtil.loadMappedFile(this, "modell.tflite")
            interpreter = Interpreter(modelFile, options)

            Log.d(TAG, "load thanh cong ")
        } catch (e: IOException) {
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)

        }

        btTestModal.setOnClickListener {

            val bitmap = loadImageFromAssets(this, "test/fire2_mp4-41_jpg.rf.ac0feefe3b0bb6e946a6987d96b5459a.jpg")
            bitmap?.let {
                bitmapBuffer = Bitmap.createBitmap(
                    bitmap.width,
                    bitmap.height,
                    Bitmap.Config.ARGB_8888
                )
                TesstClass(bitmap, 90)
            } ?: Log.e("sdsdsds:", "Bitmap is null")

        }
        btSaveModel.setOnClickListener {
            val inputssdsd: MutableMap<String, Any> = HashMap()

            val outputsGGG: MutableMap<String, Any> = HashMap()

            val path =this.filesDir.absolutePath
            val outputFile = File(path, "checkpoint.ckpt")
            inputssdsd["checkpoint_path"] = outputFile.absolutePath

            interpreter?.runSignature(inputssdsd, outputsGGG, "save")
            textViewResult.text = "save done"


        }


        btloadDataSet.setOnClickListener {
            // Handle the button click here
            Log.e("load dataa:", "load iamge")


            try {
                val files = assets.list("rose")
                files?.forEach {
                   Log.e("Link file:" ,"$it")

                    val inputValue = editTextNumber.text.toString()


                    val bitmap = loadImageFromAssets(this, "rose/$it")
                    bitmap?.let {
                        Log.e("sdsdsds:", "${it.width} dsds ${it.height}")
                        bitmapBuffer = Bitmap.createBitmap(
                            bitmap.width,
                            bitmap.height,
                            Bitmap.Config.ARGB_8888
                        )
                        addSample(bitmap, inputValue)
                    } ?: Log.e("sdsdsds:", "Bitmap is null")

                }

                val inputValue = editTextNumber.text.toString()

                textViewResult.text = "Load data cho class:$inputValue"
            } catch (e: IOException) {
                e.printStackTrace()
            }

        }

        btTrandDataSet.setOnClickListener {
            // Handle the button click here
            trainImage()
        }


    }


    private fun addSample(image: Bitmap, className: String) {

        convertImageToTenso(image, className, 90)
    }

    private fun TesstClass(bitmap: Bitmap, rotation: Int) {

        processInputImage(bitmap, rotation)?.let { image ->


            val inputs: MutableMap<String, Any> = HashMap()
            inputs[INFERENCE_INPUT_KEY] = image.buffer

            val outputs: MutableMap<String, Any> = HashMap()
            val output = TensorBuffer.createFixedSize(
                intArrayOf(1, 4),
                DataType.FLOAT32
            )
            outputs[INFERENCE_OUTPUT_KEY] = output.buffer

            interpreter?.runSignature(inputs, outputs, INFERENCE_KEY)
            val tensorLabel = TensorLabel(classes.keys.toList(), output)
            val result = tensorLabel.categoryList
            Log.e("ket qua tesst:", result.toString())


        }

    }

    private fun trainImage() {

        var totalLoss = 0f
        var numBatchesProcessed = 0
        var avgLoss: Float

        // Shuffle training samples to reduce overfitting and
        // variance.
        trainingListData.shuffle()

        val trainBatchSize = getTrainBatchSize()

        trainingBatches(trainBatchSize)
            .forEach { trainingSamples ->
                val trainingBatchBottlenecks =
                    MutableList(trainBatchSize) {
                        FloatArray(
                            BOTTLENECK_SIZE
                        )
                    }

                val trainingBatchLabels =
                    MutableList(trainBatchSize) {
                        FloatArray(
                            classes.size
                        )
                    }

                // Copy a training sample list into two different
                // input training lists.
                trainingSamples.forEachIndexed { index, trainingSample ->
                    trainingBatchBottlenecks[index] =
                        trainingSample.bottleneck
                    trainingBatchLabels[index] =
                        trainingSample.label
                }

                val loss = training(
                    trainingBatchBottlenecks,
                    trainingBatchLabels
                )
                totalLoss += loss
                numBatchesProcessed++
            }
        Log.e("Losss", totalLoss.toString())
        // Calculate the average loss after training all batches.
        avgLoss = totalLoss / numBatchesProcessed
    }

    private fun training(
        bottlenecks: MutableList<FloatArray>,
        labels: MutableList<FloatArray>
    ): Float {
        Log.e("labels121", labels.toString())
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[TRAINING_INPUT_BOTTLENECK_KEY] = bottlenecks.toTypedArray()
        inputs[TRAINING_INPUT_LABELS_KEY] = labels.toTypedArray()
        val outputs: MutableMap<String, Any> = HashMap()
        val loss = FloatBuffer.allocate(1)
        outputs[TRAINING_OUTPUT_KEY] = loss
        interpreter?.runSignature(inputs, outputs, TRAINING_KEY)

        return loss.get(0)
    }

    private fun getTrainBatchSize(): Int {
        return min(
            max( /* at least one sample needed */1, trainingListData.size),
            EXPECTED_BATCH_SIZE
        )
    }

    private fun trainingBatches(trainBatchSize: Int): Iterator<List<DataTraining>> {
        return object : Iterator<List<DataTraining>> {
            private var nextIndex = 0

            override fun hasNext(): Boolean {
                return nextIndex < trainingListData.size
            }

            override fun next(): List<DataTraining> {
                val fromIndex = nextIndex
                val toIndex: Int = nextIndex + trainBatchSize
                nextIndex = toIndex
                return if (toIndex >= trainingListData.size) {
                    // To keep batch size consistent, last batch may include some elements from the
                    // next-to-last batch.
                    trainingListData.subList(
                        trainingListData.size - trainBatchSize,
                        trainingListData.size
                    )
                } else {
                    trainingListData.subList(fromIndex, toIndex)
                }
            }
        }
    }


    fun loadImageFromAssets(context: Context, fileName: String): Bitmap? {
        try {
            val inputStream = context.assets.open(fileName)
            val dataBitMap = BitmapFactory.decodeStream(inputStream)

            return dataBitMap
        } catch (e: IOException) {
            e.printStackTrace()
            return null
        }
    }


    private fun convertImageToTenso(image: Bitmap, className: String, rotation: Int) {
        Log.e("className:",className)
        processInputImage(image, rotation)?.let { tensorImage ->
            val bottleneck = loadBottleneck(tensorImage)
            trainingListData.add(
                DataTraining(
                    bottleneck,
                    encoding(classes.getValue(className))
                )
            )
        }

    }

    private fun loadBottleneck(image: TensorImage): FloatArray {
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[LOAD_BOTTLENECK_INPUT_KEY] = image.buffer
        val outputs: MutableMap<String, Any> = HashMap()
        val bottleneck = Array(1) { FloatArray(BOTTLENECK_SIZE) }
        outputs[LOAD_BOTTLENECK_OUTPUT_KEY] = bottleneck
        interpreter?.runSignature(inputs, outputs, LOAD_BOTTLENECK_KEY)

        return bottleneck[0]
    }

    // encode the classes name to float array
    private fun encoding(id: Int): FloatArray {
        val classEncoded = FloatArray(4) { 0f }
        classEncoded[id] = 1f
        return classEncoded
    }


    private fun processInputImage(
        image: Bitmap,
        imageRotation: Int
    ): TensorImage? {
        val height = image.height
        val width = image.width
        val cropSize = min(height, width)
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
                ResizeOp(
                    targetHeight,
                    targetWidth,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            )
            .add(NormalizeOp(0f, 255f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        return imageProcessor.process(tensorImage)
    }


    companion object {
        const val CLASS_ONE = "1"
        const val CLASS_TWO = "2"
        const val CLASS_THREE = "3"
        const val CLASS_FOUR = "4"
        private val classes = mapOf(
            CLASS_ONE to 0,
            CLASS_TWO to 1,
            CLASS_THREE to 2,
            CLASS_FOUR to 3
        )
        private const val LOAD_BOTTLENECK_INPUT_KEY = "feature"
        private const val LOAD_BOTTLENECK_OUTPUT_KEY = "bottleneck"
        private const val LOAD_BOTTLENECK_KEY = "load"

        private const val TRAINING_INPUT_BOTTLENECK_KEY = "bottleneck"
        private const val TRAINING_INPUT_LABELS_KEY = "label"
        private const val TRAINING_OUTPUT_KEY = "loss"
        private const val TRAINING_KEY = "train"

        private const val INFERENCE_INPUT_KEY = "feature"
        private const val INFERENCE_OUTPUT_KEY = "output"
        private const val INFERENCE_KEY = "infer"

        private const val BOTTLENECK_SIZE = 1 * 7 * 7 * 1280
        private const val EXPECTED_BATCH_SIZE = 20
        private const val TAG = "TrainImage"
    }

    data class DataTraining(val bottleneck: FloatArray, val label: FloatArray)


}