using System;
using Microsoft.ML.OnnxRuntime;
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Models;

namespace KnowYourLoadExactlyCSharp.Crane_Load_Detector;

public abstract class DetectModel
{
    protected string _modelPath;
    protected readonly InferenceSession inferenceSession;
    protected SKBitmap _image;

    public DetectModel(string modelPath, SKBitmap image)
    {
        _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        inferenceSession = LoadSession();
        _image = image;
    }
    protected InferenceSession LoadSession()
    {
        try
        {
            var session = new InferenceSession(_modelPath);
            Console.WriteLine("✅ Model loaded successfully!");
            return session;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Failed to load ONNX model: {ex.Message}");
            throw;
        }
    }
}

public class YoloDetectModel : DetectModel
{
    // Must be Onnx opset17 =<
    private Yolo _yolo;
    private YoloOptions _yoloOptions;

    public YoloDetectModel(string modelPath, SKBitmap image, YoloOptions yoloOptions) :
        base(modelPath, image)
    {
        _yoloOptions = yoloOptions;
        _yolo = new Yolo(_yoloOptions);
    }

    public List<OBBDetection> RunYoloObbDetectorModel()
    {
        // Display model metadata
        Console.WriteLine($"Model Type: {_yolo.ModelInfo}");

        // Run object detection
        var results = _yolo.RunObbDetection(_image, confidence: 0.25, iou: 0.7);
        return results;
    }

    public void Dispose()
    {
        _yolo?.Dispose();
    }
}
