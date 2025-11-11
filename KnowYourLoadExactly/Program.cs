using SkiaSharp;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;
using YoloDotNet.Core;
using KnowYourLoadExactlyCSharp.Utils;
using KnowYourLoadExactlyCSharp.Crane_Load_Detector;

var currDir = Directory.GetCurrentDirectory();
var assetsRelativePath = @"../../../assets";
string assetsPath = Utils.GetAbsolutePath(assetsRelativePath);
var modelFilePath = Path.Combine(assetsPath, "Models", "detection(opset17).onnx");

var imageFolder = Path.Combine(currDir, "../../../images");
var imageFilePath = Path.Combine(imageFolder, "2507190940060000_jpg.rf.25413f580b6dab2700fab296e0eca813.jpg");
var outputFolder = Path.Combine(imageFolder, "outputs", "result.jpg");

System.Console.WriteLine($"releative path {assetsRelativePath}");

System.Console.WriteLine($"asset path {assetsPath}");

System.Console.WriteLine($"model path {modelFilePath}");


var yoloOptions = new YoloOptions
{
    OnnxModel = modelFilePath,
    // Path to your trained model.
    // Ensure this model matches the preprocessing and training settings you use below.

    // OnnxModelBytes = modelBytes
    // Load model in byte[] format (e.g. for embedded scenarios)

    ExecutionProvider = new CpuExecutionProvider(),
    // Sets the execution backend.
    // Available options:
    //   - CpuExecutionProvider         → CPU-only (no GPU required)
    //   - CudaExecutionProvider        → GPU via CUDA (NVIDIA required)
    //   - TensorRtExecutionProvider    → GPU via NVIDIA TensorRT for maximum performance

    ImageResize = ImageResize.Proportional,
    // IMPORTANT: Match this to your model's training preprocessing.
    // Proportional = the dataset images were not distorted; their aspect ratio was preserved.
    // Stretched = the dataset images were resized directly to the model's input size, ignoring aspect ratio.

    SamplingOptions = new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default
                                                                                     // IMPORTANT: This defines how pixel data is resampled when resizing the image.
                                                                                     // The choice of sampling method can directly affect detection accuracy, 
                                                                                     // as different resampling methods (Nearest, Bilinear, Cubic, etc.) slightly alter object shapes and edges.
                                                                                     // Check the benchmarks for examples and guidance: 
                                                                                     // https://github.com/NickSwardh/YoloDotNet/tree/master/test/YoloDotNet.Benchmarks
};

using var imageLoad = SKBitmap.Decode(imageFilePath);

YoloDetectModel detectModel = new (modelFilePath, imageLoad, yoloOptions);

var results = detectModel.RunYoloObbDetectorModel();

// Get image dimensions
int imageWidth = imageLoad.Width;
int imageHeight = imageLoad.Height;

foreach (var result in results)
{
    if (result.Label.Name == "crane_hook")
    {
        GetObbDimsNorm ObbDims = new GetObbDimsNorm(
        result.BoundingBox,
        result.OrientationAngle,
        imageWidth,
        imageHeight 
        );

        float[] corners = ObbDims.GetCorners();

        Console.WriteLine($"xyxyxyxyn: [{corners[0]},{corners[1]},{corners[2]},{corners[3]},{corners[4]},{corners[5]},{corners[6]},{corners[7]}]");
        System.Console.WriteLine("XY n tensor([0.5288, 0.2479, 0.5289, 0.1929, 0.5065, 0.1929, 0.5064, 0.2478])");
    }
    System.Console.WriteLine(result.Label);
}
imageLoad.Draw(results);         // Draw boxes and labels
imageLoad.Save(outputFolder);

detectModel.Dispose();