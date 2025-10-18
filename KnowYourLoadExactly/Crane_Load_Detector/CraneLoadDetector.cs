// using System;
// using YoloDotNet.Models;
// using SkiaSharp;
// using OpenCvSharp;

// namespace KnowYourLoadExactlyCSharp.CraneLoadDetector;

// public class CraneLoadDetector
// {
//     private string _outputFolder;
//     private SKBitmap imageLoad;
//     public CraneLoadDetector(string outputFolder)
//     {
//         _outputFolder = outputFolder;
//     }

//     private void LoadImage(string imagePath)
//     {
//         using var imageLoad = SKBitmap.Decode(imagePath);
//     } 

//     public void ParseYoloObb(OBBDetection results)
//     {
//     // Get image dimensions
//         int imageWidth = imageLoad.Width;
//         int imageHeight = imageLoad.Height;

//         foreach (var result in results)
//         {
//             if (result.Label.Name == "crane_hook")
//             {
//                 GetObbDimsNorm ObbDims = new GetObbDimsNorm(
//                 result.BoundingBox,
//                 result.OrientationAngle,
//                 imageWidth,  // your original image width
//                 imageHeight  // your original image height
//                 );

//                 float[] corners = ObbDims.GetCorners();

//                 Console.WriteLine($"xyxyxyxyn: [{corners[0]},{corners[1]},{corners[2]},{corners[3]},{corners[4]},{corners[5]},{corners[6]},{corners[7]}]");
//                 System.Console.WriteLine("XY n tensor([0.5288, 0.2479, 0.5289, 0.1929, 0.5065, 0.1929, 0.5064, 0.2478])");
//             }
//             System.Console.WriteLine(result.Label);
//         }
//         imageLoad.Draw(results);         // Draw boxes and labels
//         imageLoad.Save(outputFolder);  
//     }
// }
