using System.Drawing;
using System.Drawing.Drawing2D;
using Microsoft.ML;

var assetsRelativePath = @"../../../assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
var modelFilePath = Path.Combine(assetsPath, "Model", "best.onnx");
var imagesFolder = Path.Combine(assetsPath, "images");
var outputFolder = Path.Combine(assetsPath, "images", "output");


// string imagePath = @"/Users/jackoliver/Documents/Documents/C#_projects/KnowYourLoadExactly/KnowYourLoadExactly/test_images/images/2503061230060000_jpg.rf.25f5fb4afcbba2631a14691fc0869d85.jpg";

System.Console.WriteLine(modelFilePath);

string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;

    string fullPath = Path.Combine(assemblyFolderPath, relativePath);

    return fullPath;
}