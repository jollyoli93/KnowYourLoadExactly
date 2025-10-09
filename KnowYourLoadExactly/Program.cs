using OpenCvSharp4



string imagePath = @"/Users/jackoliver/Documents/Documents/C#_projects/KnowYourLoadExactly/KnowYourLoadExactly/test_images/images/2503061230060000_jpg.rf.25f5fb4afcbba2631a14691fc0869d85.jpg";
Mat src = new Mat(imagePath, ImreadModes.Grayscale);

Mat dst = new Mat();

Cv2.Canny(src, dst, 50, 200);
using (new OpenCvSharp.Window("src image", src))
using (new OpenCvSharp.Window("dst image", dst))
{
Cv2.WaitKey();
}
