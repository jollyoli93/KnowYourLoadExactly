using SkiaSharp;

public class GetObbDims
{
    public float Xc { get; }
    public float Yc { get; }
    public virtual float Width { get; }
    public virtual float Height { get; }
    public virtual float Area => Width * Height;
    public virtual float Ratio => Width / Height;
    public float Angle { get; }
    protected float HalfW => Width / 2f;
    protected float HalfH => Height / 2f;
    protected float Cos { get; }
    protected float Sin { get; }

    public GetObbDims(SKRectI boundingBox, float angle)
    {
        Xc = boundingBox.MidX;
        Yc = boundingBox.MidY;
        Width = boundingBox.Width;
        Height = boundingBox.Height;
        Angle = angle; 
        Cos = (float)Math.Cos(angle);
        Sin = (float)Math.Sin(angle);
    }
    
    public virtual float[] GetCorners()
    {
        return
        [
            Xc + HalfW * Cos - HalfH * Sin,
            Yc + HalfW * Sin + HalfH * Cos,
            Xc + HalfW * Cos + HalfH * Sin,
            Yc + HalfW * Sin - HalfH * Cos,
            Xc - HalfW * Cos + HalfH * Sin,
            Yc - HalfW * Sin - HalfH * Cos,
            Xc - HalfW * Cos - HalfH * Sin,
            Yc - HalfW * Sin + HalfH * Cos,
        ];
    }
}

public class GetObbDimsNorm : GetObbDims
{
    private readonly int ImageHeight;
    private readonly int ImageWidth;
    
    public float WidthNorm => base.Width / ImageWidth;
    public float HeightNorm => base.Height / ImageHeight;
    public float AreaNorm => WidthNorm * HeightNorm;
    public float RatioNorm => WidthNorm / HeightNorm;

    public GetObbDimsNorm(SKRectI boundingBox, float angle, int imageWidth, int imageHeight)
        : base(boundingBox, angle)
    {
        ImageWidth = imageWidth;
        ImageHeight = imageHeight;
    }

    public override float[] GetCorners()
    {
        float[] corners = base.GetCorners();

        // Normalize all coordinates
        for (int i = 0; i < corners.Length; i++)
        {
            corners[i] = i % 2 == 0 
                ? corners[i] / ImageWidth   // x coordinates (even indices)
                : corners[i] / ImageHeight; // y coordinates (odd indices)
        }
        
        return corners;
    }
    
    public float[] GetCornersAbsolute()
    {
        return base.GetCorners();
    }
}