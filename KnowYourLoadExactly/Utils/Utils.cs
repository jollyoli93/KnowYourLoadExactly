using System;
using System.IO;

namespace KnowYourLoadExactlyCSharp.Utils;

public static class Utils
{
    public static string GetAbsolutePath(string relativePath)
    {
        FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
        // Directory can be null in some environments; fall back to the file's directory or current directory
        string assemblyFolderPath = _dataRoot.Directory?.FullName
                                    ?? Path.GetDirectoryName(_dataRoot.FullName)
                                    ?? Environment.CurrentDirectory;

        string fullPath = Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }
}

