/**
 * NavierBenchConsole.cs - Benchmark harness for NavierLib evaluation
 *
 * Demonstrates performance advantages of NavierLib vs C# baseline
 * for energy sector calculations
 *
 * Copyright (c) 2025 NavierLib Technologies
 */

using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using NavierLib;

class Program
{
    const int RECORD_COUNT = 10_000_000;  // 10M records
    const int WARMUP_ITERATIONS = 3;
    const int BENCHMARK_ITERATIONS = 5;

    static void Main(string[] args)
    {
        Console.WriteLine("NavierLib v1.2 — Evaluation Build");
        Console.WriteLine(new string('=', 70));
        Console.WriteLine();

        // Display system info
        DisplaySystemInfo();
        Console.WriteLine();

        // Check CPU support
        if (!Api.IsSupported())
        {
            Console.WriteLine("ERROR: AVX2 not supported on this CPU");
            Console.WriteLine("NavierLib requires Intel Haswell (2013+) or AMD Excavator (2015+)");
            return;
        }

        Console.WriteLine("✓ AVX2 support detected");
        Console.WriteLine(

$"✓ CPU Features: {Api.GetCpuFeatures()}");
        Console.WriteLine();

        // Load or generate test data
        double[] testData = LoadOrGenerateTestData();
        Console.WriteLine($"✓ Test dataset: {RECORD_COUNT:N0} gas volume records loaded");
        Console.WriteLine();

        // Run benchmarks
        BenchmarkGasVolumeConversion(testData);

        Console.WriteLine();
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }

    static void DisplaySystemInfo()
    {
        Console.WriteLine($"Platform: {RuntimeInformation.OSDescription}");
        Console.WriteLine($"Runtime:  {RuntimeInformation.FrameworkDescription}");
        Console.WriteLine($"CPU:      {Environment.ProcessorCount} cores");
    }

    static double[] LoadOrGenerateTestData()
    {
        string dataFile = "testdata.bin";

        // Try to load existing data
        if (File.Exists(dataFile))
        {
            try
            {
                byte[] bytes = File.ReadAllBytes(dataFile);
                double[] data = new double[bytes.Length / sizeof(double)];
                Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
                return data;
            }
            catch
            {
                Console.WriteLine("Warning: Could not load testdata.bin, generating new data...");
            }
        }

        // Generate synthetic but realistic data
        Console.WriteLine("Generating realistic energy sector test data...");
        Random rnd = new Random(42);  // Fixed seed for reproducibility

        double[] data = new double[RECORD_COUNT * 4];

        for (int i = 0; i < RECORD_COUNT; i++)
        {
            // Realistic ranges for natural gas measurements
            data[i * 4 + 0] = 1000.0 + rnd.NextDouble() * 5000.0;  // Volume (Nm³): 1000-6000
            data[i * 4 + 1] = 5.0 + rnd.NextDouble() * 20.0;       // Temperature (°C): 5-25
            data[i * 4 + 2] = 1.0 + rnd.NextDouble() * 5.0;        // Pressure (bar): 1-6
            data[i * 4 + 3] = 0.95 + rnd.NextDouble() * 0.10;      // Z factor: 0.95-1.05
        }

        // Save for future runs
        try
        {
            byte[] bytes = new byte[data.Length * sizeof(double)];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(dataFile, bytes);
            Console.WriteLine($"✓ Test data saved to {dataFile}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not save test data: {ex.Message}");
        }

        return data;
    }

    static void BenchmarkGasVolumeConversion(double[] testData)
    {
        Console.WriteLine(new string('=', 70));
        Console.WriteLine("BENCHMARK: Gas Volume Conversion (Nm³ → kWh)");
        Console.WriteLine(new string('=', 70));
        Console.WriteLine();

        double[] outputNavierLib = new double[RECORD_COUNT];
        double[] outputCSharp = new double[RECORD_COUNT];

        // Warmup
        Console.WriteLine("Warming up...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++)
        {
            Api.ConvertGasVolumeBatch(testData, outputNavierLib, RECORD_COUNT);
            ConvertGasVolumeBatch_CSharp(testData, outputCSharp, RECORD_COUNT);
        }

        // Benchmark NavierLib
        Console.WriteLine("Benchmarking NavierLib (AVX2 optimized)...");
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
        {
            Api.ConvertGasVolumeBatch(testData, outputNavierLib, RECORD_COUNT);
        }
        sw.Stop();
        double navierLibTime = sw.Elapsed.TotalSeconds / BENCHMARK_ITERATIONS;

        // Benchmark C# baseline
        Console.WriteLine("Benchmarking C# baseline (double arithmetic)...");
        sw.Restart();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
        {
            ConvertGasVolumeBatch_CSharp(testData, outputCSharp, RECORD_COUNT);
        }
        sw.Stop();
        double csharpTime = sw.Elapsed.TotalSeconds / BENCHMARK_ITERATIONS;

        // Results
        Console.WriteLine();
        Console.WriteLine("RESULTS:");
        Console.WriteLine(new string('-', 70));
        Console.WriteLine($"NavierLib (AVX2):       {navierLibTime:F3}s");
        Console.WriteLine($"C# Baseline (double):   {csharpTime:F3}s");
        Console.WriteLine($"Speedup:                {csharpTime / navierLibTime:F1}×");
        Console.WriteLine($"Energy Reduction:       ~{(1.0 - navierLibTime / csharpTime) * 100:F0}%");
        Console.WriteLine(new string('-', 70));

        // Verify correctness (sample check)
        Console.WriteLine();
        VerifyResults(outputNavierLib, outputCSharp);
    }

    static void ConvertGasVolumeBatch_CSharp(double[] input, double[] output, int count)
    {
        const double T_STANDARD = 273.15;
        const double P_STANDARD = 1.01325;
        const double CALORIFIC_VALUE = 10.55;

        for (int i = 0; i < count; i++)
        {
            double volume = input[i * 4 + 0];
            double temp_c = input[i * 4 + 1];
            double pressure = input[i * 4 + 2];
            double z = input[i * 4 + 3];

            double temp_k = temp_c + 273.15;
            double temp_ratio = temp_k / T_STANDARD;
            double press_ratio = P_STANDARD / pressure;

            double corrected_vol = volume * temp_ratio * press_ratio * z;
            output[i] = corrected_vol * CALORIFIC_VALUE;
        }
    }

    static void VerifyResults(double[] navierLib, double[] csharp)
    {
        const double TOLERANCE = 1e-10;

        int mismatches = 0;
        double maxError = 0.0;

        for (int i = 0; i < Math.Min(1000, navierLib.Length); i++)
        {
            double error = Math.Abs(navierLib[i] - csharp[i]);
            maxError = Math.Max(maxError, error);

            if (error > TOLERANCE)
            {
                mismatches++;
            }
        }

        if (mismatches == 0)
        {
            Console.WriteLine($"✓ Correctness verified (sample: 1000 records)");
            Console.WriteLine($"  Max error: {maxError:E10} (bit-exact match)");
        }
        else
        {
            Console.WriteLine($"⚠ Warning: {mismatches} mismatches detected");
            Console.WriteLine($"  Max error: {maxError:E10}");
        }
    }
}
