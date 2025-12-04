/**
 * LoadProfilingBenchmark.cs - Benchmark for load profile classification
 *
 * Validates NavierLib's ternary-based classification against C# baseline
 * for energy sector load profiling use case.
 *
 * Copyright (c) 2025 NavierLib Technologies
 */

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Linq;

namespace NavierLib.Benchmarks
{
    /// <summary>
    /// Load category enum matching ternary encoding
    /// </summary>
    public enum LoadCategory : byte
    {
        BelowBaseline = 0b00,  // -1: Low consumption
        Normal = 0b01,         //  0: Normal consumption
        PeakDemand = 0b10      // +1: High consumption
    }

    /// <summary>
    /// P/Invoke declarations for NavierLib load profiling
    /// </summary>
    public static class LoadProfilingNative
    {
        private const string DllName = "navierlib.dll";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nv_classify_load_profile(
            [In] double[] consumption,
            [In] double[] baseline,
            [Out] byte[] categories,
            long count,
            double lowRatio,
            double highRatio
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nv_aggregate_load_bands(
            [In] byte[] categories,
            long count,
            out long belowCount,
            out long normalCount,
            out long peakCount
        );
    }

    /// <summary>
    /// C# baseline implementation for comparison
    /// </summary>
    public static class LoadProfilingBaseline
    {
        public static void ClassifySequential(
            double[] consumption,
            double[] baseline,
            LoadCategory[] categories,
            double lowRatio,
            double highRatio)
        {
            for (int i = 0; i < consumption.Length; i++)
            {
                double ratio = consumption[i] / baseline[i];

                if (ratio < lowRatio)
                    categories[i] = LoadCategory.BelowBaseline;
                else if (ratio > highRatio)
                    categories[i] = LoadCategory.PeakDemand;
                else
                    categories[i] = LoadCategory.Normal;
            }
        }

        public static void AggregateLinq(
            LoadCategory[] categories,
            out long belowCount,
            out long normalCount,
            out long peakCount)
        {
            belowCount = categories.Count(c => c == LoadCategory.BelowBaseline);
            normalCount = categories.Count(c => c == LoadCategory.Normal);
            peakCount = categories.Count(c => c == LoadCategory.PeakDemand);
        }

        public static void AggregateLoop(
            LoadCategory[] categories,
            out long belowCount,
            out long normalCount,
            out long peakCount)
        {
            belowCount = normalCount = peakCount = 0;

            foreach (var cat in categories)
            {
                switch (cat)
                {
                    case LoadCategory.BelowBaseline: belowCount++; break;
                    case LoadCategory.Normal: normalCount++; break;
                    case LoadCategory.PeakDemand: peakCount++; break;
                }
            }
        }
    }

    /// <summary>
    /// Realistic energy consumption data generator
    /// </summary>
    public static class EnergyDataGenerator
    {
        private static readonly Random _rnd = new Random(42);  // Fixed seed

        /// <summary>
        /// Generate realistic consumption patterns
        /// </summary>
        /// <param name="count">Number of intervals</param>
        /// <param name="baselineKwh">Average baseline consumption</param>
        /// <returns>Consumption and baseline arrays</returns>
        public static (double[] consumption, double[] baseline) Generate(int count, double baselineKwh = 100.0)
        {
            double[] consumption = new double[count];
            double[] baseline = new double[count];

            for (int i = 0; i < count; i++)
            {
                // Baseline varies ±10% for different customers/time periods
                baseline[i] = baselineKwh * (0.9 + _rnd.NextDouble() * 0.2);

                // Consumption follows realistic distribution:
                // - 20% below baseline (off-peak, night)
                // - 60% normal (±20% of baseline)
                // - 20% peak demand (daytime, events)

                double pattern = _rnd.NextDouble();

                if (pattern < 0.2)
                {
                    // Below baseline: 50-80% of baseline
                    consumption[i] = baseline[i] * (0.5 + _rnd.NextDouble() * 0.3);
                }
                else if (pattern < 0.8)
                {
                    // Normal: 80-120% of baseline
                    consumption[i] = baseline[i] * (0.8 + _rnd.NextDouble() * 0.4);
                }
                else
                {
                    // Peak demand: 120-200% of baseline
                    consumption[i] = baseline[i] * (1.2 + _rnd.NextDouble() * 0.8);
                }
            }

            return (consumption, baseline);
        }
    }

    /// <summary>
    /// Main benchmark harness
    /// </summary>
    class Program
    {
        const int INTERVAL_COUNT = 1_000_000;  // 1M intervals
        const int WARMUP_ITERATIONS = 3;
        const int BENCHMARK_ITERATIONS = 5;
        const double LOW_THRESHOLD = 0.8;
        const double HIGH_THRESHOLD = 1.2;

        static void Main(string[] args)
        {
            Console.WriteLine("NavierLib Load Profiling Benchmark");
            Console.WriteLine("================================================================================");
            Console.WriteLine();
            Console.WriteLine($"Dataset: {INTERVAL_COUNT:N0} 15-minute intervals");
            Console.WriteLine($"Thresholds: Below {LOW_THRESHOLD:P0}, Peak {HIGH_THRESHOLD:P0}");
            Console.WriteLine($"Warmup: {WARMUP_ITERATIONS} iterations");
            Console.WriteLine($"Benchmark: {BENCHMARK_ITERATIONS} iterations");
            Console.WriteLine();

            // Generate realistic test data
            Console.WriteLine("Generating realistic energy consumption data...");
            var (consumption, baseline) = EnergyDataGenerator.Generate(INTERVAL_COUNT);
            Console.WriteLine("✓ Test data generated");
            Console.WriteLine();

            // Run benchmarks
            RunClassificationBenchmark(consumption, baseline);
            Console.WriteLine();
            RunAggregationBenchmark();
            Console.WriteLine();
            RunDeterminismTest(consumption, baseline);

            Console.WriteLine();
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        static void RunClassificationBenchmark(double[] consumption, double[] baseline)
        {
            Console.WriteLine("BENCHMARK 1: Classification (consumption → load bands)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            // Prepare output arrays
            LoadCategory[] categoriesBaseline = new LoadCategory[INTERVAL_COUNT];
            byte[] categoriesTernary = new byte[(INTERVAL_COUNT + 3) / 4];  // Packed 4 trits/byte

            // Warmup
            Console.WriteLine("Warming up...");
            for (int i = 0; i < WARMUP_ITERATIONS; i++)
            {
                LoadProfilingBaseline.ClassifySequential(
                    consumption, baseline, categoriesBaseline, LOW_THRESHOLD, HIGH_THRESHOLD);

                LoadProfilingNative.nv_classify_load_profile(
                    consumption, baseline, categoriesTernary, INTERVAL_COUNT, LOW_THRESHOLD, HIGH_THRESHOLD);
            }

            // Benchmark C# baseline
            Console.WriteLine("Benchmarking C# sequential classification...");
            Stopwatch sw = Stopwatch.StartNew();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
            {
                LoadProfilingBaseline.ClassifySequential(
                    consumption, baseline, categoriesBaseline, LOW_THRESHOLD, HIGH_THRESHOLD);
            }
            sw.Stop();
            double baselineTime = sw.Elapsed.TotalSeconds / BENCHMARK_ITERATIONS;

            // Benchmark NavierLib ternary
            Console.WriteLine("Benchmarking NavierLib ternary classification...");
            sw.Restart();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
            {
                LoadProfilingNative.nv_classify_load_profile(
                    consumption, baseline, categoriesTernary, INTERVAL_COUNT, LOW_THRESHOLD, HIGH_THRESHOLD);
            }
            sw.Stop();
            double ternaryTime = sw.Elapsed.TotalSeconds / BENCHMARK_ITERATIONS;

            // Results
            Console.WriteLine();
            Console.WriteLine("RESULTS:");
            Console.WriteLine($"  C# Sequential:      {baselineTime * 1000:F2} ms");
            Console.WriteLine($"  NavierLib Ternary:  {ternaryTime * 1000:F2} ms");
            Console.WriteLine($"  Speedup:            {baselineTime / ternaryTime:F1}×");
            Console.WriteLine($"  Throughput:         {INTERVAL_COUNT / ternaryTime / 1_000_000:F2} M intervals/sec");
            Console.WriteLine();

            // Verify correctness (sample check)
            VerifyClassification(categoriesBaseline, categoriesTernary);
        }

        static void RunAggregationBenchmark()
        {
            Console.WriteLine("BENCHMARK 2: Aggregation (count per load band)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            // Generate sample classifications
            LoadCategory[] categoriesEnum = new LoadCategory[INTERVAL_COUNT];
            byte[] categoriesPacked = new byte[(INTERVAL_COUNT + 3) / 4];
            Random rnd = new Random(42);

            for (int i = 0; i < INTERVAL_COUNT; i++)
            {
                double r = rnd.NextDouble();
                LoadCategory cat = r < 0.2 ? LoadCategory.BelowBaseline :
                                  r < 0.8 ? LoadCategory.Normal :
                                  LoadCategory.PeakDemand;
                categoriesEnum[i] = cat;

                // Pack into ternary format
                int byteIdx = i / 4;
                int tritIdx = i % 4;
                categoriesPacked[byteIdx] |= (byte)((byte)cat << (tritIdx * 2));
            }

            long below, normal, peak;

            // Warmup
            Console.WriteLine("Warming up...");
            for (int i = 0; i < WARMUP_ITERATIONS; i++)
            {
                LoadProfilingBaseline.AggregateLoop(categoriesEnum, out below, out normal, out peak);
                LoadProfilingNative.nv_aggregate_load_bands(categoriesPacked, INTERVAL_COUNT,
                    out below, out normal, out peak);
            }

            // Benchmark C# loop
            Console.WriteLine("Benchmarking C# loop aggregation...");
            Stopwatch sw = Stopwatch.StartNew();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
            {
                LoadProfilingBaseline.AggregateLoop(categoriesEnum, out below, out normal, out peak);
            }
            sw.Stop();
            double loopTime = sw.Elapsed.TotalSeconds / BENCHMARK_ITERATIONS;

            // Benchmark C# LINQ
            Console.WriteLine("Benchmarking C# LINQ aggregation...");
            sw.Restart();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
            {
                LoadProfilingBaseline.AggregateLinq(categoriesEnum, out below, out normal, out peak);
            }
            sw.Stop();
            double linqTime = sw.Elapsed.TotalSeconds / BENCHMARK_ITERATIONS;

            // Benchmark NavierLib ternary
            Console.WriteLine("Benchmarking NavierLib ternary aggregation...");
            sw.Restart();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
            {
                LoadProfilingNative.nv_aggregate_load_bands(categoriesPacked, INTERVAL_COUNT,
                    out below, out normal, out peak);
            }
            sw.Stop();
            double ternaryTime = sw.Elapsed.TotalSeconds / BENCHMARK_ITERATIONS;

            // Results
            Console.WriteLine();
            Console.WriteLine("RESULTS:");
            Console.WriteLine($"  C# Loop:            {loopTime * 1000:F2} ms");
            Console.WriteLine($"  C# LINQ:            {linqTime * 1000:F2} ms");
            Console.WriteLine($"  NavierLib Ternary:  {ternaryTime * 1000:F2} ms");
            Console.WriteLine($"  Speedup vs Loop:    {loopTime / ternaryTime:F1}×");
            Console.WriteLine($"  Speedup vs LINQ:    {linqTime / ternaryTime:F1}×");
            Console.WriteLine();

            // Verify counts match
            LoadProfilingBaseline.AggregateLoop(categoriesEnum, out long belowBaseline, out long normalBaseline, out long peakBaseline);
            LoadProfilingNative.nv_aggregate_load_bands(categoriesPacked, INTERVAL_COUNT,
                out long belowTernary, out long normalTernary, out long peakTernary);

            Console.WriteLine($"  Verification: Below={belowBaseline == belowTernary}, " +
                             $"Normal={normalBaseline == normalTernary}, " +
                             $"Peak={peakBaseline == peakTernary}");
        }

        static void RunDeterminismTest(double[] consumption, double[] baseline)
        {
            Console.WriteLine("TEST: Determinism (1000 runs with identical input)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            const int RUNS = 1000;
            byte[] firstResult = new byte[(INTERVAL_COUNT + 3) / 4];
            byte[] currentResult = new byte[(INTERVAL_COUNT + 3) / 4];

            // First run
            LoadProfilingNative.nv_classify_load_profile(
                consumption, baseline, firstResult, INTERVAL_COUNT, LOW_THRESHOLD, HIGH_THRESHOLD);

            // Verify next 999 runs produce identical output
            bool allMatch = true;
            for (int run = 1; run < RUNS; run++)
            {
                LoadProfilingNative.nv_classify_load_profile(
                    consumption, baseline, currentResult, INTERVAL_COUNT, LOW_THRESHOLD, HIGH_THRESHOLD);

                // Compare byte-by-byte
                for (int i = 0; i < firstResult.Length; i++)
                {
                    if (firstResult[i] != currentResult[i])
                    {
                        allMatch = false;
                        Console.WriteLine($"  ✗ Mismatch at byte {i} on run {run}");
                        break;
                    }
                }

                if (!allMatch) break;
            }

            if (allMatch)
            {
                Console.WriteLine($"  ✓ All {RUNS} runs produced identical output (bit-exact determinism)");
                Console.WriteLine("  ✓ Suitable for EU regulatory compliance and billing audits");
            }
            else
            {
                Console.WriteLine($"  ✗ Determinism check FAILED");
            }
        }

        static void VerifyClassification(LoadCategory[] baseline, byte[] ternary)
        {
            int mismatches = 0;
            const int SAMPLE_SIZE = 1000;

            for (int i = 0; i < SAMPLE_SIZE; i++)
            {
                int byteIdx = i / 4;
                int tritIdx = i % 4;

                // Extract trit from packed byte
                byte packed = ternary[byteIdx];
                byte tritValue = (byte)((packed >> (tritIdx * 2)) & 0b11);

                // Compare
                if ((byte)baseline[i] != tritValue)
                {
                    mismatches++;
                }
            }

            if (mismatches == 0)
            {
                Console.WriteLine($"  ✓ Correctness verified (sample: {SAMPLE_SIZE} intervals, 0 mismatches)");
            }
            else
            {
                Console.WriteLine($"  ✗ WARNING: {mismatches}/{SAMPLE_SIZE} mismatches detected");
            }
        }
    }
}
