#!/usr/bin/env python3
"""
Example script showing how to use the analysis utilities for stress test results.
This can be easily adapted for use in a Jupyter notebook.
"""

from notebooks.analysis_utils import (
    StressTestAnalyzer, 
    StressTestVisualizer, 
    StressTestReportGenerator,
    quick_analysis,
    create_all_visualizations
)
import matplotlib.pyplot as plt

def main():
    """
    Example usage of the analysis utilities
    """
    
    print("=== LLM Stress Test Analysis Example ===\n")
    
    # Method 1: Quick analysis (loads latest results automatically)
    print("1. Quick Analysis - Loading latest results...")
    try:
        analyzer, visualizer, report_gen = quick_analysis("output")
        print(f"✓ Loaded data with {len(analyzer.data)} rows")
    except FileNotFoundError:
        print("✗ No test results found. Run stress_test.py first!")
        return
    
    # Method 2: Manual loading (if you want specific results)
    # analyzer = StressTestAnalyzer("output")
    # analyzer.load_results_by_timestamp("20250106_143022")  # Specific timestamp
    # visualizer = StressTestVisualizer(analyzer)
    # report_gen = StressTestReportGenerator(analyzer)
    
    print("\n2. Basic Data Analysis...")
    
    # Get summary statistics
    summary = analyzer.get_summary_stats()
    print(f"✓ Summary stats for {len(summary)} frameworks")
    for framework, stats in summary.items():
        print(f"  {framework.upper()}: Avg RPS={stats['max_rps']:.1f}, "
              f"Avg RT={stats['avg_response_time']:.1f}ms, "
              f"Success Rate={stats['avg_success_rate']:.1f}%")
    
    # Find breaking points
    breaking_points = analyzer.find_breaking_points()
    print(f"\n✓ Breaking points analysis:")
    for framework, bp in breaking_points.items():
        print(f"  {framework.upper()}: Breaking point at {bp['breaking_point']} "
              f"concurrent requests (Peak RPS: {bp['peak_rps']:.1f})")
    
    print("\n3. Creating Visualizations...")
    
    # Create individual plots
    fig1 = visualizer.plot_response_time_comparison()
    fig1.suptitle("Response Time Analysis", fontsize=16)
    plt.show()
    
    fig2 = visualizer.plot_throughput_analysis()
    fig2.suptitle("Throughput Analysis", fontsize=16)
    plt.show()
    
    fig3 = visualizer.plot_success_rate_analysis()
    fig3.suptitle("Success Rate Analysis", fontsize=16)
    plt.show()
    
    fig4 = visualizer.plot_scalability_curves()
    fig4.suptitle("Scalability Dashboard", fontsize=16)
    plt.show()
    
    # Create error analysis if available
    fig5 = visualizer.plot_error_analysis()
    if fig5:
        fig5.suptitle("Error Analysis", fontsize=16)
        plt.show()
    else:
        print("  No error data available for visualization")
    
    print("\n4. Creating Comparison Table...")
    
    # Create comparison table
    comparison_df = visualizer.create_comparison_table()
    print("✓ Comparison table created:")
    print(comparison_df.to_string(index=False))
    
    print("\n5. Generating Reports...")
    
    # Generate executive summary
    exec_summary = report_gen.generate_executive_summary()
    print(f"✓ Executive summary generated")
    print(f"  Overall winner: {exec_summary['performance_comparison']['overall_winner'].upper()}")
    print(f"  Recommendations: {len(exec_summary['recommendations'])} items")
    
    # Generate detailed insights
    insights = report_gen.generate_detailed_insights()
    print(f"✓ Detailed insights for {len(insights)} frameworks")
    
    # Create markdown report
    report = report_gen.create_markdown_report("output/stress_test_report.md")
    print(f"✓ Markdown report saved")
    
    print("\n6. One-Command Visualization Generation...")
    
    # Generate all plots at once and save them
    all_plots = create_all_visualizations("output", save_plots=True)
    print(f"✓ Generated and saved {len([p for p in all_plots.values() if p is not None])} plots")
    
    print("\n=== Analysis Complete! ===")
    print("\nNext steps for Jupyter notebook:")
    print("1. Import: from analysis_utils import quick_analysis")
    print("2. Load: analyzer, visualizer, report_gen = quick_analysis()")
    print("3. Visualize: fig = visualizer.plot_response_time_comparison()")
    print("4. Analyze: summary = analyzer.get_summary_stats()")
    print("5. Report: report_gen.create_markdown_report()")

if __name__ == "__main__":
    main()