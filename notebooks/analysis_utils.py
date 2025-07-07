#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class StressTestAnalyzer:
    """
    Utility class for analyzing and visualizing LLM stress test results
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.data = None
        self.json_data = None
        
        # Set default plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_latest_results(self) -> pd.DataFrame:
        """Load the most recent test results from the output directory"""
        csv_files = list(self.output_dir.glob("llm_stress_test_*.csv"))
        json_files = list(self.output_dir.glob("llm_stress_test_*.json"))
        
        if not csv_files:
            raise FileNotFoundError(f"No test results found in {self.output_dir}")
        
        # Get the most recent file
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime) if json_files else None
        
        print(f"Loading results from: {latest_csv.name}")
        
        # Load CSV data
        self.data = pd.read_csv(latest_csv)
        
        # Load JSON data for detailed analysis
        if latest_json:
            with open(latest_json, 'r') as f:
                self.json_data = json.load(f)
        
        return self.data
    
    def load_results_by_timestamp(self, timestamp: str) -> pd.DataFrame:
        """Load specific test results by timestamp (e.g., '20250106_143022')"""
        csv_file = self.output_dir / f"test_{timestamp}.csv"
        json_file = self.output_dir / f"test_{timestamp}.json"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Results file not found: {csv_file}")
        
        print(f"Loading results from: {csv_file.name}")
        
        # Load CSV data
        self.data = pd.read_csv(csv_file)
        
        # Load JSON data
        if json_file.exists():
            with open(json_file, 'r') as f:
                self.json_data = json.load(f)
        
        return self.data
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for both frameworks"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_latest_results() first.")
        
        summary = {}
        
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            
            summary[framework] = {
                'avg_response_time': framework_data['avg_response_time'].mean(),
                'max_response_time': framework_data['avg_response_time'].max(),
                'min_response_time': framework_data['avg_response_time'].min(),
                'avg_rps': framework_data['rps'].mean(),
                'max_rps': framework_data['rps'].max(),
                'avg_success_rate': framework_data['success_rate'].mean(),
                'min_success_rate': framework_data['success_rate'].min(),
                'total_requests': framework_data['total_requests'].sum(),
                'total_failed_requests': framework_data['failed_requests'].sum(),
                'concurrency_levels_tested': len(framework_data),
                'peak_concurrency': framework_data['concurrency_level'].max()
            }
        
        return summary
    
    def compare_frameworks(self) -> pd.DataFrame:
        """Create a side-by-side comparison of frameworks"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_latest_results() first.")
        
        comparison_data = []
        
        for concurrency in self.data['concurrency_level'].unique():
            level_data = self.data[self.data['concurrency_level'] == concurrency]
            
            row = {'concurrency_level': concurrency}
            
            for framework in level_data['framework'].unique():
                framework_data = level_data[level_data['framework'] == framework]
                if not framework_data.empty:
                    row[f'{framework}_avg_response_time'] = framework_data['avg_response_time'].iloc[0]
                    row[f'{framework}_p95_response_time'] = framework_data['p95_response_time'].iloc[0]
                    row[f'{framework}_rps'] = framework_data['rps'].iloc[0]
                    row[f'{framework}_success_rate'] = framework_data['success_rate'].iloc[0]
                    row[f'{framework}_failed_requests'] = framework_data['failed_requests'].iloc[0]
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values('concurrency_level')
    
    def find_breaking_points(self) -> Dict:
        """Find the breaking points for each framework (when success rate drops below 90%)"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_latest_results() first.")
        
        breaking_points = {}
        
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework].sort_values('concurrency_level')
            
            breaking_point = None
            for _, row in framework_data.iterrows():
                if row['success_rate'] < 90:
                    breaking_point = row['concurrency_level']
                    break
            
            peak_rps = framework_data['rps'].max()
            optimal_concurrency = framework_data[framework_data['rps'] == peak_rps]['concurrency_level'].iloc[0]
            
            breaking_points[framework] = {
                'breaking_point': breaking_point,
                'peak_rps': peak_rps,
                'optimal_concurrency': optimal_concurrency
            }
        
        return breaking_points
    
    def get_error_analysis(self) -> Dict:
        """Analyze error patterns from JSON data"""
        if self.json_data is None:
            return {"error": "No JSON data available for error analysis"}
        
        error_analysis = {}
        
        for result in self.json_data:
            framework = result['framework']
            if framework not in error_analysis:
                error_analysis[framework] = {}
            
            error_summary = result.get('error_summary', {})
            concurrency = result['concurrency_level']
            
            for error_type, count in error_summary.items():
                if error_type not in error_analysis[framework]:
                    error_analysis[framework][error_type] = []
                error_analysis[framework][error_type].append({
                    'concurrency': concurrency,
                    'count': count
                })
        
        return error_analysis
    
    def list_available_results(self) -> List[Dict]:
        """List all available test result files"""
        csv_files = list(self.output_dir.glob("test_*.csv"))
        results = []
        
        for csv_file in sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True):
            timestamp = csv_file.stem.replace('test_', '')
            file_size = csv_file.stat().st_size
            modification_time = csv_file.stat().st_mtime
            
            results.append({
                'timestamp': timestamp,
                'filename': csv_file.name,
                'size_bytes': file_size,
                'modification_time': modification_time,
                'readable_time': pd.to_datetime(modification_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return results

class StressTestVisualizer:
    """
    Visualization utilities for stress test results
    """
    
    def __init__(self, analyzer: StressTestAnalyzer):
        self.analyzer = analyzer
        self.data = analyzer.data
        
    def plot_response_time_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot response time comparison between frameworks"""
        if self.data is None:
            raise ValueError("No data loaded in analyzer")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Response Time Analysis', fontsize=16, fontweight='bold')
        
        # Average Response Time
        axes[0, 0].set_title('Average Response Time vs Concurrency')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            axes[0, 0].plot(framework_data['concurrency_level'], framework_data['avg_response_time'], 
                          marker='o', linewidth=2, label=framework.upper())
        axes[0, 0].set_xlabel('Concurrency Level')
        axes[0, 0].set_ylabel('Average Response Time (ms)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # P95 Response Time
        axes[0, 1].set_title('P95 Response Time vs Concurrency')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            axes[0, 1].plot(framework_data['concurrency_level'], framework_data['p95_response_time'], 
                          marker='s', linewidth=2, label=framework.upper())
        axes[0, 1].set_xlabel('Concurrency Level')
        axes[0, 1].set_ylabel('P95 Response Time (ms)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Response Time Distribution
        axes[1, 0].set_title('Response Time Distribution')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            axes[1, 0].hist(framework_data['avg_response_time'], alpha=0.7, 
                          label=framework.upper(), bins=10)
        axes[1, 0].set_xlabel('Average Response Time (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Response Time Heatmap
        pivot_data = self.data.pivot(index='concurrency_level', columns='framework', values='avg_response_time')
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Response Time Heatmap')
        axes[1, 1].set_xlabel('Framework')
        axes[1, 1].set_ylabel('Concurrency Level')
        
        plt.tight_layout()
        return fig
    
    def plot_throughput_analysis(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot throughput (RPS) analysis"""
        if self.data is None:
            raise ValueError("No data loaded in analyzer")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Throughput Analysis', fontsize=16, fontweight='bold')
        
        # RPS vs Concurrency
        axes[0].set_title('Requests per Second vs Concurrency')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            axes[0].plot(framework_data['concurrency_level'], framework_data['rps'], 
                        marker='o', linewidth=2, label=framework.upper())
        axes[0].set_xlabel('Concurrency Level')
        axes[0].set_ylabel('Requests per Second')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Throughput Bar Chart
        axes[1].set_title('Average Throughput Comparison')
        framework_avg_rps = self.data.groupby('framework')['rps'].mean()
        bars = axes[1].bar(framework_avg_rps.index.str.upper(), framework_avg_rps.values)
        axes[1].set_ylabel('Average RPS')
        axes[1].set_xlabel('Framework')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_success_rate_analysis(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot success rate and error analysis"""
        if self.data is None:
            raise ValueError("No data loaded in analyzer")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Reliability Analysis', fontsize=16, fontweight='bold')
        
        # Success Rate vs Concurrency
        axes[0].set_title('Success Rate vs Concurrency')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            axes[0].plot(framework_data['concurrency_level'], framework_data['success_rate'], 
                        marker='o', linewidth=2, label=framework.upper())
        axes[0].axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
        axes[0].set_xlabel('Concurrency Level')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)
        
        # Failed Requests
        axes[1].set_title('Failed Requests by Concurrency')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            axes[1].bar(framework_data['concurrency_level'], framework_data['failed_requests'], 
                       alpha=0.7, label=framework.upper())
        axes[1].set_xlabel('Concurrency Level')
        axes[1].set_ylabel('Number of Failed Requests')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_scalability_curves(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot comprehensive scalability analysis"""
        if self.data is None:
            raise ValueError("No data loaded in analyzer")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Scalability Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Efficiency: RPS per Concurrency Unit
        axes[0, 0].set_title('Efficiency: RPS per Concurrency Unit')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            efficiency = framework_data['rps'] / framework_data['concurrency_level']
            axes[0, 0].plot(framework_data['concurrency_level'], efficiency, 
                          marker='o', linewidth=2, label=framework.upper())
        axes[0, 0].set_xlabel('Concurrency Level')
        axes[0, 0].set_ylabel('RPS per Concurrency Unit')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Response Time vs Throughput
        axes[0, 1].set_title('Response Time vs Throughput')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            scatter = axes[0, 1].scatter(framework_data['rps'], framework_data['avg_response_time'], 
                                       s=framework_data['concurrency_level']*5, alpha=0.7, 
                                       label=framework.upper())
        axes[0, 1].set_xlabel('Requests per Second')
        axes[0, 1].set_ylabel('Average Response Time (ms)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance Index (RPS / Response Time)
        axes[1, 0].set_title('Performance Index (RPS / Response Time)')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            performance_index = framework_data['rps'] / framework_data['avg_response_time']
            axes[1, 0].plot(framework_data['concurrency_level'], performance_index, 
                          marker='o', linewidth=2, label=framework.upper())
        axes[1, 0].set_xlabel('Concurrency Level')
        axes[1, 0].set_ylabel('Performance Index (RPS/ms)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Resource Utilization (Success Rate * RPS)
        axes[1, 1].set_title('Resource Utilization Score')
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework]
            utilization = (framework_data['success_rate'] / 100) * framework_data['rps']
            axes[1, 1].plot(framework_data['concurrency_level'], utilization, 
                          marker='o', linewidth=2, label=framework.upper())
        axes[1, 1].set_xlabel('Concurrency Level')
        axes[1, 1].set_ylabel('Utilization Score (Success Rate * RPS)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comparison_table(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Create a detailed comparison table"""
        if self.data is None:
            raise ValueError("No data loaded in analyzer")
        
        comparison_df = self.analyzer.compare_frameworks()
        
        # Style the dataframe for better visualization
        styled_df = comparison_df.style.format({
            col: '{:.2f}' for col in comparison_df.columns 
            if any(metric in col for metric in ['response_time', 'rps'])
        }).format({
            col: '{:.1f}%' for col in comparison_df.columns 
            if 'success_rate' in col
        }).format({
            col: '{:.0f}' for col in comparison_df.columns 
            if 'failed_requests' in col or 'concurrency_level' in col
        })
        
        if save_path:
            styled_df.to_excel(save_path, index=False)
            print(f"Comparison table saved to: {save_path}")
        
        return comparison_df
    
    def plot_error_analysis(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot error analysis if JSON data is available"""
        error_data = self.analyzer.get_error_analysis()
        
        if "error" in error_data:
            print("No error data available for visualization")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')
        
        # Error types per framework
        error_counts = {}
        for framework, errors in error_data.items():
            error_counts[framework] = {}
            for error_type, occurrences in errors.items():
                total_count = sum(occ['count'] for occ in occurrences)
                error_counts[framework][error_type] = total_count
        
        # Plot error distribution
        framework_names = list(error_counts.keys())
        if len(framework_names) >= 2:
            for i, framework in enumerate(framework_names[:2]):
                if error_counts[framework]:
                    axes[0, i].pie(error_counts[framework].values(), 
                                  labels=error_counts[framework].keys(), 
                                  autopct='%1.1f%%')
                    axes[0, i].set_title(f'{framework.upper()} Error Distribution')
        
        # Error trend over concurrency levels
        for framework, errors in error_data.items():
            for error_type, occurrences in errors.items():
                concurrency_levels = [occ['concurrency'] for occ in occurrences]
                error_counts_list = [occ['count'] for occ in occurrences]
                axes[1, 0].plot(concurrency_levels, error_counts_list, 
                              marker='o', label=f'{framework.upper()} - {error_type}')
        
        axes[1, 0].set_xlabel('Concurrency Level')
        axes[1, 0].set_ylabel('Error Count')
        axes[1, 0].set_title('Error Trends by Concurrency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total errors by framework
        total_errors = {framework: sum(sum(occ['count'] for occ in occurrences) 
                                     for occurrences in errors.values()) 
                       for framework, errors in error_data.items()}
        
        axes[1, 1].bar(total_errors.keys(), total_errors.values())
        axes[1, 1].set_title('Total Errors by Framework')
        axes[1, 1].set_ylabel('Total Error Count')
        
        plt.tight_layout()
        return fig

class StressTestReportGenerator:
    """
    Generate comprehensive reports and insights from stress test results
    """
    
    def __init__(self, analyzer: StressTestAnalyzer):
        self.analyzer = analyzer
        self.data = analyzer.data
        
    def generate_executive_summary(self) -> Dict:
        """Generate executive summary with key insights"""
        if self.data is None:
            raise ValueError("No data loaded in analyzer")
        
        summary_stats = self.analyzer.get_summary_stats()
        breaking_points = self.analyzer.find_breaking_points()
        comparison_df = self.analyzer.compare_frameworks()
        
        # Determine winner in each category
        framework_names = list(summary_stats.keys())
        if len(framework_names) < 2:
            return {"error": "Need at least 2 frameworks for comparison"}
        
        f1, f2 = framework_names[0], framework_names[1]
        
        # Performance comparison
        better_avg_response_time = f1 if summary_stats[f1]['avg_response_time'] < summary_stats[f2]['avg_response_time'] else f2
        better_max_rps = f1 if summary_stats[f1]['max_rps'] > summary_stats[f2]['max_rps'] else f2
        better_reliability = f1 if summary_stats[f1]['avg_success_rate'] > summary_stats[f2]['avg_success_rate'] else f2
        
        # Calculate performance scores (0-100)
        performance_scores = {}
        for framework in framework_names:
            # Lower response time is better (invert for scoring)
            rt_score = 100 - min(summary_stats[framework]['avg_response_time'] / 10, 100)
            rps_score = min(summary_stats[framework]['max_rps'] / 10, 100)
            reliability_score = summary_stats[framework]['avg_success_rate']
            
            # Weighted average: 30% response time, 40% throughput, 30% reliability
            overall_score = (rt_score * 0.3) + (rps_score * 0.4) + (reliability_score * 0.3)
            performance_scores[framework] = {
                'response_time_score': rt_score,
                'throughput_score': rps_score,
                'reliability_score': reliability_score,
                'overall_score': overall_score
            }
        
        # Determine overall winner
        overall_winner = max(performance_scores.keys(), key=lambda x: performance_scores[x]['overall_score'])
        
        # Generate recommendations
        recommendations = []
        
        if performance_scores[overall_winner]['overall_score'] > 70:
            recommendations.append(f"Strong recommendation: {overall_winner.upper()} shows superior performance across metrics")
        elif abs(performance_scores[f1]['overall_score'] - performance_scores[f2]['overall_score']) < 10:
            recommendations.append("Both frameworks show similar performance - choose based on specific requirements")
        
        if any(bp['breaking_point'] for bp in breaking_points.values()):
            recommendations.append("Consider implementing load balancing or auto-scaling at breaking points")
        
        return {
            'test_summary': {
                'total_concurrency_levels': len(self.data['concurrency_level'].unique()),
                'total_requests_sent': self.data['total_requests'].sum(),
                'total_failed_requests': self.data['failed_requests'].sum(),
                'frameworks_tested': framework_names
            },
            'performance_comparison': {
                'better_response_time': better_avg_response_time,
                'better_throughput': better_max_rps,
                'better_reliability': better_reliability,
                'overall_winner': overall_winner
            },
            'performance_scores': performance_scores,
            'breaking_points': breaking_points,
            'key_metrics': summary_stats,
            'recommendations': recommendations
        }
    
    def generate_detailed_insights(self) -> Dict:
        """Generate detailed technical insights"""
        if self.data is None:
            raise ValueError("No data loaded in analyzer")
        
        insights = {}
        
        # Scalability analysis
        for framework in self.data['framework'].unique():
            framework_data = self.data[self.data['framework'] == framework].sort_values('concurrency_level')
            
            # Calculate scaling efficiency
            efficiency_scores = []
            for i in range(1, len(framework_data)):
                prev_rps = framework_data.iloc[i-1]['rps']
                curr_rps = framework_data.iloc[i]['rps']
                prev_concurrency = framework_data.iloc[i-1]['concurrency_level']
                curr_concurrency = framework_data.iloc[i]['concurrency_level']
                
                if prev_rps > 0 and curr_concurrency > prev_concurrency:
                    scaling_factor = curr_concurrency / prev_concurrency
                    rps_improvement = curr_rps / prev_rps
                    efficiency = rps_improvement / scaling_factor
                    efficiency_scores.append(efficiency)
            
            avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0
            
            # Identify performance degradation points
            degradation_points = []
            for i in range(1, len(framework_data)):
                prev_rps = framework_data.iloc[i-1]['rps']
                curr_rps = framework_data.iloc[i]['rps']
                if curr_rps < prev_rps * 0.9:  # 10% degradation
                    degradation_points.append(framework_data.iloc[i]['concurrency_level'])
            
            # Response time analysis
            response_times = framework_data['avg_response_time'].values
            response_time_trend = 'increasing' if response_times[-1] > response_times[0] * 1.5 else 'stable'
            
            insights[framework] = {
                'scaling_efficiency': avg_efficiency,
                'degradation_points': degradation_points,
                'response_time_trend': response_time_trend,
                'optimal_concurrency': framework_data.loc[framework_data['rps'].idxmax(), 'concurrency_level'],
                'peak_performance': {
                    'rps': framework_data['rps'].max(),
                    'best_response_time': framework_data['avg_response_time'].min(),
                    'concurrency_at_peak': framework_data.loc[framework_data['rps'].idxmax(), 'concurrency_level']
                }
            }
        
        return insights
    
    def create_markdown_report(self, save_path: str = "stress_test_report.md") -> str:
        """Generate a comprehensive markdown report"""
        if self.data is None:
            raise ValueError("No data loaded in analyzer")
        
        exec_summary = self.generate_executive_summary()
        insights = self.generate_detailed_insights()
        
        report = f"""# LLM Framework Stress Test Report

## Executive Summary

### Test Overview
- **Frameworks Tested**: {', '.join(exec_summary['test_summary']['frameworks_tested'])}
- **Total Concurrency Levels**: {exec_summary['test_summary']['total_concurrency_levels']}
- **Total Requests Sent**: {exec_summary['test_summary']['total_requests_sent']:,}
- **Total Failed Requests**: {exec_summary['test_summary']['total_failed_requests']:,}

### Performance Comparison
- **Better Response Time**: {exec_summary['performance_comparison']['better_response_time'].upper()}
- **Better Throughput**: {exec_summary['performance_comparison']['better_throughput'].upper()}
- **Better Reliability**: {exec_summary['performance_comparison']['better_reliability'].upper()}
- **Overall Winner**: {exec_summary['performance_comparison']['overall_winner'].upper()}

### Performance Scores (0-100)
"""
        
        for framework, scores in exec_summary['performance_scores'].items():
            report += f"""
#### {framework.upper()}
- Response Time Score: {scores['response_time_score']:.1f}
- Throughput Score: {scores['throughput_score']:.1f}
- Reliability Score: {scores['reliability_score']:.1f}
- **Overall Score: {scores['overall_score']:.1f}**
"""
        
        report += f"""
## Key Metrics by Framework

"""
        
        for framework, metrics in exec_summary['key_metrics'].items():
            report += f"""
### {framework.upper()}
- Average Response Time: {metrics['avg_response_time']:.2f}ms
- Maximum RPS: {metrics['max_rps']:.2f}
- Average Success Rate: {metrics['avg_success_rate']:.1f}%
- Peak Concurrency Tested: {metrics['peak_concurrency']}
- Total Requests: {metrics['total_requests']:,}
- Total Failed Requests: {metrics['total_failed_requests']:,}
"""
        
        report += """
## Scalability Analysis

"""
        
        for framework, insight in insights.items():
            report += f"""
### {framework.upper()}
- Scaling Efficiency: {insight['scaling_efficiency']:.2f}
- Optimal Concurrency: {insight['optimal_concurrency']}
- Peak RPS: {insight['peak_performance']['rps']:.2f}
- Best Response Time: {insight['peak_performance']['best_response_time']:.2f}ms
- Response Time Trend: {insight['response_time_trend']}
"""
            
            if insight['degradation_points']:
                report += f"- Performance Degradation Points: {', '.join(map(str, insight['degradation_points']))}\n"
            else:
                report += "- No significant performance degradation detected\n"
        
        report += """
## Breaking Points Analysis

"""
        
        for framework, bp in exec_summary['breaking_points'].items():
            report += f"""
### {framework.upper()}
- Breaking Point: {bp['breaking_point'] if bp['breaking_point'] else 'Not reached'}
- Peak RPS: {bp['peak_rps']:.2f}
- Optimal Concurrency: {bp['optimal_concurrency']}
"""
        
        report += """
## Recommendations

"""
        
        for i, recommendation in enumerate(exec_summary['recommendations'], 1):
            report += f"{i}. {recommendation}\n"
        
        report += """
## Technical Notes

- Breaking point defined as success rate < 90%
- Scaling efficiency measures how well RPS scales with concurrency
- Performance scores weighted: 30% response time, 40% throughput, 30% reliability
- All tests conducted with identical prompts and parameters

---
*Report generated automatically from stress test results*
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"Markdown report saved to: {save_path}")
        return report

# Convenience functions for quick analysis
def quick_analysis(output_dir: str = "output") -> Tuple[StressTestAnalyzer, StressTestVisualizer, StressTestReportGenerator]:
    """Quickly load latest results and create analyzer, visualizer, and report generator"""
    analyzer = StressTestAnalyzer(output_dir)
    analyzer.load_latest_results()
    
    visualizer = StressTestVisualizer(analyzer)
    report_gen = StressTestReportGenerator(analyzer)
    
    return analyzer, visualizer, report_gen

def create_all_visualizations(output_dir: str = "output", save_plots: bool = True) -> Dict[str, plt.Figure]:
    """Create all standard visualizations and optionally save them"""
    analyzer, visualizer, _ = quick_analysis(output_dir)
    
    plots = {
        'response_time': visualizer.plot_response_time_comparison(),
        'throughput': visualizer.plot_throughput_analysis(),
        'success_rate': visualizer.plot_success_rate_analysis(),
        'scalability': visualizer.plot_scalability_curves(),
        'error_analysis': visualizer.plot_error_analysis()
    }
    
    if save_plots:
        import os
        plot_dir = Path(output_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        for name, fig in plots.items():
            if fig is not None:
                fig.savefig(plot_dir / f"{name}_analysis.png", dpi=300, bbox_inches='tight')
                print(f"Saved {name} plot to {plot_dir}/{name}_analysis.png")
    
    return plots