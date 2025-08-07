#!/usr/bin/env python3
"""
Plot cumulative success rate versus fix attempts across all steps in all log files.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_tester import AgentTester

def analyze_fix_attempts():
    """Analyze all log files and collect fix attempt data."""
    
    # Create a tester instance
    tester = AgentTester("dummy.h5ad", "dummy.txt", "plot_analysis")
    
    # Analyze logs from the logs directory
    logs_dir = "../logs"
    
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory not found: {logs_dir}")
        return None, None
    
    print("🔍 Analyzing log files...")
    log_stats = tester._analyze_logs(logs_dir)
    
    if not log_stats:
        print("❌ No log files found or no data parsed")
        return None, None
    
    print(f"📊 Found {len(log_stats)} log files")
    
    # Collect fix attempts data for all successful steps
    fix_attempts_needed = []  # List of fix attempts needed for each successful step
    total_steps = 0  # Total number of steps attempted
    
    for log_file_path, analysis_data in log_stats.items():
        log_file_name = os.path.basename(log_file_path)
        print(f"  📄 Processing: {log_file_name}")
        
        for analysis_num, steps in analysis_data['analyses'].items():
            for step_num, step_data in steps.items():
                total_steps += 1
                
                # Determine how many fix attempts this step needed to succeed
                if step_data['initial_success']:
                    # Succeeded on first try - 0 fix attempts needed
                    fix_attempts_needed.append(0)
                    print(f"    Analysis {analysis_num}, Step {step_num}: 0 fix attempts (initial success)")
                
                elif step_data['fix_succeeded']:
                    # Failed initially but succeeded after some fix attempts
                    # Number of fix attempts = failed_fix_attempts + 1 (for the successful fix)
                    attempts_needed = step_data['failed_fix_attempts'] + 1
                    fix_attempts_needed.append(attempts_needed)
                    print(f"    Analysis {analysis_num}, Step {step_num}: {attempts_needed} fix attempts (failed {step_data['failed_fix_attempts']}, then succeeded)")
                
                else:
                    # Step never succeeded - don't count in success rate
                    print(f"    Analysis {analysis_num}, Step {step_num}: FAILED (never succeeded)")
    
    print(f"\n📈 Summary:")
    print(f"  Total steps attempted: {total_steps}")
    print(f"  Successfully completed steps: {len(fix_attempts_needed)}")
    print(f"  Overall success rate: {len(fix_attempts_needed)/total_steps:.1%}")
    
    return fix_attempts_needed, total_steps

def plot_cumulative_success_rate(fix_attempts_needed, total_steps, max_fix_attempts=10):
    """Plot cumulative success rate versus fix attempts."""
    
    if not fix_attempts_needed:
        print("❌ No data to plot")
        return
    
    # Create cumulative success rate data
    x_values = list(range(max_fix_attempts + 1))  # 0 to max_fix_attempts
    cumulative_success_counts = []
    
    for max_attempts in x_values:
        # Count how many successful steps needed <= max_attempts fix attempts
        count = sum(1 for attempts in fix_attempts_needed if attempts <= max_attempts)
        cumulative_success_counts.append(count)
    
    # Convert to cumulative success rates (as percentage of total steps)
    cumulative_success_rates = [count / total_steps * 100 for count in cumulative_success_counts]
    
    # Print the data
    print(f"\n📊 Cumulative Success Rate Data:")
    print(f"{'Fix Attempts':<12} {'Successful Steps':<15} {'Cumulative Rate':<15}")
    print("-" * 45)
    for i, (attempts, count, rate) in enumerate(zip(x_values, cumulative_success_counts, cumulative_success_rates)):
        print(f"<= {attempts:<10} {count:<15} {rate:<14.1f}%")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative success rate
    plt.plot(x_values, cumulative_success_rates, 'b-o', linewidth=2, markersize=6, label='Cumulative Success Rate')
    
    # Add step-wise success rate (non-cumulative) as comparison
    step_success_counts = [0] * (max_fix_attempts + 1)
    for attempts in fix_attempts_needed:
        if attempts <= max_fix_attempts:
            step_success_counts[attempts] += 1
    
    step_success_rates = [count / total_steps * 100 for count in step_success_counts]
    plt.bar(x_values, step_success_rates, alpha=0.3, color='lightblue', label='Step-wise Success Rate')
    
    # Formatting
    plt.xlabel('Maximum Fix Attempts Allowed', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Cumulative Success Rate vs Fix Attempts\n(All Steps Across All Log Files)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set x-axis ticks
    plt.xticks(x_values)
    
    # Add value labels on the line
    for i, (x, y) in enumerate(zip(x_values, cumulative_success_rates)):
        if i % 2 == 0 or i == len(x_values) - 1:  # Show every other point to avoid crowding
            plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Add statistics text
    final_success_rate = cumulative_success_rates[-1]
    initial_success_rate = cumulative_success_rates[0]
    
    stats_text = f"""Statistics:
    • Total steps: {total_steps}
    • Initial success rate (0 fix attempts): {initial_success_rate:.1f}%
    • Final success rate (≤{max_fix_attempts} fix attempts): {final_success_rate:.1f}%
    • Improvement: +{final_success_rate - initial_success_rate:.1f} percentage points"""
    
    #plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "cumulative_success_rate_vs_fix_attempts.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    return cumulative_success_rates

def main():
    """Main function to run the analysis and create the plot."""
    
    print("🚀 Starting Fix Attempts Analysis")
    print("=" * 50)
    
    # Analyze fix attempts from log files
    fix_attempts_needed, total_steps = analyze_fix_attempts()
    
    if fix_attempts_needed is None:
        return 1
    
    # Plot the results
    plot_cumulative_success_rate(fix_attempts_needed, total_steps, max_fix_attempts=10)
    
    print("\n✅ Analysis complete!")
    return 0

if __name__ == "__main__":
    exit(main()) 