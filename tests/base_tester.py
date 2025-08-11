"""
Base testing framework for evaluating CellVoyager agent ablations.
Tests the percentage of code cells that successfully execute from different analyses.
"""

import os
import json
import time
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to Python path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


class AgentTester:
    """Base class for testing agent configurations and measuring code execution success rates."""
    
    def __init__(self, h5ad_path, manuscript_path, test_name="default", num_analyses=3, max_iterations=3, base_output_dir=None):
        """
        Initialize the tester.
        
        Args:
            h5ad_path (str): Path to the .h5ad file
            manuscript_path (str): Path to the .txt file containing manuscript
            test_name (str): Name for this test configuration
            num_analyses (int): Number of analyses to run
            max_iterations (int): Max iterations per analysis
            base_output_dir (str): Base directory where test results should be created (optional)
        """
        self.h5ad_path = h5ad_path
        self.manuscript_path = manuscript_path
        self.test_name = test_name
        self.num_analyses = num_analyses
        self.max_iterations = max_iterations
        
        # Create test output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        test_dir_name = f"test_results_{test_name}_{timestamp}"
        
        if base_output_dir:
            self.output_dir = os.path.join(base_output_dir, test_dir_name)
        else:
            self.output_dir = test_dir_name
        #os.makedirs(self.output_dir, exist_ok=True)
        
        # Results tracking
        self.results = {
            'test_name': test_name,
            'total_code_cells_attempted': 0,  # Total cells that were executed (including retry attempts)
            'total_failures': 0,  # Total number of code cell execution failures
            'final_successful_cells': 0,  # Cells that eventually succeeded after fixes
            'final_failed_cells': 0,  # Cells that never succeeded
            'failure_rate': 0.0,  # total_failures / total_code_cells_attempted
            'final_success_rate': 0.0,  # final_successful_cells / unique_code_cells
            'failed_fix_attempts_per_step': {},  # Per-analysis, per-step failed fix attempts data
            'analyses': [],
            'errors': []
        }
    
    def test_agent(self, use_self_critique=True, use_VLM=True, use_documentation=True):
        """
        Test a specific agent configuration using built-in ablation flags.
        
        Args:
            use_self_critique (bool): Whether to enable self-critique functionality
            use_VLM (bool): Whether to enable Vision Language Model functionality
            use_documentation (bool): Whether to enable documentation lookup functionality
        
        Returns:
            dict: Test results
        """
        print(f"\n🧪 Testing {self.test_name}...")
        print(f"   Self-critique: {'✅' if use_self_critique else '❌'}")
        print(f"   VLM: {'✅' if use_VLM else '❌'}")
        print(f"   Documentation: {'✅' if use_documentation else '❌'}")
        
        try:
            # Import the agent class here to avoid circular imports
            from agent import AnalysisAgent
            
            # Initialize agent with ablation flags
            agent = AnalysisAgent(
                h5ad_path=self.h5ad_path,
                paper_summary_path=self.manuscript_path,
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model_name="gpt-4o",
                analysis_name=f"test_{self.test_name}",
                num_analyses=self.num_analyses,
                max_iterations=self.max_iterations,
                output_home=self.output_dir,
                log_home=self.output_dir,
                prompt_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts"),
                use_self_critique=use_self_critique,
                use_VLM=use_VLM,
                use_documentation=use_documentation
            )
            
            # Run the agent
            agent.run()
            
            # Analyze results from both notebooks and logs
            self._analyze_notebooks_and_logs(agent.output_dir)
            
        except Exception as e:
            print(f"❌ Error during agent execution: {e}")
            self.results['errors'].append(str(e))
        
        return self.results
    
    def _analyze_notebooks_and_logs(self, agent_output_dir):
        """Analyze generated notebooks and logs to count total failures and final success rates."""
        print("📊 Analyzing generated notebooks and logs...")
        
        # ALWAYS analyze logs first and preserve results immediately
        print("📄 Analyzing log files...")
        log_stats = self._analyze_logs(agent_output_dir)
        
        # Save log-based statistics immediately (these are reliable)
        self.results['total_failures'] = sum(s['total_failures'] for s in log_stats.values())
        self.results['total_code_cells_attempted'] = sum(s['total_attempts'] for s in log_stats.values())
        
        # Add the detailed per-analysis, per-step failed fix attempts data
        self.results['failed_fix_attempts_per_step'] = {
            log_file: {
                analysis_num: {
                    step_num: step_data['failed_fix_attempts']
                    for step_num, step_data in steps.items()
                }
                for analysis_num, steps in analysis_data['analyses'].items()
            }
            for log_file, analysis_data in log_stats.items()
        }
        
        # Calculate log-based success estimates
        total_successful_steps = 0
        total_steps = 0
        for log_file, analysis_data in log_stats.items():
            for analysis_num, steps in analysis_data['analyses'].items():
                for step_num, step_data in steps.items():
                    total_steps += 1
                    if step_data['successful_executions'] > 0:
                        total_successful_steps += 1
        
        print(f"📊 Log analysis complete: {self.results['total_failures']} failures, {self.results['total_code_cells_attempted']} attempts")
        print(f"📊 Log-based success estimate: {total_successful_steps}/{total_steps} steps successful")
        
        # Find all notebooks in the output directory
        notebook_files = []
        for root, dirs, files in os.walk(agent_output_dir):
            for file in files:
                if file.endswith('.ipynb'):
                    notebook_files.append(os.path.join(root, file))
        
        # Try notebook analysis, but don't let it break everything
        notebook_analysis_successful = False
        try:
            if not notebook_files:
                print("⚠️ No notebooks found in output directory")
                # Use log-based estimates for final statistics
                self.results['final_successful_cells'] = total_successful_steps
                self.results['final_failed_cells'] = total_steps - total_successful_steps
            else:
                print(f"📓 Analyzing {len(notebook_files)} notebook(s)...")
                
                # Analyze notebooks for final success rates
                for notebook_path in notebook_files:
                    analysis_results = self._analyze_single_notebook(notebook_path)
                    self.results['analyses'].append(analysis_results)
                
                # Calculate final statistics from notebook analysis
                unique_code_cells = sum(a['total_code_cells'] for a in self.results['analyses'])
                final_successful = sum(a['successful_cells'] for a in self.results['analyses'])
                
                self.results['final_successful_cells'] = final_successful
                self.results['final_failed_cells'] = unique_code_cells - final_successful
                
                notebook_analysis_successful = True
                print(f"📓 Notebook analysis complete: {final_successful}/{unique_code_cells} cells successful")
                
        except Exception as e:
            print(f"⚠️ Notebook analysis failed: {e}")
            print("📊 Using log-based estimates for final statistics...")
            
            # Fall back to log-based estimates
            self.results['final_successful_cells'] = total_successful_steps
            self.results['final_failed_cells'] = total_steps - total_successful_steps
            self.results['analyses'] = []  # Clear any partial results
        
        # Calculate success rates using the final values (from either notebook or log analysis)
        final_total_cells = self.results['final_successful_cells'] + self.results['final_failed_cells']
        if final_total_cells > 0:
            self.results['final_success_rate'] = self.results['final_successful_cells'] / final_total_cells
        else:
            self.results['final_success_rate'] = 0.0
            
        if self.results['total_code_cells_attempted'] > 0:
            self.results['failure_rate'] = self.results['total_failures'] / self.results['total_code_cells_attempted']
        else:
            self.results['failure_rate'] = 0.0
        
        print(f"📈 Final Success Rate: {self.results['final_success_rate']:.2%} ({self.results['final_successful_cells']}/{final_total_cells})")
        print(f"💥 Total Failure Rate: {self.results['failure_rate']:.2%} ({self.results['total_failures']}/{self.results['total_code_cells_attempted']} attempts)")
        
        # Print detailed per-analysis, per-step failed fix attempts
        if log_stats:
            print("\n🔍 Execution Statistics per Analysis and Step:")
            for log_file, analysis_data in log_stats.items():
                print(f"  Log File: {os.path.basename(log_file)}")
                for analysis_num, steps in analysis_data['analyses'].items():
                    print(f"    Analysis {analysis_num}:")
                    for step_num, step_data in steps.items():
                        failed_attempts = step_data['failed_fix_attempts']
                        total_attempts = step_data['total_attempts']
                        successful_executions = step_data['successful_executions']
                        if failed_attempts > 0:
                            print(f"      Step {step_num}: {failed_attempts} failed fix attempts, {successful_executions} successful executions, {total_attempts} total attempts")
                        else:
                            print(f"      Step {step_num}: No failed fix attempts, {successful_executions} successful executions, {total_attempts} total attempts")
        else:
            print("\n✅ No execution data found in log analysis")
    
    def _analyze_logs(self, agent_output_dir):
        """Analyze log files to count failed fix attempts per step for each analysis."""
        all_log_stats = {}  # Will store per-log-file results
        
        # Find log files
        log_files = []
        for root, dirs, files in os.walk(agent_output_dir):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        for log_file in log_files:
            # Initialize stats for this specific log file
            log_stats = {
                'total_failures': 0, 
                'total_attempts': 0,
                'analyses': {}  # Will store per-analysis, per-step data
            }
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse the log content line by line to extract log entry names from new format
                lines = content.split('\n')
                
                for line in lines:
                    # Look for the new logging format: "RESPONSE/OUTPUT: {log_entry_name}"
                    if line.startswith("RESPONSE/OUTPUT: "):
                        log_entry_name = line.split("RESPONSE/OUTPUT: ", 1)[1].strip()
                        
                        # Parse step execution success: step_execution_success_{analysis_num}_{step_num}
                        if log_entry_name.startswith("step_execution_success_"):
                            step_name = log_entry_name.replace("step_execution_success_", "")
                            try:
                                analysis_num, step_num = step_name.split("_")
                                analysis_num, step_num = int(analysis_num), int(step_num)
                                
                                # Initialize nested dictionaries if needed
                                if analysis_num not in log_stats['analyses']:
                                    log_stats['analyses'][analysis_num] = {}
                                if step_num not in log_stats['analyses'][analysis_num]:
                                    log_stats['analyses'][analysis_num][step_num] = {
                                        'failed_fix_attempts': 0,
                                        'total_attempts': 0,
                                        'successful_executions': 0,
                                        'fix_succeeded': False,
                                        'initial_success': False
                                    }
                                
                                log_stats['analyses'][analysis_num][step_num]['initial_success'] = True
                                log_stats['analyses'][analysis_num][step_num]['total_attempts'] += 1
                                log_stats['total_attempts'] += 1
                                
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Could not parse step execution success: {log_entry_name}")
                        
                        # Parse step execution failure: step_execution_failed_{analysis_num}_{step_num}
                        elif log_entry_name.startswith("step_execution_failed_"):
                            step_name = log_entry_name.replace("step_execution_failed_", "")
                            try:
                                analysis_num, step_num = step_name.split("_")
                                analysis_num, step_num = int(analysis_num), int(step_num)
                                
                                # Initialize nested dictionaries if needed
                                if analysis_num not in log_stats['analyses']:
                                    log_stats['analyses'][analysis_num] = {}
                                if step_num not in log_stats['analyses'][analysis_num]:
                                    log_stats['analyses'][analysis_num][step_num] = {
                                        'failed_fix_attempts': 0,
                                        'total_attempts': 0,
                                        'successful_executions': 0,
                                        'fix_succeeded': False,
                                        'initial_success': False
                                    }
                                
                                log_stats['analyses'][analysis_num][step_num]['total_attempts'] += 1
                                log_stats['total_attempts'] += 1
                                log_stats['total_failures'] += 1
                                
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Could not parse step execution failure: {log_entry_name}")
                        
                        # Parse fix attempt success: fix_attempt_success_{analysis_num}_{step_num}_{fix_attempt}
                        elif log_entry_name.startswith("fix_attempt_success_"):
                            parts = log_entry_name.replace("fix_attempt_success_", "").split("_")
                            if len(parts) >= 3:
                                try:
                                    analysis_num, step_num = int(parts[0]), int(parts[1])
                                    fix_attempt = int(parts[2])
                                    
                                    # Initialize nested dictionaries if needed
                                    if analysis_num not in log_stats['analyses']:
                                        log_stats['analyses'][analysis_num] = {}
                                    if step_num not in log_stats['analyses'][analysis_num]:
                                        log_stats['analyses'][analysis_num][step_num] = {
                                            'failed_fix_attempts': 0,
                                            'total_attempts': 0,
                                            'successful_executions': 0,
                                            'fix_succeeded': False,
                                            'initial_success': False
                                        }
                                    
                                    # Mark that this step eventually succeeded through a fix attempt
                                    log_stats['analyses'][analysis_num][step_num]['fix_succeeded'] = True
                                    log_stats['analyses'][analysis_num][step_num]['total_attempts'] += 1
                                    log_stats['total_attempts'] += 1
                                    
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Could not parse fix attempt success: {log_entry_name}")
                        
                        # Parse fix attempt failure: fix_attempt_failed_{analysis_num}_{step_num}_{fix_attempt}
                        elif log_entry_name.startswith("fix_attempt_failed_"):
                            parts = log_entry_name.replace("fix_attempt_failed_", "").split("_")
                            if len(parts) >= 3:
                                try:
                                    analysis_num, step_num = int(parts[0]), int(parts[1])
                                    fix_attempt = int(parts[2])
                                    
                                    # Initialize nested dictionaries if needed
                                    if analysis_num not in log_stats['analyses']:
                                        log_stats['analyses'][analysis_num] = {}
                                    if step_num not in log_stats['analyses'][analysis_num]:
                                        log_stats['analyses'][analysis_num][step_num] = {
                                            'failed_fix_attempts': 0,
                                            'total_attempts': 0,
                                            'successful_executions': 0,
                                            'fix_succeeded': False,
                                            'initial_success': False
                                        }
                                    
                                    log_stats['analyses'][analysis_num][step_num]['failed_fix_attempts'] += 1
                                    log_stats['analyses'][analysis_num][step_num]['total_attempts'] += 1
                                    log_stats['total_attempts'] += 1
                                    log_stats['total_failures'] += 1
                                    
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Could not parse fix attempt failure: {log_entry_name}")
                        
                        # Parse fix attempt exhausted: fix_attempt_exhausted_{analysis_num}_{step_num}
                        elif log_entry_name.startswith("fix_attempt_exhausted_"):
                            step_name = log_entry_name.replace("fix_attempt_exhausted_", "")
                            try:
                                analysis_num, step_num = step_name.split("_")
                                analysis_num, step_num = int(analysis_num), int(step_num)
                                
                                # Initialize nested dictionaries if needed
                                if analysis_num not in log_stats['analyses']:
                                    log_stats['analyses'][analysis_num] = {}
                                if step_num not in log_stats['analyses'][analysis_num]:
                                    log_stats['analyses'][analysis_num][step_num] = {
                                        'failed_fix_attempts': 0,
                                        'total_attempts': 0,
                                        'successful_executions': 0,
                                        'fix_succeeded': False,
                                        'initial_success': False
                                    }
                                
                                # No need to increment counters here as the individual fix attempts are already counted
                                
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Could not parse fix attempt exhausted: {log_entry_name}")
                    
            except Exception as e:
                print(f"Warning: Could not read log file {log_file}: {e}")
            
            # Final calculation: count successful executions based on initial success OR successful fix
            for analysis_num, steps in log_stats['analyses'].items():
                for step_num, step_data in steps.items():
                    if step_data['initial_success'] or step_data['fix_succeeded']:
                        step_data['successful_executions'] = 1
                    else:
                        step_data['successful_executions'] = 0
            
            # Store this log file's results
            all_log_stats[log_file] = log_stats
        
        return all_log_stats
    
    def _analyze_single_notebook(self, notebook_path):
        """Analyze a single notebook to determine code cell success rates."""
        print(f"  📓 Analyzing {os.path.basename(notebook_path)}...")
        
        analysis_result = {
            'notebook_path': notebook_path,
            'total_code_cells': 0,
            'successful_cells': 0,
            'failed_cells': 0,
            'success_rate': 0.0,
            'cell_details': []
        }
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbf.read(f, as_version=4)
            
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    analysis_result['total_code_cells'] += 1
                    
                    # Check if cell executed successfully
                    success = self._check_cell_execution_success(cell)
                    
                    if success:
                        analysis_result['successful_cells'] += 1
                    else:
                        analysis_result['failed_cells'] += 1
                    
                    analysis_result['cell_details'].append({
                        'cell_index': i,
                        'success': success,
                        'code_preview': cell.source[:100] + '...' if len(cell.source) > 100 else cell.source
                    })
            
            if analysis_result['total_code_cells'] > 0:
                analysis_result['success_rate'] = analysis_result['successful_cells'] / analysis_result['total_code_cells']
            
        except Exception as e:
            print(f"    ❌ Error analyzing notebook: {e}")
            analysis_result['error'] = str(e)
        
        return analysis_result
    
    def _check_cell_execution_success(self, cell):
        """
        Check if a code cell executed successfully by examining its outputs.
        
        Args:
            cell: Notebook cell object
        
        Returns:
            bool: True if cell executed successfully, False otherwise
        """
        if not hasattr(cell, 'outputs') or not cell.outputs:
            # Cell has no outputs - assume it wasn't executed or had no output
            return False
        
        # Check for error outputs
        for output in cell.outputs:
            if hasattr(output, 'output_type') and output.output_type == 'error':
                return False
            elif isinstance(output, dict) and output.get('output_type') == 'error':
                return False
        
        # If we get here, the cell either had successful outputs or was empty but didn't error
        return True
    
    def save_results(self, output_path=None):
        """Save test results to JSON file."""
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"results_{self.test_name}.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        return output_path 