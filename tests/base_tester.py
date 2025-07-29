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
    
    def __init__(self, h5ad_path, manuscript_path, test_name="default", num_analyses=3, max_iterations=3):
        """
        Initialize the tester.
        
        Args:
            h5ad_path (str): Path to the .h5ad file
            manuscript_path (str): Path to the .txt file containing manuscript
            test_name (str): Name for this test configuration
            num_analyses (int): Number of analyses to run
            max_iterations (int): Max iterations per analysis
        """
        self.h5ad_path = h5ad_path
        self.manuscript_path = manuscript_path
        self.test_name = test_name
        self.num_analyses = num_analyses
        self.max_iterations = max_iterations
        
        # Create test output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"test_results_{test_name}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results tracking
        self.results = {
            'test_name': test_name,
            'total_code_cells_attempted': 0,  # Total cells that were executed (including retry attempts)
            'total_failures': 0,  # Total number of code cell execution failures
            'final_successful_cells': 0,  # Cells that eventually succeeded after fixes
            'final_failed_cells': 0,  # Cells that never succeeded
            'failure_rate': 0.0,  # total_failures / total_code_cells_attempted
            'final_success_rate': 0.0,  # final_successful_cells / unique_code_cells
            'analyses': [],
            'errors': []
        }
    
    def test_agent(self, use_self_critique=True, use_VLM=True):
        """
        Test a specific agent configuration using built-in ablation flags.
        
        Args:
            use_self_critique (bool): Whether to enable self-critique functionality
            use_VLM (bool): Whether to enable Vision Language Model functionality
        
        Returns:
            dict: Test results
        """
        print(f"\n🧪 Testing {self.test_name}...")
        print(f"   Self-critique: {'✅' if use_self_critique else '❌'}")
        print(f"   VLM: {'✅' if use_VLM else '❌'}")
        
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
                use_self_critique=use_self_critique,
                use_VLM=use_VLM
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
        
        # Find all notebooks in the output directory
        notebook_files = []
        for root, dirs, files in os.walk(agent_output_dir):
            for file in files:
                if file.endswith('.ipynb'):
                    notebook_files.append(os.path.join(root, file))
        
        if not notebook_files:
            print("⚠️ No notebooks found in output directory")
            return
        
        # Analyze logs to count total failures (including retry attempts)
        log_stats = self._analyze_logs(agent_output_dir)
        
        # Analyze notebooks for final success rates
        for notebook_path in notebook_files:
            analysis_results = self._analyze_single_notebook(notebook_path)
            self.results['analyses'].append(analysis_results)
        
        # Calculate final statistics
        unique_code_cells = sum(a['total_code_cells'] for a in self.results['analyses'])
        final_successful = sum(a['successful_cells'] for a in self.results['analyses'])
        
        self.results['final_successful_cells'] = final_successful
        self.results['final_failed_cells'] = unique_code_cells - final_successful
        self.results['total_failures'] = log_stats['total_failures']
        self.results['total_code_cells_attempted'] = log_stats['total_attempts']
        
        if unique_code_cells > 0:
            self.results['final_success_rate'] = final_successful / unique_code_cells
        if self.results['total_code_cells_attempted'] > 0:
            self.results['failure_rate'] = self.results['total_failures'] / self.results['total_code_cells_attempted']
        
        print(f"📈 Final Success Rate: {self.results['final_success_rate']:.2%} ({final_successful}/{unique_code_cells})")
        print(f"💥 Total Failure Rate: {self.results['failure_rate']:.2%} ({self.results['total_failures']}/{self.results['total_code_cells_attempted']} attempts)")
    
    def _analyze_logs(self, agent_output_dir):
        """Analyze log files to count total code execution failures and attempts."""
        log_stats = {'total_failures': 0, 'total_attempts': 0}
        
        # Find log files
        log_files = []
        for root, dirs, files in os.walk(agent_output_dir):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                total_fix_attempts = content.count("FIX ATTEMPT") - content.count("FIX ATTEMPT FAILED")
                successful_fix_attempts = content.count("FIX SUCCESSFUL")

                log_stats['total_failures'] = total_fix_attempts - successful_fix_attempts
                log_stats['total_attempts'] = content.count("STEP") + total_fix_attempts
                    
            except Exception as e:
                print(f"Warning: Could not read log file {log_file}: {e}")
        
        return log_stats
    
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