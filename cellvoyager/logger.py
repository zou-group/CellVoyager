import os
import datetime
import logging
from logging.handlers import RotatingFileHandler

class Logger:
    """Minimalist logger to track prompts and analysis outputs"""
    
    def __init__(self, analysis_name, log_dir="logs"):
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create timestamp for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{analysis_name}_log_{timestamp}.log")
        
        # Set up logger
        self.logger = logging.getLogger("agent_logger")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler
        file_handler = RotatingFileHandler(
            self.log_file, 
            maxBytes=50*1024*1024,  # 50MB max file size
            backupCount=5
        )
        
        # Create formatter for clean, readable logs
        formatter = logging.Formatter(
            "\n\n" + "="*80 + "\n%(asctime)s - %(levelname)s\n" + 
            "="*80 + "\n%(message)s"
        )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Log initialization
        self.logger.info(f"Logging started. Log file: {self.log_file}")
    
    def log_prompt(self, role, prompt_text, prompt_name=""):
        """Log a prompt sent to the model"""
        header = f"PROMPT: {prompt_name}" if prompt_name else "PROMPT"
        self.logger.info(f"{header} ({role})\n\n{prompt_text}")
    
    def log_response(self, response_text, source="model"):
        """Log a response from the model or analysis output"""
        self.logger.info(f"RESPONSE/OUTPUT: {source}\n\n{response_text}")
    
    def log_code(self, code, iteration=None, analysis=None):
        """Log code generated or executed"""
        self.logger.info(f"CODE\n\n```python\n{code}\n```")
    
    def format_traceback(self, error_name, error_value, traceback):
        """Format error information for error messages"""
        return f"ERROR: {error_name}: {error_value}\n\n{traceback}"
    
    def log_error(self, error_msg, code=None):
        """Log critical errors that need investigation"""
        msg = f"ERROR\n\n{error_msg}"
        if code:
            msg += f"\n\nIn code:\n```python\n{code}\n```"
        self.logger.error(msg)