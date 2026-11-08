import os
import tempfile

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ml_pre_learning_secret_key_2026')
    TEMPLATES_AUTO_RELOAD = True
    
    # Configure Matplotlib and Loky environments safely
    os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl_cache')
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'
