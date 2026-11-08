import os, tempfile, warnings
os.environ['MPLCONFIGDIR']       = os.path.join(tempfile.gettempdir(), 'mpl_cache')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings('ignore')

from app import create_app

app = create_app()

if __name__ == '__main__':
    print('ML Workbench (read-only Flask UI)')
    print('Database : instance/ml_workbench.db')
    print('Run http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
