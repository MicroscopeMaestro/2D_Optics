
import sys, time
print('Python:', sys.version)
for pkg in ['numpy', 'scipy', 'PySide6', 'PyQt6', 'PyQt5', 'pyqtgraph', 'build123d', 'matplotlib', 'pytest']:
    try:
        m = __import__(pkg)
        ver = getattr(m, '__version__', 'installed')
        print(f'{pkg:15s}: OK ({ver})')
    except ImportError as e:
        print(f'{pkg:15s}: NOT INSTALLED ({e})')
