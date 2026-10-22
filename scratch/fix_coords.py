import re

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('(xf[i, j], yf[i, j], zf[i, j])', '(zf[i, j], xf[i, j], yf[i, j])')
content = content.replace('(xf[i, j+1], yf[i, j+1], zf[i, j+1])', '(zf[i, j+1], xf[i, j+1], yf[i, j+1])')
content = content.replace('(xf[i+1, j+1], yf[i+1, j+1], zf[i+1, j+1])', '(zf[i+1, j+1], xf[i+1, j+1], yf[i+1, j+1])')
content = content.replace('(xf[i+1, j], yf[i+1, j], zf[i+1, j])', '(zf[i+1, j], xf[i+1, j], yf[i+1, j])')

content = content.replace('(xb[i, j], yb[i, j], zb[i, j])', '(zb[i, j], xb[i, j], yb[i, j])')
content = content.replace('(xb[i, j+1], yb[i, j+1], zb[i, j+1])', '(zb[i, j+1], xb[i, j+1], yb[i, j+1])')
content = content.replace('(xb[i+1, j+1], yb[i+1, j+1], zb[i+1, j+1])', '(zb[i+1, j+1], xb[i+1, j+1], yb[i+1, j+1])')
content = content.replace('(xb[i+1, j], yb[i+1, j], zb[i+1, j])', '(zb[i+1, j], xb[i+1, j], yb[i+1, j])')

content = content.replace('(xf[-1, j], yf[-1, j], zf[-1, j])', '(zf[-1, j], xf[-1, j], yf[-1, j])')
content = content.replace('(xf[-1, j+1], yf[-1, j+1], zf[-1, j+1])', '(zf[-1, j+1], xf[-1, j+1], yf[-1, j+1])')
content = content.replace('(xb[-1, j+1], yb[-1, j+1], zb[-1, j+1])', '(zb[-1, j+1], xb[-1, j+1], yb[-1, j+1])')
content = content.replace('(xb[-1, j], yb[-1, j], zb[-1, j])', '(zb[-1, j], xb[-1, j], yb[-1, j])')

content = content.replace('(xf[0, j], yf[0, j], zf[0, j])', '(zf[0, j], xf[0, j], yf[0, j])')
content = content.replace('(xf[0, j+1], yf[0, j+1], zf[0, j+1])', '(zf[0, j+1], xf[0, j+1], yf[0, j+1])')
content = content.replace('(xb[0, j+1], yb[0, j+1], zb[0, j+1])', '(zb[0, j+1], xb[0, j+1], yb[0, j+1])')
content = content.replace('(xb[0, j], yb[0, j], zb[0, j])', '(zb[0, j], xb[0, j], yb[0, j])')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Replaced coordinate order to (z, x, y) successfully.')
