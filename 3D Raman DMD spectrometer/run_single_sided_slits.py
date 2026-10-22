import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import build_slit_cartridge

build_slit_cartridge.build_slit_cartridge(150, "external_slit_cartridge_150um.step")
build_slit_cartridge.build_slit_cartridge(200, "external_slit_cartridge_200um.step")
build_slit_cartridge.build_slit_cartridge(200, "external_slit_cartridge.step")
print("All single-sided beveled slit cartridges generated successfully.")
