# -*- coding: utf-8 -*-
import os, re

p_lit = r'c:/Users/Juan/Documents/GitHub/2D_Optics/.agents/survey_literature_optics/research_notes.md'
p_phys = r'c:/Users/Juan/Documents/GitHub/2D_Optics/.agents/survey_thorlabs_physics/analysis.md'
p_out = r'c:/Users/Juan/Documents/GitHub/2D_Optics/research_notes.md'

with open(p_lit, 'r', encoding='utf-8') as f:
    lit_text = f.read()

with open(p_phys, 'r', encoding='utf-8') as f:
    phys_text = f.read()

# Let's verify we can parse both into section dictionaries
def parse_sections(text):
    lines = text.split('\n')
    sections = {}
    current_title = "HEADER"
    current_lines = []
    for line in lines:
        if line.startswith('## '):
            sections[current_title] = '\n'.join(current_lines)
            current_title = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    sections[current_title] = '\n'.join(current_lines)
    return sections

lit_sec = parse_sections(lit_text)
phys_sec = parse_sections(phys_text)

print("Lit section keys:", list(lit_sec.keys()))
print("Phys section keys:", list(phys_sec.keys()))
