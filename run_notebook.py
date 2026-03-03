"""Execute the notebook, skipping the last cell (interactive input)."""
import json
import subprocess
import sys

NB = 'SAE_503_Livrable_Final.ipynb'

# Read notebook
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove last code cell (input()) temporarily
last_cell = nb['cells'].pop()

# Write temp notebook without input cell
TEMP = '_temp_run.ipynb'
with open(TEMP, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False)

# Execute
result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook',
     '--execute', '--ExecutePreprocessor.timeout=600',
     '--inplace', TEMP],
    capture_output=True, text=True
)

print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)
print('Return code:', result.returncode)

if result.returncode == 0:
    # Read back executed notebook
    with open(TEMP, 'r', encoding='utf-8') as f:
        nb_exec = json.load(f)
    # Add back the input cell
    nb_exec['cells'].append(last_cell)
    # Write final notebook
    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb_exec, f, ensure_ascii=False, indent=1)
    print('Notebook executed successfully and saved.')
else:
    print('ERROR: Notebook execution failed.')

# Cleanup
import os
if os.path.exists(TEMP):
    os.remove(TEMP)
