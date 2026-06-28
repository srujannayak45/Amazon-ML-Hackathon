import os
import zipfile

# Define the files and directories to include
files_to_include = [
    'test_out.csv',
    'README.md',
    'LICENSE',
    'Methodology_1page.pdf',
    'inference/run_inference.py',
    'inference/predict_utils.py',
    'inference/requirements.txt',
    'models/best_model/model_metadata.json',
    'src/train.py',
    'src/featurize.py',
    'src/utils.py',
    'notebooks/sample_inference.ipynb',
    'evaluation/local_eval.py',
    'manifest.txt'
]

# Create a zip file
output_path = 'submission.zip'
with zipfile.ZipFile(output_path, 'w') as zipf:
    print(f"Creating {output_path}...")
    for file_path in files_to_include:
        # Check if file exists (to handle missing optional files)
        if os.path.exists(file_path):
            print(f"Adding {file_path}...")
            zipf.write(file_path)
        else:
            print(f"Warning: {file_path} does not exist, skipping.")

print(f"Submission zip file created successfully at: {os.path.abspath(output_path)}")