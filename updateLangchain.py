import json
import subprocess
import sys

def update_packages():
    # Get list of outdated packages in JSON format
    outdated_cmd = ['pip', '--disable-pip-version-check', 'list', '--outdated', '--format=json']
    try:
        result = subprocess.run(outdated_cmd, capture_output=True, text=True)
        outdated_packages = json.loads(result.stdout)
        
        if not outdated_packages:
            print("No outdated packages found!")
            return
            
        print(f"Found {len(outdated_packages)} outdated packages. Starting update...")
        
        # Update each package
        for package in outdated_packages:
            package_name = package['name']
            print(f"\nUpdating {package_name}...")
            update_cmd = ['pip', 'install', '-U', package_name]
            subprocess.run(update_cmd)
            
        print("\nAll packages have been processed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

update_packages()