import os
import re

# List of page files to check and fix
page_files = [
    "pages/AI_Analytics.py",
    "pages/Advanced_Formulas.py",
    "pages/Advanced_Statistics.py",
    "pages/Business_Features.py",
    "pages/Data_Enrichment.py",
]

# Session state initialization code to add if not present
session_state_code = """
# Initialize session state variables if they don't exist
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using this feature.")
    st.stop()

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()
"""

# Process each file
for file_path in page_files:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} - file does not exist")
        continue
        
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Check if the file already has session state initialization
    if "if 'data' not in st.session_state" in content:
        print(f"Skipping {file_path} - already has session state check")
        continue
    
    # Find position to insert (after imports and page configuration)
    # Look for patterns that typically come after imports and before main code
    patterns = [
        r'# Page title with styling\n',
        r'st.markdown\(.*main-header.*\)\n',
        r'st.set_page_config\([\s\S]*?\)\n',
    ]
    
    insert_pos = None
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            insert_pos = match.end()
            break
    
    if insert_pos:
        # Insert the session state code after the matched pattern
        new_content = content[:insert_pos] + session_state_code + content[insert_pos:]
        
        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.write(new_content)
        
        print(f"Updated {file_path} - added session state initialization")
    else:
        print(f"Could not determine insert position for {file_path}")
