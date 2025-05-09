import pandas as pd
import os
from datetime import datetime

def save_dataframe_to_csv(df, filename=None, output_dir='saved_data'):
    """
    Save a pandas DataFrame to a CSV file in the specified directory
    
    Args:
        df (pandas.DataFrame): The DataFrame to save
        filename (str, optional): Name for the CSV file. If None, generates a timestamp-based name
        output_dir (str): Directory where the file will be saved
    
    Returns:
        str: Path to the saved file
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data_{timestamp}.csv'
        elif not filename.endswith('.csv'):
            filename = f'{filename}.csv'
            
        # Create full file path
        file_path = os.path.join(output_dir, filename)
        
        # Save the DataFrame to CSV
        df.to_csv(file_path, index=False)
        print(f'File successfully saved to: {file_path}')
        return file_path
        
    except Exception as e:
        print(f'Error saving file: {str(e)}')
        return None

# Example usage:
if __name__ == "__main__":
    # Example DataFrame (replace this with your actual data)
    sample_df = pd.DataFrame({
        'Column1': [1, 2, 3],
        'Column2': ['A', 'B', 'C']
    })
    
    # Save with auto-generated filename
    save_dataframe_to_csv(sample_df)
    
    # Or save with specific filename
    save_dataframe_to_csv(sample_df, filename='my_data') 