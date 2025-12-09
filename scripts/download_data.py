import kagglehub
import pandas as pd
import os

def main():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    print("Path to dataset files:", path)

    # Find the CSV file
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                break
    
    if not csv_file:
        print("No CSV file found in the dataset.")
        return

    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Original shape: {df.shape}")
    
    # Sample data to keep it lightweight for the demo
    # We want a good mix, so maybe just random sample for now
    # Or we could ensure we have enough diversity if we were being fancy, but random is fine for a demo.
    df_sample = df.sample(n=2000, random_state=42)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    output_path = "data/songs.csv"
    df_sample.to_csv(output_path, index=False)
    print(f"Saved sampled data to {output_path} with shape {df_sample.shape}")

if __name__ == "__main__":
    main()
