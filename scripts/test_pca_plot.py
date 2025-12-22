import sys
import os
import pandas as pd
import plotly.io as pio

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

def main():
    print("--- Starting PCA Plot Test ---")
    
    # 1. Load Data
    try:
        df_songs = pd.read_csv("data/songs.csv")
        df_pca = pd.read_csv("data/df_pca.csv")
        print(f"Loaded songs.csv: {df_songs.shape}")
        print(f"Loaded df_pca.csv: {df_pca.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Mock Context
    # Pick the first song as context
    context_song = df_songs.iloc[0]
    print(f"Context Song: {context_song['track_name']} ({context_song['track_id']})")

    # Mock Recommendations (Top 5)
    recs = df_songs.iloc[1:6]
    
    # Mock Candidates
    step1 = df_songs.iloc[10:15].to_dict('records')
    step2 = df_songs.iloc[20:25].to_dict('records')
    
    # 3. Generate Plot
    print("\nGenerating Plot...")
    try:
        fig = utils.plot_pca_visualization(
            df_songs,
            context_song,
            recs,
            user_history=[],
            step1_cands=step1,
            step2_cands=step2,
            df_pca=df_pca
        )
        
        print("\n--- Figure Analysis ---")
        # Check Traces
        print(f"Number of Traces: {len(fig.data)}")
        
        count = 0
        for trace in fig.data:
            print(f"Trace Type: {trace.type}")
            print(f"Trace Name: {trace.name if hasattr(trace, 'name') else 'N/A'}")
            print(f"X points: {len(trace.x)}")
            print(f"Y points: {len(trace.y)}")
            print(f"First 3 X: {trace.x[:3]}")
            print(f"First 3 Y: {trace.y[:3]}")
            count += len(trace.x)
            
        print(f"Total Points Plotted: {count}")
        
        # Save to HTML
        output_file = "test_plot.html"
        pio.write_html(fig, output_file)
        print(f"\nSuccess! Plot saved to {output_file}")
        
    except Exception as e:
        import traceback
        print(f"Plotting Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
