
def save_vote_to_csv(vote_data, filepath="data/user_votes.csv"):
    """
    Appends a new vote record to the CSV file.
    vote_data: dict containing 'timestamp', 'persona', 'selected_song', 'reason_vote', 'song_vote', etc.
    """
    file_exists = os.path.isfile(filepath)
    df = pd.DataFrame([vote_data])
    
    # Append to CSV, add header only if file does not exist
    df.to_csv(filepath, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')


def load_vote_stats(filepath="data/user_votes.csv"):
    """
    Loads voting data and returns aggregated statistics.
    Returns: DataFrame of votes, or None if file doesn't exist.
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        return pd.read_csv(filepath, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error loading votes: {e}")
        return None
