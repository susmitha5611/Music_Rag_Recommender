"""
Data Loader for Music Recommendation RAG System
Loads and preprocesses real music datasets
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class MusicDataLoader:
    def __init__(self):
        self.tracks_df = None
    
    def load_final_dataset(self, file_path="final_dataset.csv"):
        """
        Load the final_dataset.csv with proper column mapping
        """
        try:
            print(f"📁 Loading dataset from {file_path}...")
            raw_df = pd.read_csv(file_path)
            print(f"✅ Raw dataset loaded: {raw_df.shape}")
            
            # Map columns from final_dataset.csv to expected format
            column_mapping = {
                'track_id': 'track_id',
                'track_name': 'track_name', 
                'track_artist': 'artist_name',
                'track_album_name': 'album_name',
                'playlist_genre': 'genre',
                'track_popularity': 'popularity',
                'track_album_release_date': 'release_date',
                'duration_ms': 'duration_ms',
                'danceability': 'danceability',
                'energy': 'energy',
                'valence': 'valence',
                'acousticness': 'acousticness',
                'instrumentalness': 'instrumentalness',
                'speechiness': 'speechiness',
                'liveness': 'liveness',
                'loudness': 'loudness',
                'tempo': 'tempo',
                'mode': 'mode',
                'key': 'key',
                'time_signature': 'time_signature',
                'playlist_subgenre': 'subgenre'
            }
            
            # Create standardized dataframe
            self.tracks_df = pd.DataFrame()
            
            for final_col, standard_col in column_mapping.items():
                if final_col in raw_df.columns:
                    self.tracks_df[standard_col] = raw_df[final_col]
            
            # Handle missing essential columns
            if 'track_id' not in self.tracks_df.columns:
                self.tracks_df['track_id'] = [f'track_{i}' for i in range(len(self.tracks_df))]
            
            # Clean and process the data
            self._clean_dataset()
            
            # Create text descriptions for RAG
            self._create_descriptions()
            
            print(f"✅ Dataset processed: {self.tracks_df.shape}")
            print(f"🎵 Unique artists: {self.tracks_df['artist_name'].nunique()}")
            print(f"🎭 Genres: {self.tracks_df['genre'].nunique()}")
            
            return self.tracks_df
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            print("📥 Generating sample dataset...")
            return self._generate_sample_dataset()
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("📥 Generating sample dataset...")
            return self._generate_sample_dataset()
    
    def _clean_dataset(self):
        """Clean and preprocess the dataset"""
        print("🧹 Cleaning dataset...")
        
        # Remove rows with missing essential data
        essential_cols = ['track_name', 'artist_name']
        for col in essential_cols:
            if col in self.tracks_df.columns:
                self.tracks_df = self.tracks_df.dropna(subset=[col])
        
        # Fill missing values for specific columns
        if 'genre' in self.tracks_df.columns:
            self.tracks_df['genre'] = self.tracks_df['genre'].fillna('unknown')
        if 'popularity' in self.tracks_df.columns:
            self.tracks_df['popularity'] = self.tracks_df['popularity'].fillna(0)
        if 'album_name' in self.tracks_df.columns:
            self.tracks_df['album_name'] = self.tracks_df['album_name'].fillna('Unknown Album')
        if 'duration_ms' in self.tracks_df.columns:
            self.tracks_df['duration_ms'] = pd.to_numeric(self.tracks_df['duration_ms'], errors='coerce').fillna(0)
        
        # Fill missing values for mode, key, time_signature with their mode (most frequent value)
        for col in ['mode', 'key', 'time_signature']:
            if col in self.tracks_df.columns:
                self.tracks_df[col] = pd.to_numeric(self.tracks_df[col], errors='coerce')
                self.tracks_df[col] = self.tracks_df[col].fillna(self.tracks_df[col].mode()[0] if not self.tracks_df[col].mode().empty else 0)

        # Ensure audio features are numeric and in valid ranges
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'speechiness', 'liveness']
        
        for feature in audio_features:
            if feature in self.tracks_df.columns:
                self.tracks_df[feature] = pd.to_numeric(self.tracks_df[feature], errors='coerce')
                self.tracks_df[feature] = self.tracks_df[feature].fillna(0.5)
                self.tracks_df[feature] = self.tracks_df[feature].clip(0, 1)
        
        # Clean tempo and loudness
        if 'tempo' in self.tracks_df.columns:
            self.tracks_df['tempo'] = pd.to_numeric(self.tracks_df['tempo'], errors='coerce')
            self.tracks_df['tempo'] = self.tracks_df['tempo'].fillna(120)
            self.tracks_df['tempo'] = self.tracks_df['tempo'].clip(60, 200)
        
        if 'loudness' in self.tracks_df.columns:
            self.tracks_df['loudness'] = pd.to_numeric(self.tracks_df['loudness'], errors='coerce')
            self.tracks_df['loudness'] = self.tracks_df['loudness'].fillna(-10)
        
        # Extract year from release date if available
        if 'release_date' in self.tracks_df.columns:
            try:
                self.tracks_df['year'] = pd.to_datetime(self.tracks_df['release_date'], errors='coerce').dt.year
                self.tracks_df['year'] = self.tracks_df['year'].fillna(2020) # Fill NaN years after conversion
            except:
                self.tracks_df['year'] = 2020  # Default year
        else:
            self.tracks_df['year'] = 2020
        
        print("✅ Dataset cleaned")
    
    def _create_descriptions(self):
        """Create rich text descriptions for RAG"""
        print("📝 Creating track descriptions...")
        
        descriptions = []
        for _, row in self.tracks_df.iterrows():
            desc_parts = []
            
            # Basic track info
            desc_parts.append(f"Track: {row['track_name']} by {row['artist_name']}")
            
            if 'album_name' in row and pd.notna(row.get('album_name')):
                desc_parts.append(f"from album {row['album_name']}")
            
            if 'year' in row and pd.notna(row.get('year')):
                desc_parts.append(f"released in {int(row['year'])}")
            
            # Genre information
            if 'genre' in row and pd.notna(row.get('genre')):
                desc_parts.append(f"Genre: {row['genre']}")
            
            if 'subgenre' in row and pd.notna(row.get('subgenre')):
                desc_parts.append(f"Subgenre: {row['subgenre']}")
            
            # Audio characteristics
            audio_desc = []
            
            if 'energy' in row and pd.notna(row.get('energy')):
                energy_level = "high energy" if row['energy'] > 0.7 else "medium energy" if row['energy'] > 0.3 else "low energy"
                audio_desc.append(energy_level)
            
            if 'danceability' in row and pd.notna(row.get('danceability')):
                dance_level = "very danceable" if row['danceability'] > 0.7 else "moderately danceable" if row['danceability'] > 0.3 else "not very danceable"
                audio_desc.append(dance_level)
            
            if 'valence' in row and pd.notna(row.get('valence')):
                mood = "happy and upbeat" if row['valence'] > 0.7 else "neutral mood" if row['valence'] > 0.3 else "sad or melancholic"
                audio_desc.append(mood)
            
            if 'acousticness' in row and pd.notna(row.get('acousticness')):
                acoustic_level = "acoustic" if row['acousticness'] > 0.7 else "semi-acoustic" if row['acousticness'] > 0.3 else "electronic"
                audio_desc.append(acoustic_level)
            
            if 'tempo' in row and pd.notna(row.get('tempo')):
                tempo_desc = "fast tempo" if row['tempo'] > 140 else "medium tempo" if row['tempo'] > 90 else "slow tempo"
                audio_desc.append(tempo_desc)
            
            if audio_desc:
                desc_parts.append(f"Musical characteristics: {', '.join(audio_desc)}")
            
            # Popularity
            if 'popularity' in row and pd.notna(row.get('popularity')):
                pop_level = "very popular" if row['popularity'] > 80 else "moderately popular" if row['popularity'] > 50 else "lesser known"
                desc_parts.append(f"Popularity: {pop_level}")
            
            descriptions.append(". ".join(desc_parts) + ".")
        
        self.tracks_df['description'] = descriptions
        print("✅ Descriptions created")
    
    def load_spotify_dataset(self, file_path=None):
        """
        Wrapper method for backward compatibility
        Now loads final_dataset.csv by default
        """
        if file_path is None:
            file_path = 'final_dataset.csv'
        
        return self.load_final_dataset(file_path)
    
    def _generate_sample_dataset(self):
        """
        Generate a realistic sample dataset for demonstration
        """
        print("🎵 Generating realistic sample dataset...")
        
        np.random.seed(42)
        n_tracks = 1000
        
        # Realistic music data
        genres = ['pop', 'rock', 'hip-hop', 'electronic', 'indie', 'jazz', 'classical', 'country', 'r&b', 'latin']
        
        # Generate track metadata
        tracks_data = {
            'track_id': [f'sample_track_{i:06d}' for i in range(n_tracks)],
            'track_name': [f'Sample Track {i}' for i in range(n_tracks)],
            'artist_name': [f'Sample Artist {i//10}' for i in range(n_tracks)],
            'album_name': [f'Sample Album {i//20}' for i in range(n_tracks)],
            'genre': np.random.choice(genres, n_tracks),
            'popularity': np.random.beta(2, 5, n_tracks) * 100,
            'year': np.random.randint(1960, 2024, n_tracks),
            'danceability': np.random.beta(2, 2, n_tracks),
            'energy': np.random.beta(2, 2, n_tracks),
            'valence': np.random.beta(2, 2, n_tracks),
            'acousticness': np.random.beta(1, 3, n_tracks),
            'instrumentalness': np.random.beta(1, 5, n_tracks),
            'speechiness': np.random.beta(1, 5, n_tracks),
            'liveness': np.random.beta(1, 4, n_tracks),
            'loudness': np.random.normal(-10, 5, n_tracks),
            'tempo': np.random.normal(120, 30, n_tracks).clip(60, 200),
            'duration_ms': np.random.normal(210000, 60000, n_tracks).clip(60000, 600000),
        }
        
        self.tracks_df = pd.DataFrame(tracks_data)
        
        # Create descriptions
        self._create_descriptions()
        
        print(f"✅ Generated {len(self.tracks_df)} sample tracks")
        return self.tracks_df
    
    def get_dataset_info(self):
        """Get dataset information"""
        if self.tracks_df is None:
            return {"error": "No dataset loaded"}
        
        info = {
            "total_tracks": len(self.tracks_df),
            "unique_artists": self.tracks_df['artist_name'].nunique(),
            "unique_albums": self.tracks_df.get('album_name', pd.Series()).nunique(),
            "genres": self.tracks_df['genre'].value_counts().to_dict(),
            "date_range": {
                "min_year": self.tracks_df['year'].min(),
                "max_year": self.tracks_df['year'].max()
            },
            "audio_features_available": [col for col in ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness'] if col in self.tracks_df.columns]
        }
        
        return info
    
    def get_sample_tracks(self, n=5):
        """Get sample tracks for testing"""
        if self.tracks_df is None or len(self.tracks_df) == 0:
            return []
        
        sample_df = self.tracks_df.sample(n=min(n, len(self.tracks_df)))
        return sample_df['track_id'].tolist()

def download_spotify_dataset():
    """
    Placeholder for dataset download functionality
    """
    print("📥 Dataset download functionality not implemented.")
    print("Please ensure final_dataset.csv is in the current directory.")
    return False

# Example usage and dataset download instructions
def download_spotify_dataset():
    """
    Instructions for downloading public Spotify datasets:
    
    1. Kaggle Spotify Dataset:
       - Visit: https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks
       - Download 'tracks.csv'
       - Place in project directory
    
    2. Alternative datasets:
       - Spotify Million Playlist Dataset: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
       - Last.fm Dataset: http://millionsongdataset.com/lastfm/
       - Free Music Archive: https://github.com/mdeff/fma
    """
    print("📥 To use real datasets:")
    print("1. Download from Kaggle: https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks")
    print("2. Save as 'spotify_tracks.csv' in project directory")
    print("3. Run: data_loader.load_spotify_dataset('spotify_tracks.csv')")

if __name__ == "__main__":
    # Example usage
    loader = MusicDataLoader()
    
    # Try to load real dataset, fallback to sample
    tracks_df = loader.load_spotify_dataset("spotify_tracks.csv")  # Will generate sample if file not found
    
    # Show dataset info
    info = loader.get_dataset_info()
    print(f"\n📊 Dataset Info:")
    print(f"Total tracks: {info['total_tracks']}")
    print(f"Unique artists: {info['unique_artists']}")
    print(f"Genres: {info['genres']}")
    
    # Show download instructions
    download_spotify_dataset()

