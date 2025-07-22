"""
User Simulator for Music Recommendation System
Generates realistic user listening histories and preferences
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple

class UserSimulator:
    def __init__(self, tracks_df: pd.DataFrame, seed: int = 42):
        """
        Initialize user simulator
        
        Args:
            tracks_df: DataFrame containing track information
            seed: Random seed for reproducibility
        """
        self.tracks_df = tracks_df
        np.random.seed(seed)
        random.seed(seed)
        self.genre_weights = self._calculate_genre_weights()
        
    def _calculate_genre_weights(self) -> Dict[str, float]:
        """Calculate genre popularity weights based on dataset"""
        genre_counts = self.tracks_df['genre'].value_counts()
        total_tracks = len(self.tracks_df)
        
        # Normalize to probabilities
        weights = {}
        for genre, count in genre_counts.items():
            weights[genre] = count / total_tracks
        
        return weights
    
    def generate_user_histories(self, n_users: int = 100) -> List[Dict]:
        """
        Generate realistic user listening histories with preferences
        
        Args:
            n_users: Number of users to generate
            
        Returns:
            List of user dictionaries with listening histories and preferences
        """
        users = []
        
        # Define user archetypes with different listening patterns
        user_archetypes = [
            {
                'name': 'mainstream_listener',
                'genre_diversity': 0.3,  # Low diversity, focused on popular genres
                'popularity_preference': 0.8,  # Prefers popular tracks
                'history_size_range': (30, 100)
            },
            {
                'name': 'music_explorer',
                'genre_diversity': 0.8,  # High diversity, explores many genres
                'popularity_preference': 0.4,  # Mixed popularity preferences
                'history_size_range': (50, 150)
            },
            {
                'name': 'niche_enthusiast',
                'genre_diversity': 0.4,  # Medium diversity, focused on specific niches
                'popularity_preference': 0.2,  # Prefers less popular, underground music
                'history_size_range': (40, 120)
            },
            {
                'name': 'casual_listener',
                'genre_diversity': 0.5,  # Medium diversity
                'popularity_preference': 0.6,  # Somewhat popular music
                'history_size_range': (15, 60)
            }
        ]
        
        for user_id in range(n_users):
            # Choose random archetype
            archetype = random.choice(user_archetypes)
            
            # Generate user preferences based on archetype
            user = self._generate_single_user(user_id, archetype)
            users.append(user)
        
        print(f"✅ Generated {len(users)} users with diverse listening patterns")
        return users
    
    def _generate_single_user(self, user_id: int, archetype: Dict) -> Dict:
        """Generate a single user based on archetype"""
        
        # Determine preferred genres based on diversity
        available_genres = list(self.genre_weights.keys())
        n_preferred_genres = max(1, int(len(available_genres) * archetype['genre_diversity']))
        
        # Weight genre selection by popularity and archetype preferences
        if archetype['popularity_preference'] > 0.6:
            # Mainstream listener - prefer popular genres
            genre_probs = [self.genre_weights[g] for g in available_genres]
        else:
            # Niche listener - uniform selection or inverse popularity
            genre_probs = [1/len(available_genres) for _ in available_genres]
        
        # Normalize probabilities
        total_prob = sum(genre_probs)
        genre_probs = [p/total_prob for p in genre_probs]
        
        preferred_genres = np.random.choice(
            available_genres, 
            size=n_preferred_genres, 
            replace=False,
            p=genre_probs
        )
        
        # Generate listening history
        history_size = random.randint(*archetype['history_size_range'])
        listening_history = self._generate_listening_history(
            preferred_genres, 
            history_size, 
            archetype['popularity_preference']
        )
        
        # Generate audio feature preferences based on listening history
        audio_preferences = self._extract_audio_preferences(listening_history)
        
        # Generate textual preferences
        text_preferences = self._generate_text_preferences(preferred_genres, audio_preferences)
        
        return {
            'user_id': f'user_{user_id:03d}',
            'archetype': archetype['name'],
            'preferred_genres': list(preferred_genres),
            'listening_history': listening_history,
            'audio_preferences': audio_preferences,
            'text_preferences': text_preferences,
            'history_size': len(listening_history)
        }
    
    def _generate_listening_history(self, preferred_genres: List[str], 
                                  history_size: int, 
                                  popularity_preference: float) -> List[str]:
        """Generate listening history for a user"""
        
        # Filter tracks by preferred genres (80% of listening)
        preferred_tracks = self.tracks_df[self.tracks_df['genre'].isin(preferred_genres)]
        other_tracks = self.tracks_df[~self.tracks_df['genre'].isin(preferred_genres)]
        
        # Calculate how many tracks from each category
        n_preferred = int(history_size * 0.8)
        n_other = history_size - n_preferred
        
        history = []
        
        # Add preferred genre tracks with popularity weighting
        if len(preferred_tracks) > 0 and n_preferred > 0:
            if 'popularity' in preferred_tracks.columns:
                # Weight by popularity based on user preference
                popularity_scores = preferred_tracks['popularity'].values
                if popularity_preference > 0.5:
                    # Prefer popular tracks
                    weights = (popularity_scores / 100) ** 2
                else:
                    # Prefer less popular tracks
                    weights = ((100 - popularity_scores) / 100) ** 2
                
                # Normalize weights
                weights = weights / weights.sum()
                
                selected_indices = np.random.choice(
                    len(preferred_tracks),
                    size=min(n_preferred, len(preferred_tracks)),
                    replace=False,
                    p=weights
                )
                
                selected_tracks = preferred_tracks.iloc[selected_indices]
            else:
                # Random selection if no popularity data
                selected_tracks = preferred_tracks.sample(
                    n=min(n_preferred, len(preferred_tracks)), 
                    replace=False
                )
            
            history.extend(selected_tracks['track_id'].tolist())
        
        # Add some diversity from other genres
        if len(other_tracks) > 0 and n_other > 0:
            diverse_tracks = other_tracks.sample(
                n=min(n_other, len(other_tracks)), 
                replace=False
            )
            history.extend(diverse_tracks['track_id'].tolist())
        
        return history
    
    def _extract_audio_preferences(self, listening_history: List[str]) -> Dict[str, float]:
        """Extract audio feature preferences from listening history"""
        
        if not listening_history:
            return {}
        
        history_tracks = self.tracks_df[self.tracks_df['track_id'].isin(listening_history)]
        
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'speechiness', 'liveness']
        
        preferences = {}
        for feature in audio_features:
            if feature in history_tracks.columns:
                preferences[feature] = float(history_tracks[feature].mean())
        
        return preferences
    
    def _generate_text_preferences(self, preferred_genres: List[str], 
                                 audio_preferences: Dict[str, float]) -> List[str]:
        """Generate textual preference descriptions"""
        
        preferences = []
        
        # Genre-based preferences
        if 'electronic' in preferred_genres:
            preferences.append("electronic beats and synthesized sounds")
        if 'rock' in preferred_genres:
            preferences.append("guitar-driven rock music")
        if 'jazz' in preferred_genres:
            preferences.append("sophisticated jazz compositions")
        if 'classical' in preferred_genres:
            preferences.append("orchestral and classical music")
        if 'hip-hop' in preferred_genres:
            preferences.append("hip-hop beats and rap vocals")
        if 'pop' in preferred_genres:
            preferences.append("catchy pop melodies")
        
        # Audio feature-based preferences
        if audio_preferences.get('energy', 0.5) > 0.7:
            preferences.append("high energy and intense music")
        elif audio_preferences.get('energy', 0.5) < 0.3:
            preferences.append("calm and relaxing music")
        
        if audio_preferences.get('danceability', 0.5) > 0.7:
            preferences.append("danceable and rhythmic tracks")
        
        if audio_preferences.get('valence', 0.5) > 0.7:
            preferences.append("upbeat and positive music")
        elif audio_preferences.get('valence', 0.5) < 0.3:
            preferences.append("melancholic and emotional songs")
        
        if audio_preferences.get('acousticness', 0.5) > 0.7:
            preferences.append("acoustic and organic instruments")
        elif audio_preferences.get('acousticness', 0.5) < 0.3:
            preferences.append("electronic and synthesized production")
        
        # Context-based preferences
        context_preferences = [
            "music for working out and exercise",
            "relaxing music for studying",
            "party and social gathering music",
            "background music for focus",
            "emotional music for reflection",
            "driving and road trip music",
            "morning motivation music",
            "evening wind-down music"
        ]
        
        # Add some context preferences
        preferences.extend(random.sample(context_preferences, k=random.randint(1, 3)))
        
        return preferences
    
    def get_user_preference_text(self, user: Dict) -> str:
        """Get a random preference text for a user"""
        if not user['text_preferences']:
            return "music recommendations"
        
        # Combine random preferences into a natural sentence
        selected_prefs = random.sample(
            user['text_preferences'], 
            k=min(random.randint(1, 3), len(user['text_preferences']))
        )
        
        if len(selected_prefs) == 1:
            return f"I like {selected_prefs[0]}"
        elif len(selected_prefs) == 2:
            return f"I enjoy {selected_prefs[0]} and {selected_prefs[1]}"
        else:
            return f"I'm looking for {', '.join(selected_prefs[:-1])}, and {selected_prefs[-1]}"
    
    def simulate_user_session(self, user: Dict, n_requests: int = 3) -> List[Dict]:
        """
        Simulate a user session with multiple recommendation requests
        
        Args:
            user: User dictionary
            n_requests: Number of recommendation requests to simulate
            
        Returns:
            List of request dictionaries
        """
        session_requests = []
        
        for request_id in range(n_requests):
            # Generate different types of requests
            request_types = ['preference_based', 'mood_based', 'context_based', 'feature_based']
            request_type = random.choice(request_types)
            
            request = {
                'request_id': request_id,
                'user_id': user['user_id'],
                'type': request_type,
                'timestamp': f"2024-01-01 {10 + request_id:02d}:00:00"
            }
            
            if request_type == 'preference_based':
                request['preference_text'] = self.get_user_preference_text(user)
                request['target_features'] = None
                
            elif request_type == 'mood_based':
                moods = ['happy and energetic', 'sad and contemplative', 'calm and peaceful', 
                        'exciting and intense', 'nostalgic and emotional']
                request['preference_text'] = f"I'm feeling {random.choice(moods)}"
                request['target_features'] = None
                
            elif request_type == 'context_based':
                contexts = ['working out', 'studying', 'party', 'driving', 'relaxing', 'cooking']
                request['preference_text'] = f"music for {random.choice(contexts)}"
                request['target_features'] = None
                
            elif request_type == 'feature_based':
                request['preference_text'] = ""
                # Generate target features based on user preferences with some variation
                request['target_features'] = {}
                for feature, value in user['audio_preferences'].items():
                    # Add some randomness to the target
                    variation = random.uniform(-0.2, 0.2)
                    target_value = max(0, min(1, value + variation))
                    request['target_features'][feature] = target_value
            
            session_requests.append(request)
        
        return session_requests
    
    def get_user_stats(self, users: List[Dict]) -> Dict:
        """Get statistics about generated users"""
        
        stats = {
            'total_users': len(users),
            'avg_history_size': np.mean([u['history_size'] for u in users]),
            'archetype_distribution': {},
            'genre_preferences': {},
            'audio_feature_means': {}
        }
        
        # Archetype distribution
        archetypes = [u['archetype'] for u in users]
        unique_archetypes = list(set(archetypes))
        for archetype in unique_archetypes:
            stats['archetype_distribution'][archetype] = archetypes.count(archetype)
        
        # Genre preferences
        all_genres = []
        for user in users:
            all_genres.extend(user['preferred_genres'])
        
        unique_genres = list(set(all_genres))
        for genre in unique_genres:
            stats['genre_preferences'][genre] = all_genres.count(genre)
        
        # Audio feature means
        audio_features = ['danceability', 'energy', 'valence', 'acousticness']
        for feature in audio_features:
            feature_values = []
            for user in users:
                if feature in user['audio_preferences']:
                    feature_values.append(user['audio_preferences'][feature])
            
            if feature_values:
                stats['audio_feature_means'][feature] = np.mean(feature_values)
        
        return stats

if __name__ == "__main__":
    # Example usage
    from data_loader import MusicDataLoader
    
    # Load data
    loader = MusicDataLoader()
    tracks_df = loader.load_spotify_dataset()
    
    # Generate users
    simulator = UserSimulator(tracks_df)
    users = simulator.generate_user_histories(n_users=20)
    
    # Show example user
    example_user = users[0]
    print(f"\n👤 Example user: {example_user['user_id']}")
    print(f"🎭 Archetype: {example_user['archetype']}")
    print(f"🎵 Preferred genres: {example_user['preferred_genres']}")
    print(f"📚 Listening history: {example_user['history_size']} tracks")
    print(f"💭 Sample preference: {simulator.get_user_preference_text(example_user)}")
    
    # Show stats
    stats = simulator.get_user_stats(users)
    print(f"\n📊 User Statistics:")
    print(f"Total users: {stats['total_users']}")
    print(f"Average history size: {stats['avg_history_size']:.1f}")
    print(f"Archetype distribution: {stats['archetype_distribution']}")
    print(f"Popular genres: {dict(list(stats['genre_preferences'].items())[:5])}")
    
    # Simulate user session
    session = simulator.simulate_user_session(example_user, n_requests=3)
    print(f"\n🎮 Example user session:")
    for request in session:
        print(f"  {request['type']}: {request.get('preference_text', 'Feature-based request')}") 