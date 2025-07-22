"""
Evaluation Module for Music Recommendation System
Comprehensive metrics for diversity, novelty, and relevance
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

class MusicRecommendationEvaluator:
    def __init__(self, tracks_df: pd.DataFrame):
        """
        Initialize the evaluator
        
        Args:
            tracks_df: DataFrame containing track information
        """
        self.tracks_df = tracks_df
        self.audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                              'instrumentalness', 'speechiness', 'liveness']
        
    def evaluate_diversity(self, recommended_ids: List[str]) -> Dict:
        """
        Measure genre and audio feature diversity in recommendations
        
        Args:
            recommended_ids: List of recommended track IDs
            
        Returns:
            Dictionary with diversity metrics
        """
        recommended_tracks = self.tracks_df[self.tracks_df['track_id'].isin(recommended_ids)]
        
        if recommended_tracks.empty:
            return {'error': 'No recommended tracks found in dataset'}
        
        # Genre diversity (number of unique genres)
        unique_genres = len(recommended_tracks['genre'].unique())
        total_genres = len(self.tracks_df['genre'].unique())
        genre_diversity = unique_genres / total_genres
        
        # Artist diversity
        unique_artists = len(recommended_tracks['artist_name'].unique())
        artist_diversity = unique_artists / len(recommended_tracks)
        
        # Audio feature diversity (standard deviation across recommendations)
        available_features = [f for f in self.audio_features if f in recommended_tracks.columns]
        feature_diversities = {}
        avg_feature_diversity = 0
        
        if available_features:
            for feature in available_features:
                feature_std = recommended_tracks[feature].std()
                feature_diversities[feature] = feature_std
            
            avg_feature_diversity = np.mean(list(feature_diversities.values()))
        
        # Temporal diversity (if year information available)
        temporal_diversity = 0
        if 'year' in recommended_tracks.columns:
            year_range = recommended_tracks['year'].max() - recommended_tracks['year'].min()
            max_year_range = self.tracks_df['year'].max() - self.tracks_df['year'].min()
            temporal_diversity = year_range / max_year_range if max_year_range > 0 else 0
        
        # Popularity diversity
        popularity_diversity = 0
        if 'popularity' in recommended_tracks.columns:
            popularity_std = recommended_tracks['popularity'].std()
            max_popularity_std = self.tracks_df['popularity'].std()
            popularity_diversity = popularity_std / max_popularity_std if max_popularity_std > 0 else 0
        
        # Overall diversity score (weighted combination)
        diversity_score = (
            genre_diversity * 0.3 +
            artist_diversity * 0.2 +
            min(avg_feature_diversity, 1.0) * 0.3 +
            temporal_diversity * 0.1 +
            popularity_diversity * 0.1
        )
        
        return {
            'diversity_score': diversity_score,
            'genre_diversity': genre_diversity,
            'unique_genres': unique_genres,
            'artist_diversity': artist_diversity,
            'unique_artists': unique_artists,
            'audio_diversity': avg_feature_diversity,
            'feature_diversities': feature_diversities,
            'temporal_diversity': temporal_diversity,
            'popularity_diversity': popularity_diversity
        }
    
    def evaluate_novelty(self, recommended_ids: List[str], user_history: List[str]) -> Dict:
        """
        Measure how novel recommendations are compared to user history
        
        Args:
            recommended_ids: List of recommended track IDs
            user_history: List of track IDs in user's listening history
            
        Returns:
            Dictionary with novelty metrics
        """
        recommended_tracks = self.tracks_df[self.tracks_df['track_id'].isin(recommended_ids)]
        history_tracks = self.tracks_df[self.tracks_df['track_id'].isin(user_history)]
        
        if recommended_tracks.empty:
            return {'error': 'No recommended tracks found in dataset'}
        
        if history_tracks.empty:
            return {
                'novelty_score': 1.0,
                'new_artists': 1.0,
                'new_genres': 1.0,
                'new_albums': 1.0,
                'audio_novelty': 1.0,
                'popularity_novelty': 0.5
            }
        
        # Artist novelty
        history_artists = set(history_tracks['artist_name'])
        recommended_artists = set(recommended_tracks['artist_name'])
        new_artists = len(recommended_artists - history_artists) / len(recommended_artists)
        
        # Genre novelty
        history_genres = set(history_tracks['genre'])
        recommended_genres = set(recommended_tracks['genre'])
        new_genres = len(recommended_genres - history_genres) / len(recommended_genres)
        
        # Album novelty (if album information available)
        new_albums = 0
        if 'album_name' in recommended_tracks.columns and 'album_name' in history_tracks.columns:
            history_albums = set(history_tracks['album_name'])
            recommended_albums = set(recommended_tracks['album_name'])
            new_albums = len(recommended_albums - history_albums) / len(recommended_albums)
        
        # Audio feature novelty
        audio_novelty = 0
        available_features = [f for f in self.audio_features if f in recommended_tracks.columns and f in history_tracks.columns]
        
        if available_features:
            history_profile = history_tracks[available_features].mean()
            recommended_profile = recommended_tracks[available_features].mean()
            
            # Calculate Euclidean distance between profiles
            distance = np.linalg.norm(history_profile - recommended_profile)
            # Normalize to [0, 1] range (max distance is sqrt(len(features)))
            max_distance = np.sqrt(len(available_features))
            audio_novelty = min(distance / max_distance, 1.0)
        
        # Popularity novelty (recommending less popular items is more novel)
        popularity_novelty = 0.5  # Default neutral
        if 'popularity' in recommended_tracks.columns:
            avg_rec_popularity = recommended_tracks['popularity'].mean()
            # Higher novelty for lower popularity
            popularity_novelty = (100 - avg_rec_popularity) / 100
        
        # Temporal novelty (if year information available)
        temporal_novelty = 0
        if 'year' in recommended_tracks.columns and 'year' in history_tracks.columns:
            avg_history_year = history_tracks['year'].mean()
            avg_rec_year = recommended_tracks['year'].mean()
            year_diff = abs(avg_rec_year - avg_history_year)
            max_year_diff = self.tracks_df['year'].max() - self.tracks_df['year'].min()
            temporal_novelty = min(year_diff / max_year_diff, 1.0) if max_year_diff > 0 else 0
        
        # Overall novelty score
        novelty_score = (
            new_artists * 0.25 +
            new_genres * 0.25 +
            audio_novelty * 0.25 +
            popularity_novelty * 0.15 +
            temporal_novelty * 0.1
        )
        
        return {
            'novelty_score': novelty_score,
            'new_artists': new_artists,
            'new_genres': new_genres,
            'new_albums': new_albums,
            'audio_novelty': audio_novelty,
            'popularity_novelty': popularity_novelty,
            'temporal_novelty': temporal_novelty
        }
    
    def evaluate_relevance(self, recommended_ids: List[str], user_history: List[str], 
                          preference_text: str = "", target_features: Dict[str, float] = None) -> Dict:
        """
        Measure relevance to user preferences and history
        
        Args:
            recommended_ids: List of recommended track IDs
            user_history: List of track IDs in user's listening history
            preference_text: User's textual preferences
            target_features: Target audio features
            
        Returns:
            Dictionary with relevance metrics
        """
        recommended_tracks = self.tracks_df[self.tracks_df['track_id'].isin(recommended_ids)]
        history_tracks = self.tracks_df[self.tracks_df['track_id'].isin(user_history)]
        
        if recommended_tracks.empty:
            return {'error': 'No recommended tracks found in dataset'}
        
        relevance_components = {}
        
        # Audio feature similarity to history
        audio_similarity = 0.5  # Default neutral
        if not history_tracks.empty:
            available_features = [f for f in self.audio_features if f in recommended_tracks.columns and f in history_tracks.columns]
            
            if available_features:
                history_profile = history_tracks[available_features].mean()
                recommended_profile = recommended_tracks[available_features].mean()
                
                # Cosine similarity between profiles
                audio_similarity = cosine_similarity([history_profile], [recommended_profile])[0][0]
                audio_similarity = max(0, audio_similarity)  # Ensure non-negative
        
        relevance_components['audio_similarity'] = audio_similarity
        
        # Genre overlap with history
        genre_overlap = 0
        if not history_tracks.empty:
            history_genres = set(history_tracks['genre'])
            recommended_genres = set(recommended_tracks['genre'])
            if recommended_genres:
                genre_overlap = len(history_genres & recommended_genres) / len(recommended_genres)
        
        relevance_components['genre_overlap'] = genre_overlap
        
        # Target feature matching (if provided)
        feature_matching = 0.5  # Default neutral
        if target_features:
            available_target_features = [f for f in target_features.keys() if f in recommended_tracks.columns]
            
            if available_target_features:
                target_values = np.array([target_features[f] for f in available_target_features])
                rec_values = recommended_tracks[available_target_features].mean().values
                
                # Calculate similarity (1 - normalized MSE)
                mse = np.mean((target_values - rec_values) ** 2)
                feature_matching = max(0, 1 - mse)
        
        relevance_components['feature_matching'] = feature_matching
        
        # Textual preference matching (basic keyword matching)
        text_relevance = 0.5  # Default neutral
        if preference_text:
            text_relevance = self._calculate_text_relevance(recommended_tracks, preference_text)
        
        relevance_components['text_relevance'] = text_relevance
        
        # Popularity alignment
        popularity_alignment = 0.5  # Default neutral
        if not history_tracks.empty and 'popularity' in recommended_tracks.columns and 'popularity' in history_tracks.columns:
            history_popularity = history_tracks['popularity'].mean()
            rec_popularity = recommended_tracks['popularity'].mean()
            
            # Higher alignment for similar popularity levels
            popularity_diff = abs(history_popularity - rec_popularity) / 100
            popularity_alignment = max(0, 1 - popularity_diff)
        
        relevance_components['popularity_alignment'] = popularity_alignment
        
        # Overall relevance score
        if target_features and preference_text:
            # Both target features and text provided
            relevance_score = (
                audio_similarity * 0.25 +
                genre_overlap * 0.2 +
                feature_matching * 0.3 +
                text_relevance * 0.15 +
                popularity_alignment * 0.1
            )
        elif target_features:
            # Only target features provided
            relevance_score = (
                audio_similarity * 0.3 +
                genre_overlap * 0.25 +
                feature_matching * 0.35 +
                popularity_alignment * 0.1
            )
        elif preference_text:
            # Only text preferences provided
            relevance_score = (
                audio_similarity * 0.35 +
                genre_overlap * 0.3 +
                text_relevance * 0.25 +
                popularity_alignment * 0.1
            )
        else:
            # Only history-based relevance
            relevance_score = (
                audio_similarity * 0.5 +
                genre_overlap * 0.4 +
                popularity_alignment * 0.1
            )
        
        return {
            'relevance_score': relevance_score,
            **relevance_components
        }
    
    def _calculate_text_relevance(self, recommended_tracks: pd.DataFrame, preference_text: str) -> float:
        """Calculate relevance based on textual preferences"""
        preference_lower = preference_text.lower()
        
        # Extract keywords and map to features/genres
        keyword_scores = []
        
        # Genre keywords
        genre_keywords = {
            'rock': ['rock', 'guitar'],
            'pop': ['pop', 'catchy', 'mainstream'],
            'electronic': ['electronic', 'synth', 'edm', 'dance'],
            'jazz': ['jazz', 'swing', 'improvisation'],
            'classical': ['classical', 'orchestra', 'symphony'],
            'hip-hop': ['hip-hop', 'rap', 'beats'],
            'country': ['country', 'folk'],
            'r&b': ['r&b', 'soul']
        }
        
        # Check genre relevance
        for genre, keywords in genre_keywords.items():
            if any(keyword in preference_lower for keyword in keywords):
                genre_count = (recommended_tracks['genre'] == genre).sum()
                genre_score = genre_count / len(recommended_tracks)
                keyword_scores.append(genre_score)
        
        # Audio feature keywords
        feature_keywords = {
            'energetic': ('energy', 0.7, 1.0),
            'calm': ('energy', 0.0, 0.4),
            'danceable': ('danceability', 0.7, 1.0),
            'happy': ('valence', 0.7, 1.0),
            'sad': ('valence', 0.0, 0.3),
            'acoustic': ('acousticness', 0.7, 1.0),
            'party': ('danceability', 0.7, 1.0),
            'workout': ('energy', 0.7, 1.0),
            'relaxing': ('energy', 0.0, 0.4)
        }
        
        for keyword, (feature, min_val, max_val) in feature_keywords.items():
            if keyword in preference_lower and feature in recommended_tracks.columns:
                feature_values = recommended_tracks[feature]
                matching_tracks = ((feature_values >= min_val) & (feature_values <= max_val)).sum()
                feature_score = matching_tracks / len(recommended_tracks)
                keyword_scores.append(feature_score)
        
        # Return average of all matched keyword scores
        return np.mean(keyword_scores) if keyword_scores else 0.5
    
    def evaluate_all(self, recommended_ids: List[str], user_history: List[str], 
                    preference_text: str = "", target_features: Dict[str, float] = None) -> Dict:
        """
        Comprehensive evaluation of recommendations
        
        Args:
            recommended_ids: List of recommended track IDs
            user_history: List of track IDs in user's listening history
            preference_text: User's textual preferences
            target_features: Target audio features
            
        Returns:
            Dictionary with all evaluation metrics
        """
        diversity = self.evaluate_diversity(recommended_ids)
        novelty = self.evaluate_novelty(recommended_ids, user_history)
        relevance = self.evaluate_relevance(recommended_ids, user_history, preference_text, target_features)
        
        # Check for errors
        if 'error' in diversity:
            return {'error': diversity['error']}
        
        # Overall score (balanced combination)
        overall_score = (
            diversity['diversity_score'] * 0.3 +
            novelty['novelty_score'] * 0.3 +
            relevance['relevance_score'] * 0.4
        )
        
        return {
            'overall_score': overall_score,
            'diversity': diversity,
            'novelty': novelty,
            'relevance': relevance,
            'n_recommendations': len(recommended_ids)
        }
    
    def evaluate_batch(self, evaluations: List[Dict]) -> Dict:
        """
        Aggregate evaluation results across multiple users/sessions
        
        Args:
            evaluations: List of evaluation dictionaries
            
        Returns:
            Aggregated statistics
        """
        if not evaluations:
            return {'error': 'No evaluations provided'}
        
        # Filter out evaluations with errors
        valid_evaluations = [e for e in evaluations if 'error' not in e]
        
        if not valid_evaluations:
            return {'error': 'No valid evaluations found'}
        
        # Aggregate metrics
        aggregated = {
            'n_evaluations': len(valid_evaluations),
            'overall_score': {
                'mean': np.mean([e['overall_score'] for e in valid_evaluations]),
                'std': np.std([e['overall_score'] for e in valid_evaluations]),
                'min': np.min([e['overall_score'] for e in valid_evaluations]),
                'max': np.max([e['overall_score'] for e in valid_evaluations])
            }
        }
        
        # Aggregate component scores
        for component in ['diversity', 'novelty', 'relevance']:
            scores = [e[component][f'{component}_score'] for e in valid_evaluations]
            aggregated[component] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        # Detailed diversity metrics
        diversity_metrics = ['genre_diversity', 'artist_diversity', 'audio_diversity']
        for metric in diversity_metrics:
            values = [e['diversity'][metric] for e in valid_evaluations if metric in e['diversity']]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
        
        # Detailed novelty metrics
        novelty_metrics = ['new_artists', 'new_genres', 'audio_novelty']
        for metric in novelty_metrics:
            values = [e['novelty'][metric] for e in valid_evaluations if metric in e['novelty']]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
        
        # Detailed relevance metrics
        relevance_metrics = ['audio_similarity', 'genre_overlap', 'feature_matching']
        for metric in relevance_metrics:
            values = [e['relevance'][metric] for e in valid_evaluations if metric in e['relevance']]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
        
        return aggregated
    
    def compare_algorithms(self, algorithm_results: Dict[str, List[Dict]]) -> Dict:
        """
        Compare multiple recommendation algorithms
        
        Args:
            algorithm_results: Dict mapping algorithm names to lists of evaluation results
            
        Returns:
            Comparison statistics
        """
        comparison = {}
        
        for algo_name, results in algorithm_results.items():
            aggregated = self.evaluate_batch(results)
            comparison[algo_name] = aggregated
        
        # Create comparison summary
        summary = {
            'algorithm_count': len(algorithm_results),
            'best_overall': max(comparison.keys(), key=lambda k: comparison[k]['overall_score']['mean']),
            'best_diversity': max(comparison.keys(), key=lambda k: comparison[k]['diversity']['mean']),
            'best_novelty': max(comparison.keys(), key=lambda k: comparison[k]['novelty']['mean']),
            'best_relevance': max(comparison.keys(), key=lambda k: comparison[k]['relevance']['mean'])
        }
        
        return {
            'summary': summary,
            'detailed_comparison': comparison
        }

if __name__ == "__main__":
    # Example usage
    from data_loader import MusicDataLoader
    from music_rag_system import MusicRAGSystem
    from user_simulator import UserSimulator
    
    # Load data and setup system
    loader = MusicDataLoader()
    tracks_df = loader.load_spotify_dataset()
    
    rag_system = MusicRAGSystem()
    rag_system.setup_vector_store(tracks_df)
    
    simulator = UserSimulator(tracks_df)
    users = simulator.generate_user_histories(n_users=5)
    
    # Initialize evaluator
    evaluator = MusicRecommendationEvaluator(tracks_df)
    
    # Test evaluation
    test_user = users[0]
    preference_text = simulator.get_user_preference_text(test_user)
    
    recommended_ids, _ = rag_system.get_recommendations(
        user_listening_history=test_user['listening_history'],
        preference_text=preference_text,
        n_recommendations=10
    )
    
    # Evaluate recommendations
    evaluation = evaluator.evaluate_all(
        recommended_ids=recommended_ids,
        user_history=test_user['listening_history'],
        preference_text=preference_text
    )
    
    print("📈 Evaluation Results:")
    print(f"Overall Score: {evaluation['overall_score']:.3f}")
    print(f"Diversity: {evaluation['diversity']['diversity_score']:.3f}")
    print(f"Novelty: {evaluation['novelty']['novelty_score']:.3f}")
    print(f"Relevance: {evaluation['relevance']['relevance_score']:.3f}") 