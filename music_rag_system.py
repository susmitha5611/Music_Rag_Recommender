"""
Music Recommendation RAG System
Core implementation of the RAG-based music recommendation engine
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
from typing import List, Dict, Tuple, Optional

class MusicRAGSystem:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the Music RAG System
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        print("🚀 Initializing Music RAG System...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.tracks_df = None
        print("✅ RAG system initialized")
        
    def setup_vector_store(self, tracks_df):
        """
        Setup ChromaDB collection with track embeddings
        
        Args:
            tracks_df: DataFrame containing track information with descriptions
        """
        self.tracks_df = tracks_df
        
        # Create or get collection
        try:
            # Try to delete existing collection first
            try:
                self.chroma_client.delete_collection(name="music_tracks")
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name="music_tracks",
                metadata={"description": "Music track embeddings for RAG"}
            )
        except Exception as e:
            print(f"⚠️ Collection creation issue: {e}")
            self.collection = self.chroma_client.get_collection("music_tracks")
        
        # Generate embeddings for track descriptions
        print("🔄 Generating embeddings for track descriptions...")
        descriptions = tracks_df['description'].tolist()
        
        # Process in batches to avoid memory issues
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_descriptions, show_progress_bar=True)
            all_embeddings.extend(batch_embeddings.tolist())
        
        # Add to ChromaDB
        print("📚 Adding tracks to vector database...")
        
        # Prepare metadata (only serializable types)
        metadatas = []
        for _, row in tracks_df.iterrows():
            metadata = {
                'track_id': str(row['track_id']),
                'track_name': str(row['track_name']),
                'artist_name': str(row['artist_name']),
                'genre': str(row['genre'])
            }
            if 'popularity' in row and pd.notna(row['popularity']):
                metadata['popularity'] = float(row['popularity'])
            if 'year' in row and pd.notna(row['year']):
                metadata['year'] = int(row['year'])
            metadatas.append(metadata)
        
        self.collection.add(
            embeddings=all_embeddings,
            documents=descriptions,
            metadatas=metadatas,
            ids=tracks_df['track_id'].astype(str).tolist()
        )
        
        print(f"✅ Added {len(descriptions)} tracks to vector store")
    
    def search_similar_tracks(self, query: str, n_results: int = 10, filter_metadata: Dict = None):
        """
        Search for similar tracks using semantic similarity
        
        Args:
            query: Text query describing desired music
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            ChromaDB query results
        """
        if not self.collection:
            raise ValueError("Vector store not initialized. Call setup_vector_store first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        search_kwargs = {
            'query_embeddings': query_embedding.tolist(),
            'n_results': n_results
        }
        
        if filter_metadata:
            search_kwargs['where'] = filter_metadata
        
        results = self.collection.query(**search_kwargs)
        
        return results
    
    def get_recommendations(self, 
                          user_listening_history: List[str] = None, 
                          preference_text: str = "", 
                          n_recommendations: int = 10,
                          genre_filter: str = None) -> Tuple[List[str], Dict]:
        """
        Generate recommendations using RAG approach
        
        Args:
            user_listening_history: List of track IDs user has listened to
            preference_text: Text description of user preferences
            n_recommendations: Number of recommendations to return
            genre_filter: Optional genre to filter by
            
        Returns:
            Tuple of (recommended_track_ids, search_results)
        """
        if user_listening_history is None:
            user_listening_history = []
        
        # Create query from listening history and preferences
        if user_listening_history and len(user_listening_history) > 0:
            history_tracks = self.tracks_df[self.tracks_df['track_id'].isin(user_listening_history)]
            history_description = self._create_history_summary(history_tracks)
        else:
            history_description = "No listening history available."
        
        # Combine with user preferences
        full_query = f"{preference_text} {history_description}".strip()
        
        # Add genre filter if specified
        filter_metadata = None
        if genre_filter:
            filter_metadata = {"genre": {"$eq": genre_filter}}
        
        # Search for similar tracks
        search_results = self.search_similar_tracks(
            full_query, 
            n_results=n_recommendations*2,  # Get more to filter duplicates
            filter_metadata=filter_metadata
        )
        
        # Filter out tracks already in listening history
        recommended_ids = []
        for track_id in search_results['ids'][0]:
            if track_id not in user_listening_history and len(recommended_ids) < n_recommendations:
                recommended_ids.append(track_id)
        
        return recommended_ids, search_results
    
    def get_recommendations_by_audio_features(self,
                                            target_features: Dict[str, float],
                                            user_listening_history: List[str] = None,
                                            n_recommendations: int = 10) -> List[str]:
        """
        Get recommendations based on audio features similarity
        
        Args:
            target_features: Dict of audio features (e.g., {'danceability': 0.8, 'energy': 0.9})
            user_listening_history: List of track IDs to exclude
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended track IDs
        """
        if user_listening_history is None:
            user_listening_history = []
        
        available_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'instrumentalness', 'speechiness', 'liveness']
        
        # Filter features that exist in both target and dataset
        feature_cols = [f for f in available_features if f in target_features and f in self.tracks_df.columns]
        
        if not feature_cols:
            raise ValueError("No matching audio features found in dataset")
        
        # Calculate similarity scores
        target_vector = np.array([target_features[f] for f in feature_cols]).reshape(1, -1)
        track_vectors = self.tracks_df[feature_cols].values
        
        # Compute cosine similarity
        similarities = cosine_similarity(target_vector, track_vectors)[0]
        
        # Create recommendations dataframe
        recs_df = self.tracks_df.copy()
        recs_df['similarity'] = similarities
        
        # Filter out listening history
        recs_df = recs_df[~recs_df['track_id'].isin(user_listening_history)]
        
        # Sort by similarity and get top recommendations
        top_recs = recs_df.nlargest(n_recommendations, 'similarity')
        
        return top_recs['track_id'].tolist()
    
    def get_hybrid_recommendations(self,
                                 user_listening_history: List[str] = None,
                                 preference_text: str = "",
                                 target_features: Dict[str, float] = None,
                                 n_recommendations: int = 10,
                                 semantic_weight: float = 0.7) -> List[str]:
        """
        Hybrid recommendations combining semantic search and audio features
        
        Args:
            user_listening_history: List of track IDs user has listened to
            preference_text: Text description of preferences
            target_features: Target audio features
            n_recommendations: Number of recommendations
            semantic_weight: Weight for semantic vs feature-based recommendations (0-1)
            
        Returns:
            List of recommended track IDs
        """
        if user_listening_history is None:
            user_listening_history = []
        
        recommendations = []
        
        # Get semantic recommendations
        if preference_text or user_listening_history:
            semantic_recs, _ = self.get_recommendations(
                user_listening_history=user_listening_history,
                preference_text=preference_text,
                n_recommendations=int(n_recommendations * semantic_weight * 1.5)
            )
            recommendations.extend(semantic_recs[:int(n_recommendations * semantic_weight)])
        
        # Get feature-based recommendations
        if target_features:
            feature_recs = self.get_recommendations_by_audio_features(
                target_features=target_features,
                user_listening_history=user_listening_history + recommendations,
                n_recommendations=int(n_recommendations * (1 - semantic_weight) * 1.5)
            )
            recommendations.extend(feature_recs[:int(n_recommendations * (1 - semantic_weight))])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for track_id in recommendations:
            if track_id not in seen:
                seen.add(track_id)
                unique_recs.append(track_id)
        
        return unique_recs[:n_recommendations]
    
    def _create_history_summary(self, history_tracks: pd.DataFrame) -> str:
        """
        Create summary of user's listening history
        
        Args:
            history_tracks: DataFrame of tracks in user's history
            
        Returns:
            Text summary of listening patterns
        """
        if history_tracks.empty:
            return "No listening history."
        
        # Analyze listening patterns
        top_genres = history_tracks['genre'].value_counts().head(3)
        
        summary = f"User frequently listens to {', '.join(top_genres.index)} music. "
        
        # Audio feature preferences
        audio_features = ['danceability', 'energy', 'valence']
        available_features = [f for f in audio_features if f in history_tracks.columns]
        
        if available_features:
            avg_features = history_tracks[available_features].mean()
            
            feature_descriptions = []
            for feature in available_features:
                value = avg_features[feature]
                if feature == 'danceability':
                    if value > 0.7:
                        feature_descriptions.append("highly danceable music")
                    elif value > 0.5:
                        feature_descriptions.append("moderately danceable music")
                    else:
                        feature_descriptions.append("less danceable music")
                elif feature == 'energy':
                    if value > 0.7:
                        feature_descriptions.append("high energy tracks")
                    elif value > 0.5:
                        feature_descriptions.append("moderate energy music")
                    else:
                        feature_descriptions.append("calm and relaxed music")
                elif feature == 'valence':
                    if value > 0.7:
                        feature_descriptions.append("positive and uplifting songs")
                    elif value > 0.3:
                        feature_descriptions.append("emotionally neutral music")
                    else:
                        feature_descriptions.append("melancholic or sad songs")
            
            if feature_descriptions:
                summary += f"They prefer {', '.join(feature_descriptions)}. "
        
        # Popularity and era preferences
        if 'popularity' in history_tracks.columns:
            avg_popularity = history_tracks['popularity'].mean()
            if avg_popularity > 70:
                summary += "They tend to like popular mainstream music. "
            elif avg_popularity < 30:
                summary += "They prefer underground or niche music. "
        
        if 'year' in history_tracks.columns:
            avg_year = history_tracks['year'].mean()
            if avg_year > 2015:
                summary += "They prefer recent contemporary music."
            elif avg_year < 1990:
                summary += "They enjoy classic older music."
            else:
                summary += "They like music from various eras."
        
        return summary
    
    def get_track_info(self, track_ids: List[str]) -> pd.DataFrame:
        """
        Get detailed information for specific tracks
        
        Args:
            track_ids: List of track IDs
            
        Returns:
            DataFrame with track information
        """
        if self.tracks_df is None:
            raise ValueError("No tracks dataset loaded")
        
        return self.tracks_df[self.tracks_df['track_id'].isin(track_ids)]
    
    def explain_recommendation(self, track_id: str, query: str) -> str:
        """
        Generate explanation for why a track was recommended
        
        Args:
            track_id: ID of recommended track
            query: Original query/preference
            
        Returns:
            Text explanation
        """
        track_info = self.tracks_df[self.tracks_df['track_id'] == track_id]
        
        if track_info.empty:
            return "Track not found in database."
        
        track = track_info.iloc[0]
        
        explanation = f"Recommended '{track['track_name']}' by {track['artist_name']} "
        explanation += f"because it's a {track['genre']} track that matches your preferences. "
        
        # Add audio feature explanations based on query keywords
        query_lower = query.lower()
        
        if 'energetic' in query_lower or 'energy' in query_lower:
            if 'energy' in track and track['energy'] > 0.7:
                explanation += "It has high energy that fits your request for energetic music. "
        
        if 'dance' in query_lower or 'party' in query_lower:
            if 'danceability' in track and track['danceability'] > 0.7:
                explanation += "It's highly danceable, perfect for dancing or parties. "
        
        if 'calm' in query_lower or 'relax' in query_lower:
            if 'energy' in track and track['energy'] < 0.4:
                explanation += "It has a calm, relaxed energy suitable for relaxation. "
        
        if 'happy' in query_lower or 'positive' in query_lower:
            if 'valence' in track and track['valence'] > 0.7:
                explanation += "It has a positive, uplifting mood. "
        
        return explanation

if __name__ == "__main__":
    # Example usage
    from data_loader import MusicDataLoader
    
    # Load data
    loader = MusicDataLoader()
    tracks_df = loader.load_spotify_dataset()
    
    # Initialize RAG system
    rag_system = MusicRAGSystem()
    rag_system.setup_vector_store(tracks_df)
    
    # Test recommendations
    user_preferences = "I want upbeat, energetic songs for working out"
    recommended_ids, _ = rag_system.get_recommendations(
        preference_text=user_preferences,
        n_recommendations=5
    )
    
    # Show results
    recommended_tracks = rag_system.get_track_info(recommended_ids)
    print(f"\n🎵 Recommendations for: '{user_preferences}'")
    print(recommended_tracks[['track_name', 'artist_name', 'genre', 'energy', 'danceability']].to_string(index=False)) 

    def update_embedding_model(self, new_model):
        """
        Update the embedding model used by the RAG system.
        This is useful after fine-tuning with RLHF.
        """
        self.embedding_model = new_model
        print("✅ Embedding model updated.")


