"""
Main Script for Music Recommendation RAG System
Comprehensive example demonstrating the full pipeline
"""

import argparse
import time
import warnings
warnings.filterwarnings('ignore')

from data_loader import MusicDataLoader, download_spotify_dataset
from music_rag_system import MusicRAGSystem
from user_simulator import UserSimulator
from evaluator import MusicRecommendationEvaluator
from visualization import MusicRecommendationVisualizer
from rlhf_module import RLHFTrainer

def setup_system(dataset_path=None, n_sample_users=50):
    """
    Setup the complete Music RAG system
    
    Args:
        dataset_path: Path to dataset file (optional)
        n_sample_users: Number of sample users to generate
        
    Returns:
        Tuple of (rag_system, tracks_df, users, evaluator, visualizer)
    """
    print("🎵 Setting up Music Recommendation RAG System...")
    
    # 1. Load Dataset
    print("\n📊 Step 1: Loading Dataset")
    loader = MusicDataLoader()
    
    # Use final_dataset.csv by default, or specified path
    if dataset_path is None:
        tracks_df = loader.load_final_dataset('final_dataset.csv')
    else:
        tracks_df = loader.load_final_dataset(dataset_path)
    
    # Show dataset info
    info = loader.get_dataset_info()
    print(f"✅ Dataset loaded: {info['total_tracks']} tracks, {info['unique_artists']} artists")
    
    # 2. Initialize RAG System
    print("\n🧠 Step 2: Initializing RAG System")
    rag_system = MusicRAGSystem(embedding_model_name='all-MiniLM-L6-v2')
    rag_system.setup_vector_store(tracks_df)
    
    # 3. Generate Users
    print(f"\n👥 Step 3: Generating {n_sample_users} Sample Users")
    simulator = UserSimulator(tracks_df)
    users = simulator.generate_user_histories(n_users=n_sample_users)
    
    # Show user stats
    stats = simulator.get_user_stats(users)
    print(f"✅ Generated users with {stats['avg_history_size']:.1f} avg tracks per user")
    
    # 4. Initialize Evaluator and Visualizer
    print("\n📈 Step 4: Setting up Evaluation & Visualization")
    evaluator = MusicRecommendationEvaluator(tracks_df)
    visualizer = MusicRecommendationVisualizer(tracks_df)
    
    print("✅ System setup complete!")
    return rag_system, tracks_df, users, evaluator, visualizer

def test_single_user_recommendations(rag_system, users, simulator, evaluator):
    """
    Test recommendations for a single user
    """
    print("\n🎯 Testing Single User Recommendations")
    print("=" * 50)
    
    # Select test user
    test_user = users[0]
    print(f"👤 Test User: {test_user['user_id']}")
    print(f"🎭 Archetype: {test_user['archetype']}")
    print(f"🎵 Preferred Genres: {test_user['preferred_genres']}")
    print(f"📚 Listening History: {test_user['history_size']} tracks")
    
    # Generate preference text
    preference_text = simulator.get_user_preference_text(test_user)
    print(f"💭 User Preference: '{preference_text}'")
    
    # Get recommendations using different methods
    print("\n🔍 Getting Recommendations...")
    
    # 1. Semantic RAG recommendations
    start_time = time.time()
    semantic_recs, _ = rag_system.get_recommendations(
        user_listening_history=test_user['listening_history'],
        preference_text=preference_text,
        n_recommendations=10
    )
    semantic_time = time.time() - start_time
    
    # 2. Audio feature-based recommendations
    target_features = {
        'energy': 0.8,
        'danceability': 0.7,
        'valence': 0.6
    }
    
    start_time = time.time()
    feature_recs = rag_system.get_recommendations_by_audio_features(
        target_features=target_features,
        user_listening_history=test_user['listening_history'],
        n_recommendations=10
    )
    feature_time = time.time() - start_time
    
    # 3. Hybrid recommendations
    start_time = time.time()
    hybrid_recs = rag_system.get_hybrid_recommendations(
        user_listening_history=test_user['listening_history'],
        preference_text=preference_text,
        target_features=target_features,
        n_recommendations=10,
        semantic_weight=0.7
    )
    hybrid_time = time.time() - start_time
    
    print(f"⚡ Recommendation Times:")
    print(f"  • Semantic RAG: {semantic_time:.2f}s")
    print(f"  • Audio Features: {feature_time:.2f}s")
    print(f"  • Hybrid: {hybrid_time:.2f}s")
    
    # Show recommendations
    semantic_tracks = rag_system.get_track_info(semantic_recs)
    print(f"\n🎵 Semantic RAG Recommendations:")
    print(semantic_tracks[['track_name', 'artist_name', 'genre', 'energy', 'danceability']].to_string(index=False))
    
    # Evaluate recommendations
    print(f"\n📊 Evaluating Recommendations...")
    
    evaluations = {}
    for name, rec_ids in [
        ('Semantic RAG', semantic_recs),
        ('Audio Features', feature_recs),
        ('Hybrid', hybrid_recs)
    ]:
        evaluation = evaluator.evaluate_all(
            recommended_ids=rec_ids,
            user_history=test_user['listening_history'],
            preference_text=preference_text,
            target_features=target_features
        )
        evaluations[name] = evaluation
        
        print(f"\n{name} Results:")
        print(f"  • Overall Score: {evaluation['overall_score']:.3f}")
        print(f"  • Diversity: {evaluation['diversity']['diversity_score']:.3f}")
        print(f"  • Novelty: {evaluation['novelty']['novelty_score']:.3f}")
        print(f"  • Relevance: {evaluation['relevance']['relevance_score']:.3f}")
    
    # Generate explanations
    print(f"\n💡 Recommendation Explanations:")
    for i, track_id in enumerate(semantic_recs[:3]):
        explanation = rag_system.explain_recommendation(track_id, preference_text)
        print(f"  {i+1}. {explanation}")
    
    return evaluations, semantic_tracks, test_user

def run_large_scale_evaluation(rag_system, users, simulator, evaluator, n_test_users=20):
    """
    Run large-scale evaluation across multiple users
    """
    print(f"\n📊 Running Large-Scale Evaluation ({n_test_users} users)")
    print("=" * 50)
    
    all_evaluations = []
    test_preferences = [
        "upbeat energetic workout music",
        "relaxing calm evening music", 
        "party dance music",
        "focus instrumental music",
        "emotional sad songs",
        "acoustic folk music",
        "electronic dance beats",
        "jazz and blues music",
        "rock and alternative music",
        "classical and orchestral music"
    ]
    
    print("🔄 Processing users...")
    for i, user in enumerate(users[:n_test_users]):
        if i % 5 == 0:
            print(f"  Progress: {i}/{n_test_users} users processed")
        
        # Rotate through different preference types
        preference = test_preferences[i % len(test_preferences)]
        
        try:
            # Generate recommendations
            recommended_ids, _ = rag_system.get_recommendations(
                user_listening_history=user['listening_history'],
                preference_text=preference,
                n_recommendations=10
            )
            
            # Evaluate
            evaluation = evaluator.evaluate_all(
                recommended_ids=recommended_ids,
                user_history=user['listening_history'],
                preference_text=preference
            )
            
            evaluation['user_id'] = user['user_id']
            evaluation['user_archetype'] = user['archetype']
            evaluation['preference_type'] = preference.split()[0]  # First word as category
            all_evaluations.append(evaluation)
            
        except Exception as e:
            print(f"⚠️ Error evaluating user {user['user_id']}: {e}")
    
    print(f"✅ Completed evaluation for {len(all_evaluations)} users")
    
    # Aggregate results
    aggregated = evaluator.evaluate_batch(all_evaluations)
    
    print(f"\n📈 Aggregate Results:")
    print(f"🎯 Average Overall Score: {aggregated['overall_score']['mean']:.3f} ± {aggregated['overall_score']['std']:.3f}")
    print(f"🌈 Average Diversity: {aggregated['diversity']['mean']:.3f}")
    print(f"✨ Average Novelty: {aggregated['novelty']['mean']:.3f}")
    print(f"🎵 Average Relevance: {aggregated['relevance']['mean']:.3f}")
    
    # Analysis by user archetype
    archetype_stats = {}
    for archetype in ['mainstream_listener', 'music_explorer', 'niche_enthusiast', 'casual_listener']:
        archetype_evals = [e for e in all_evaluations if e.get('user_archetype') == archetype]
        if archetype_evals:
            avg_score = sum(e['overall_score'] for e in archetype_evals) / len(archetype_evals)
            archetype_stats[archetype] = avg_score
    
    if archetype_stats:
        print(f"\n👥 Performance by User Archetype:")
        for archetype, score in sorted(archetype_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {archetype}: {score:.3f}")
    
    return all_evaluations, aggregated

def create_visualizations(visualizer, evaluations, rag_system, test_user, users):
    """
    Create comprehensive visualizations
    """
    print(f"\n🎨 Creating Visualizations")
    print("=" * 50)
    
    try:
        # Get latest recommendations for visualization
        preference_text = "upbeat energetic music"
        recommended_ids, _ = rag_system.get_recommendations(
            user_listening_history=test_user['listening_history'],
            preference_text=preference_text,
            n_recommendations=10
        )
        
        recommended_tracks = rag_system.get_track_info(recommended_ids)
        user_history_tracks = rag_system.get_track_info(test_user['listening_history'])
        
        print("📊 1. Evaluation Score Distributions")
        visualizer.plot_evaluation_scores(evaluations)
        
        print("🎵 2. Genre Analysis")
        visualizer.plot_genre_analysis(recommended_tracks, user_history_tracks)
        
        print("👥 3. User Preference Analysis")
        visualizer.plot_user_preference_analysis(users)
        
        print("⏰ 4. Temporal Analysis")
        visualizer.plot_temporal_analysis(recommended_tracks, user_history_tracks)
        
        print("📈 5. Comprehensive Dashboard")
        visualizer.create_dashboard(evaluations, recommended_tracks, user_history_tracks)
        
        print("✅ All visualizations created!")
        
    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")
        print("Continuing without visualizations...")

def main():
    """
    Main function to run the complete Music RAG system demo
    """
    parser = argparse.ArgumentParser(description='Music Recommendation RAG System')
    parser.add_argument('--dataset', type=str, help='Path to dataset file (optional)')
    parser.add_argument('--users', type=int, default=50, help='Number of sample users to generate')
    parser.add_argument('--test-users', type=int, default=20, help='Number of users for evaluation')
    parser.add_argument('--skip-viz', action='store_true', help='Skip visualization creation')
    parser.add_argument('--download-info', action='store_true', help='Show dataset download information')
    parser.add_argument('--train-rlhf', action='store_true', help='Train RLHF model for recommendation system')
    
    args = parser.parse_args()
    
    # Show download information if requested
    if args.download_info:
        download_spotify_dataset()
        return
    
    print("🎵 Music Recommendation RAG System")
    print("=" * 60)
    print("Advanced Music Recommendation using Retrieval-Augmented Generation")
    print("Features: Real datasets, semantic search, comprehensive evaluation")
    print("=" * 60)
    
    try:
        # Setup system
        rag_system, tracks_df, users, evaluator, visualizer = setup_system(
            dataset_path=args.dataset,
            n_sample_users=args.users
        )

        # Train RLHF if requested
        if args.train_rlhf:
            print("\n🚀 Step 5: Training RLHF Model")
            rlhf_trainer = RLHFTrainer(rag_system)
            rlhf_trainer.train_rlhf(tracks_df)
            # The rag_system's embedding model is updated within train_rlhf
            print("✅ RLHF training complete and RAG system updated.")

        # Test single user
        single_user_evaluations, recommended_tracks, test_user = test_single_user_recommendations(
            rag_system, users, UserSimulator(tracks_df), evaluator
        )
        
        # Large-scale evaluation
        all_evaluations, aggregated_results = run_large_scale_evaluation(
            rag_system, users, UserSimulator(tracks_df), evaluator, 
            n_test_users=args.test_users
        )
        
        # Create visualizations
        if not args.skip_viz:
            create_visualizations(visualizer, all_evaluations, rag_system, test_user, users)
        
        # Final summary
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 50)
        print(f"✅ System successfully processed {len(all_evaluations)} user evaluations")
        print(f"📊 Overall system performance: {aggregated_results['overall_score']['mean']:.3f}")
        print(f"🎵 Dataset: {len(tracks_df)} tracks across {tracks_df['genre'].nunique()} genres")
        print(f"👥 Users: {len(users)} generated with diverse listening patterns")
        
        # Best performing aspects
        best_metric = max([
            ('Diversity', aggregated_results['diversity']['mean']),
            ('Novelty', aggregated_results['novelty']['mean']),
            ('Relevance', aggregated_results['relevance']['mean'])
        ], key=lambda x: x[1])
        
        print(f"🏆 Best performing metric: {best_metric[0]} ({best_metric[1]:.3f})")
        
        print(f"\n🚀 System is ready for production use!")
        print(f"💡 To use real Spotify data, download from:")
        print(f"   https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks")
        print(f"   and run with --dataset spotify_tracks.csv")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running system: {e}")
        print(f"Please check your setup and try again")

if __name__ == "__main__":
    main() 