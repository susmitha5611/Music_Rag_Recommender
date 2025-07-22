"""
Visualization Module for Music Recommendation System
Comprehensive analysis and plotting of recommendation results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

class MusicRecommendationVisualizer:
    def __init__(self, tracks_df: pd.DataFrame):
        """
        Initialize the visualizer
        
        Args:
            tracks_df: DataFrame containing track information
        """
        self.tracks_df = tracks_df
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_evaluation_scores(self, evaluations: List[Dict], save_path: Optional[str] = None):
        """
        Plot evaluation score distributions
        
        Args:
            evaluations: List of evaluation dictionaries
            save_path: Optional path to save the plot
        """
        # Extract scores
        scores_data = []
        for eval_result in evaluations:
            if 'error' not in eval_result:
                scores_data.append({
                    'Overall': eval_result['overall_score'],
                    'Diversity': eval_result['diversity']['diversity_score'],
                    'Novelty': eval_result['novelty']['novelty_score'],
                    'Relevance': eval_result['relevance']['relevance_score']
                })
        
        if not scores_data:
            print("No valid evaluation data to plot")
            return
        
        scores_df = pd.DataFrame(scores_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot of all scores
        scores_df.boxplot(ax=axes[0,0])
        axes[0,0].set_title('Evaluation Score Distributions')
        axes[0,0].set_ylabel('Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Histogram of overall scores
        axes[0,1].hist(scores_df['Overall'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_title('Overall Score Distribution')
        axes[0,1].set_xlabel('Overall Score')
        axes[0,1].set_ylabel('Frequency')
        
        # Correlation heatmap
        corr_matrix = scores_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Score Correlations')
        
        # Scatter plot of diversity vs relevance
        axes[1,1].scatter(scores_df['Diversity'], scores_df['Relevance'], alpha=0.6)
        axes[1,1].set_xlabel('Diversity Score')
        axes[1,1].set_ylabel('Relevance Score')
        axes[1,1].set_title('Diversity vs Relevance')
        
        # Add correlation line
        z = np.polyfit(scores_df['Diversity'], scores_df['Relevance'], 1)
        p = np.poly1d(z)
        axes[1,1].plot(scores_df['Diversity'], p(scores_df['Diversity']), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary statistics
        print("📊 Evaluation Summary Statistics:")
        print(scores_df.describe())
    
    def plot_genre_analysis(self, recommended_tracks: pd.DataFrame, user_history_tracks: pd.DataFrame = None, 
                           save_path: Optional[str] = None):
        """
        Plot genre distribution analysis
        
        Args:
            recommended_tracks: DataFrame of recommended tracks
            user_history_tracks: Optional DataFrame of user's listening history
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2 if user_history_tracks is not None else 1, figsize=(15, 6))
        
        if user_history_tracks is None:
            axes = [axes]
        
        # Recommended tracks genre distribution
        genre_counts = recommended_tracks['genre'].value_counts()
        genre_counts.plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.7)
        axes[0].set_title('Genre Distribution in Recommendations')
        axes[0].set_xlabel('Genre')
        axes[0].set_ylabel('Number of Tracks')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total_recs = len(recommended_tracks)
        for i, v in enumerate(genre_counts.values):
            axes[0].text(i, v + 0.1, f'{v/total_recs:.1%}', ha='center', va='bottom')
        
        # User history vs recommendations comparison
        if user_history_tracks is not None:
            history_genres = user_history_tracks['genre'].value_counts()
            rec_genres = recommended_tracks['genre'].value_counts()
            
            # Align genres for comparison
            all_genres = list(set(history_genres.index) | set(rec_genres.index))
            history_aligned = [history_genres.get(genre, 0) for genre in all_genres]
            rec_aligned = [rec_genres.get(genre, 0) for genre in all_genres]
            
            x = np.arange(len(all_genres))
            width = 0.35
            
            axes[1].bar(x - width/2, history_aligned, width, label='User History', alpha=0.7, color='lightcoral')
            axes[1].bar(x + width/2, rec_aligned, width, label='Recommendations', alpha=0.7, color='skyblue')
            axes[1].set_xlabel('Genre')
            axes[1].set_ylabel('Number of Tracks')
            axes[1].set_title('Genre Distribution: History vs Recommendations')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(all_genres, rotation=45)
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_audio_features_radar(self, recommended_tracks: pd.DataFrame, user_history_tracks: pd.DataFrame = None,
                                 target_features: Dict[str, float] = None, save_path: Optional[str] = None):
        """
        Create radar chart for audio features comparison
        
        Args:
            recommended_tracks: DataFrame of recommended tracks
            user_history_tracks: Optional DataFrame of user's listening history
            target_features: Optional target audio features
            save_path: Optional path to save the plot
        """
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']
        available_features = [f for f in audio_features if f in recommended_tracks.columns]
        
        if not available_features:
            print("No audio features available for radar chart")
            return
        
        # Calculate means
        rec_means = recommended_tracks[available_features].mean()
        
        # Create radar chart using plotly
        fig = go.Figure()
        
        # Add recommendations
        fig.add_trace(go.Scatterpolar(
            r=rec_means.values,
            theta=available_features,
            fill='toself',
            name='Recommendations',
            line_color='blue'
        ))
        
        # Add user history if available
        if user_history_tracks is not None and not user_history_tracks.empty:
            history_features = [f for f in available_features if f in user_history_tracks.columns]
            if history_features:
                history_means = user_history_tracks[history_features].mean()
                fig.add_trace(go.Scatterpolar(
                    r=history_means.values,
                    theta=history_features,
                    fill='toself',
                    name='User History',
                    line_color='red'
                ))
        
        # Add target features if available
        if target_features:
            target_values = [target_features.get(f, 0) for f in available_features]
            fig.add_trace(go.Scatterpolar(
                r=target_values,
                theta=available_features,
                fill='toself',
                name='Target',
                line_color='green'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Audio Features Comparison"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_diversity_vs_relevance_scatter(self, evaluations: List[Dict], save_path: Optional[str] = None):
        """
        Create scatter plot of diversity vs relevance with additional metrics
        
        Args:
            evaluations: List of evaluation dictionaries
            save_path: Optional path to save the plot
        """
        # Extract data
        plot_data = []
        for eval_result in evaluations:
            if 'error' not in eval_result:
                plot_data.append({
                    'diversity': eval_result['diversity']['diversity_score'],
                    'relevance': eval_result['relevance']['relevance_score'],
                    'novelty': eval_result['novelty']['novelty_score'],
                    'overall': eval_result['overall_score']
                })
        
        if not plot_data:
            print("No valid evaluation data to plot")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create interactive scatter plot
        fig = px.scatter(
            df,
            x='diversity',
            y='relevance',
            size='overall',
            color='novelty',
            hover_data=['overall'],
            title='Diversity vs Relevance (sized by Overall Score, colored by Novelty)',
            labels={
                'diversity': 'Diversity Score',
                'relevance': 'Relevance Score',
                'novelty': 'Novelty Score'
            }
        )
        
        # Add diagonal line for reference
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="red", width=2, dash="dash")
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_user_preference_analysis(self, users: List[Dict], save_path: Optional[str] = None):
        """
        Analyze and plot user preference patterns
        
        Args:
            users: List of user dictionaries
            save_path: Optional path to save the plot
        """
        # Extract user data
        user_data = []
        for user in users:
            user_info = {
                'user_id': user['user_id'],
                'archetype': user['archetype'],
                'history_size': user['history_size'],
                'n_preferred_genres': len(user['preferred_genres'])
            }
            
            # Add audio preferences
            for feature, value in user.get('audio_preferences', {}).items():
                user_info[f'pref_{feature}'] = value
            
            user_data.append(user_info)
        
        df = pd.DataFrame(user_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Archetype distribution
        archetype_counts = df['archetype'].value_counts()
        axes[0,0].pie(archetype_counts.values, labels=archetype_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('User Archetype Distribution')
        
        # History size by archetype
        sns.boxplot(data=df, x='archetype', y='history_size', ax=axes[0,1])
        axes[0,1].set_title('Listening History Size by Archetype')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Genre diversity by archetype
        sns.boxplot(data=df, x='archetype', y='n_preferred_genres', ax=axes[1,0])
        axes[1,0].set_title('Genre Preferences by Archetype')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Audio preference correlation
        audio_columns = [col for col in df.columns if col.startswith('pref_')]
        if audio_columns:
            audio_df = df[audio_columns].rename(columns=lambda x: x.replace('pref_', ''))
            corr_matrix = audio_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
            axes[1,1].set_title('Audio Preference Correlations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_algorithm_comparison(self, algorithm_results: Dict[str, List[Dict]], save_path: Optional[str] = None):
        """
        Compare multiple recommendation algorithms
        
        Args:
            algorithm_results: Dict mapping algorithm names to evaluation results
            save_path: Optional path to save the plot
        """
        # Aggregate results for each algorithm
        comparison_data = []
        for algo_name, results in algorithm_results.items():
            valid_results = [r for r in results if 'error' not in r]
            if valid_results:
                comparison_data.append({
                    'algorithm': algo_name,
                    'overall_mean': np.mean([r['overall_score'] for r in valid_results]),
                    'overall_std': np.std([r['overall_score'] for r in valid_results]),
                    'diversity_mean': np.mean([r['diversity']['diversity_score'] for r in valid_results]),
                    'novelty_mean': np.mean([r['novelty']['novelty_score'] for r in valid_results]),
                    'relevance_mean': np.mean([r['relevance']['relevance_score'] for r in valid_results]),
                    'n_evaluations': len(valid_results)
                })
        
        if not comparison_data:
            print("No valid algorithm comparison data")
            return
        
        df = pd.DataFrame(comparison_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall score comparison
        bars1 = axes[0,0].bar(df['algorithm'], df['overall_mean'], 
                             yerr=df['overall_std'], capsize=5, alpha=0.7)
        axes[0,0].set_title('Overall Score Comparison')
        axes[0,0].set_ylabel('Overall Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + df['overall_std'].iloc[i],
                          f'{height:.3f}', ha='center', va='bottom')
        
        # Component scores comparison
        metrics = ['diversity_mean', 'novelty_mean', 'relevance_mean']
        metric_labels = ['Diversity', 'Novelty', 'Relevance']
        
        x = np.arange(len(df))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            axes[0,1].bar(x + i*width, df[metric], width, label=label, alpha=0.7)
        
        axes[0,1].set_title('Component Scores Comparison')
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_xticks(x + width)
        axes[0,1].set_xticklabels(df['algorithm'], rotation=45)
        axes[0,1].legend()
        
        # Radar chart for all algorithms
        categories = metric_labels
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        ax_radar = plt.subplot(2, 2, 3, projection='polar')
        
        for _, row in df.iterrows():
            values = [row['diversity_mean'], row['novelty_mean'], row['relevance_mean']]
            values += [values[0]]  # Complete the circle
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=row['algorithm'])
            ax_radar.fill(angles, values, alpha=0.25)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Algorithm Performance Radar')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Number of evaluations
        bars4 = axes[1,1].bar(df['algorithm'], df['n_evaluations'], alpha=0.7, color='lightgreen')
        axes[1,1].set_title('Number of Evaluations per Algorithm')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_temporal_analysis(self, recommended_tracks: pd.DataFrame, user_history_tracks: pd.DataFrame = None,
                              save_path: Optional[str] = None):
        """
        Analyze temporal patterns in recommendations
        
        Args:
            recommended_tracks: DataFrame of recommended tracks
            user_history_tracks: Optional DataFrame of user's listening history
            save_path: Optional path to save the plot
        """
        if 'year' not in recommended_tracks.columns:
            print("No year information available for temporal analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Year distribution in recommendations
        year_counts = recommended_tracks['year'].value_counts().sort_index()
        axes[0].bar(year_counts.index, year_counts.values, alpha=0.7, color='skyblue')
        axes[0].set_title('Year Distribution in Recommendations')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of Tracks')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Decade analysis
        recommended_tracks['decade'] = (recommended_tracks['year'] // 10) * 10
        decade_counts = recommended_tracks['decade'].value_counts().sort_index()
        
        # Compare with user history if available
        if user_history_tracks is not None and 'year' in user_history_tracks.columns:
            user_history_tracks['decade'] = (user_history_tracks['year'] // 10) * 10
            history_decade_counts = user_history_tracks['decade'].value_counts().sort_index()
            
            # Align decades
            all_decades = sorted(set(decade_counts.index) | set(history_decade_counts.index))
            rec_aligned = [decade_counts.get(decade, 0) for decade in all_decades]
            hist_aligned = [history_decade_counts.get(decade, 0) for decade in all_decades]
            
            x = np.arange(len(all_decades))
            width = 0.35
            
            axes[1].bar(x - width/2, hist_aligned, width, label='User History', alpha=0.7, color='lightcoral')
            axes[1].bar(x + width/2, rec_aligned, width, label='Recommendations', alpha=0.7, color='skyblue')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f"{int(d)}s" for d in all_decades])
            axes[1].legend()
        else:
            axes[1].bar(range(len(decade_counts)), decade_counts.values, alpha=0.7, color='skyblue')
            axes[1].set_xticks(range(len(decade_counts)))
            axes[1].set_xticklabels([f"{int(d)}s" for d in decade_counts.index])
        
        axes[1].set_title('Decade Distribution')
        axes[1].set_xlabel('Decade')
        axes[1].set_ylabel('Number of Tracks')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_dashboard(self, evaluations: List[Dict], recommended_tracks: pd.DataFrame,
                        user_history_tracks: pd.DataFrame = None, save_path: Optional[str] = None):
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            evaluations: List of evaluation dictionaries
            recommended_tracks: DataFrame of recommended tracks
            user_history_tracks: Optional DataFrame of user's listening history
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Extract evaluation data
        scores_data = []
        for eval_result in evaluations:
            if 'error' not in eval_result:
                scores_data.append({
                    'Overall': eval_result['overall_score'],
                    'Diversity': eval_result['diversity']['diversity_score'],
                    'Novelty': eval_result['novelty']['novelty_score'],
                    'Relevance': eval_result['relevance']['relevance_score']
                })
        
        scores_df = pd.DataFrame(scores_data)
        
        # 1. Evaluation scores overview
        ax1 = fig.add_subplot(gs[0, :2])
        if not scores_df.empty:
            scores_df.boxplot(ax=ax1)
            ax1.set_title('Evaluation Score Distributions', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Genre distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        genre_counts = recommended_tracks['genre'].value_counts()
        genre_counts.plot(kind='bar', ax=ax2, color='skyblue', alpha=0.7)
        ax2.set_title('Genre Distribution in Recommendations', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Audio features comparison
        ax3 = fig.add_subplot(gs[1, :2])
        audio_features = ['danceability', 'energy', 'valence', 'acousticness']
        available_features = [f for f in audio_features if f in recommended_tracks.columns]
        
        if available_features:
            rec_means = recommended_tracks[available_features].mean()
            bars = ax3.bar(range(len(available_features)), rec_means.values, alpha=0.7, color='lightgreen')
            ax3.set_xticks(range(len(available_features)))
            ax3.set_xticklabels(available_features, rotation=45)
            ax3.set_title('Average Audio Features in Recommendations', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Average Value')
            
            # Add user history comparison if available
            if user_history_tracks is not None and not user_history_tracks.empty:
                hist_features = [f for f in available_features if f in user_history_tracks.columns]
                if hist_features:
                    hist_means = user_history_tracks[hist_features].mean()
                    ax3.bar(range(len(hist_features)), hist_means.values, alpha=0.5, 
                           color='red', width=0.5, label='User History')
                    ax3.legend()
        
        # 4. Popularity distribution
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'popularity' in recommended_tracks.columns:
            ax4.hist(recommended_tracks['popularity'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_title('Popularity Distribution in Recommendations', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Popularity Score')
            ax4.set_ylabel('Frequency')
        
        # 5. Score correlations
        ax5 = fig.add_subplot(gs[2, :2])
        if not scores_df.empty:
            corr_matrix = scores_df.corr()
            im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            ax5.set_xticks(range(len(corr_matrix.columns)))
            ax5.set_yticks(range(len(corr_matrix.columns)))
            ax5.set_xticklabels(corr_matrix.columns, rotation=45)
            ax5.set_yticklabels(corr_matrix.columns)
            ax5.set_title('Score Correlations', fontsize=14, fontweight='bold')
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                            ha="center", va="center", color="black")
        
        # 6. Diversity vs Relevance scatter
        ax6 = fig.add_subplot(gs[2, 2:])
        if not scores_df.empty:
            scatter = ax6.scatter(scores_df['Diversity'], scores_df['Relevance'], 
                                 c=scores_df['Overall'], s=60, alpha=0.6, cmap='viridis')
            ax6.set_xlabel('Diversity Score')
            ax6.set_ylabel('Relevance Score')
            ax6.set_title('Diversity vs Relevance (colored by Overall Score)', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax6)
        
        # 7. Summary statistics
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create summary text
        summary_text = "📊 RECOMMENDATION DASHBOARD SUMMARY\n\n"
        
        if not scores_df.empty:
            summary_text += f"🎯 Average Overall Score: {scores_df['Overall'].mean():.3f} ± {scores_df['Overall'].std():.3f}\n"
            summary_text += f"🌈 Average Diversity: {scores_df['Diversity'].mean():.3f}\n"
            summary_text += f"✨ Average Novelty: {scores_df['Novelty'].mean():.3f}\n"
            summary_text += f"🎵 Average Relevance: {scores_df['Relevance'].mean():.3f}\n\n"
        
        summary_text += f"📀 Total Recommendations Analyzed: {len(recommended_tracks)}\n"
        summary_text += f"🎼 Number of Unique Genres: {recommended_tracks['genre'].nunique()}\n"
        summary_text += f"🎤 Number of Unique Artists: {recommended_tracks['artist_name'].nunique()}\n"
        
        if 'popularity' in recommended_tracks.columns:
            summary_text += f"⭐ Average Popularity: {recommended_tracks['popularity'].mean():.1f}/100\n"
        
        if user_history_tracks is not None:
            summary_text += f"📚 User History Size: {len(user_history_tracks)} tracks\n"
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('Music Recommendation System Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

if __name__ == "__main__":
    # Example usage
    from data_loader import MusicDataLoader
    from music_rag_system import MusicRAGSystem
    from user_simulator import UserSimulator
    from evaluator import MusicRecommendationEvaluator
    
    # Load data and setup system
    loader = MusicDataLoader()
    tracks_df = loader.load_spotify_dataset()
    
    rag_system = MusicRAGSystem()
    rag_system.setup_vector_store(tracks_df)
    
    simulator = UserSimulator(tracks_df)
    users = simulator.generate_user_histories(n_users=10)
    
    evaluator = MusicRecommendationEvaluator(tracks_df)
    visualizer = MusicRecommendationVisualizer(tracks_df)
    
    # Generate evaluations
    evaluations = []
    for user in users[:5]:
        preference_text = simulator.get_user_preference_text(user)
        recommended_ids, _ = rag_system.get_recommendations(
            user_listening_history=user['listening_history'],
            preference_text=preference_text,
            n_recommendations=10
        )
        
        evaluation = evaluator.evaluate_all(
            recommended_ids=recommended_ids,
            user_history=user['listening_history'],
            preference_text=preference_text
        )
        evaluations.append(evaluation)
    
    # Create visualizations
    recommended_tracks = rag_system.get_track_info(recommended_ids)
    user_history_tracks = rag_system.get_track_info(users[0]['listening_history'])
    
    print("🎨 Creating visualizations...")
    visualizer.plot_evaluation_scores(evaluations)
    visualizer.plot_genre_analysis(recommended_tracks, user_history_tracks)
    visualizer.create_dashboard(evaluations, recommended_tracks, user_history_tracks) 