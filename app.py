"""
Music Dataset Explorer - Streamlit Application
Interactive dashboard for exploring and analyzing music data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🎵 Music Dataset Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
    .stSelectbox > div > div {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the music dataset"""
    # List of possible dataset file names to try
    dataset_files = [
        'final_dataset.csv',
        'final_dataset_enhanced.csv',
        'final_dataset_backup.csv'
    ]
    
    for filename in dataset_files:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                st.success(f"✅ Successfully loaded dataset: {filename}")
                st.info(f"📊 Dataset contains {len(df):,} tracks from {df['track_artist'].nunique():,} artists")
                return df
            except Exception as e:
                st.warning(f"⚠️ Could not load {filename}: {str(e)}")
                continue
    
    # If no files work, show detailed error
    st.error("❌ No dataset file found. Available files:")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        for file in csv_files:
            st.write(f"  📄 {file}")
        st.error("Please ensure one of the dataset files is named 'final_dataset.csv'")
    else:
        st.error("No CSV files found in the current directory.")
    
    return None

@st.cache_data
def prepare_features(df):
    """Prepare audio features for analysis"""
    # Updated to match actual dataset columns
    audio_features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 
                     'valence', 'speechiness', 'instrumentalness', 'acousticness']
    
    # Check which features actually exist in the dataset
    available_features = [col for col in audio_features if col in df.columns]
    
    if not available_features:
        st.error("No audio features found in the dataset!")
        return None, None, []
    
    feature_data = df[available_features].copy()
    
    # Handle missing values
    feature_data = feature_data.fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    
    return feature_data, feature_data_scaled, available_features

def main():
    st.markdown('<h1 class="main-header">🎵 Music Dataset Explorer</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Dataset Overview", "🎨 Audio Features Analysis", "🔍 Music Discovery", 
         "📈 Advanced Analytics", "🎯 Recommendations"]
    )
    
    if page == "📊 Dataset Overview":
        show_dataset_overview(df)
    elif page == "🎨 Audio Features Analysis":
        show_audio_features_analysis(df)
    elif page == "🔍 Music Discovery":
        show_music_discovery(df)
    elif page == "📈 Advanced Analytics":
        show_advanced_analytics(df)
    elif page == "🎯 Recommendations":
        show_recommendations(df)

def show_dataset_overview(df):
    """Display dataset overview and basic statistics"""
    st.header("📊 Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tracks", f"{len(df):,}")
    with col2:
        st.metric("Unique Artists", f"{df['track_artist'].nunique():,}")
    with col3:
        st.metric("Unique Albums", f"{df['track_album_name'].nunique():,}")
    with col4:
        st.metric("Genres", f"{df['playlist_genre'].nunique()}")
    
    # Dataset sample
    st.subheader("📋 Dataset Sample")
    display_columns = ['track_name', 'track_artist', 'track_album_name', 'playlist_genre', 
                      'track_popularity', 'energy', 'danceability', 'valence']
    st.dataframe(df[display_columns].head(10), use_container_width=True)
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Genre Distribution")
        genre_counts = df['playlist_genre'].value_counts()
        fig = px.pie(values=genre_counts.values, names=genre_counts.index,
                    title="Distribution of Music Genres")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Popularity Distribution")
        fig = px.histogram(df, x='track_popularity', nbins=30,
                          title="Track Popularity Distribution")
        fig.update_layout(xaxis_title="Popularity Score", yaxis_title="Number of Tracks")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top artists and tracks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎤 Top 10 Artists")
        top_artists = df['track_artist'].value_counts().head(10)
        fig = px.bar(x=top_artists.values, y=top_artists.index, orientation='h',
                    title="Most Frequent Artists in Dataset")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Most Popular Tracks")
        popular_tracks = df.nlargest(10, 'track_popularity')[['track_name', 'track_artist', 'track_popularity']]
        st.dataframe(popular_tracks, use_container_width=True)

def show_audio_features_analysis(df):
    """Analyze and visualize audio features"""
    st.header("🎨 Audio Features Analysis")
    
    # Audio features selection - dynamically check what's available
    all_audio_features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 
                     'valence', 'speechiness', 'instrumentalness', 'acousticness']
    audio_features = [col for col in all_audio_features if col in df.columns]
    
    if not audio_features:
        st.error("No audio features found in the dataset!")
        return
    
    # Feature correlation heatmap
    st.subheader("🔥 Audio Features Correlation")
    feature_corr = df[audio_features].corr()
    
    fig = px.imshow(feature_corr, text_auto=True, aspect="auto",
                   title="Correlation Matrix of Audio Features")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions by genre
    st.subheader("📊 Feature Distributions by Genre")
    
    selected_feature = st.selectbox("Select an audio feature:", audio_features)
    
    fig = px.box(df, x='playlist_genre', y=selected_feature,
                title=f"{selected_feature.title()} Distribution by Genre")
    fig.update_layout(xaxis_title="Genre", yaxis_title=selected_feature.title())
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-feature analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Feature Scatter Plot")
        feature_x = st.selectbox("X-axis feature:", audio_features, index=0, key="scatter_x")
        feature_y = st.selectbox("Y-axis feature:", audio_features, index=1, key="scatter_y")
        
        fig = px.scatter(df, x=feature_x, y=feature_y, color='playlist_genre',
                        hover_data=['track_name', 'track_artist'],
                        title=f"{feature_x.title()} vs {feature_y.title()}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Feature Trends")
        selected_features = st.multiselect("Select features to compare:", 
                                         audio_features, default=audio_features[:3])
        
        if selected_features:
            avg_by_genre = df.groupby('playlist_genre')[selected_features].mean()
            fig = px.line_polar(avg_by_genre.T, r=avg_by_genre.columns, 
                               theta=avg_by_genre.index,
                               title="Average Feature Values by Genre")
            st.plotly_chart(fig, use_container_width=True)

def show_music_discovery(df):
    """Music discovery and search functionality"""
    st.header("🔍 Music Discovery")
    
    # Search and filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_artist = st.text_input("🎤 Search by Artist:")
        selected_genre = st.selectbox("🎭 Filter by Genre:", 
                                     ['All'] + list(df['playlist_genre'].unique()))
    
    with col2:
        min_popularity = st.slider("⭐ Minimum Popularity:", 0, 100, 0)
        year_range = st.slider("📅 Release Year Range:", 
                              1950, 2024, (2000, 2024))
    
    with col3:
        energy_range = st.slider("⚡ Energy Level:", 0.0, 1.0, (0.0, 1.0))
        danceability_range = st.slider("💃 Danceability:", 0.0, 1.0, (0.0, 1.0))
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_artist:
        filtered_df = filtered_df[filtered_df['track_artist'].str.contains(search_artist, case=False, na=False)]
    
    if selected_genre != 'All':
        filtered_df = filtered_df[filtered_df['playlist_genre'] == selected_genre]
    
    filtered_df = filtered_df[filtered_df['track_popularity'] >= min_popularity]
    filtered_df = filtered_df[
        (filtered_df['energy'] >= energy_range[0]) & 
        (filtered_df['energy'] <= energy_range[1])
    ]
    filtered_df = filtered_df[
        (filtered_df['danceability'] >= danceability_range[0]) & 
        (filtered_df['danceability'] <= danceability_range[1])
    ]
    
    st.subheader(f"🎵 Found {len(filtered_df)} tracks")
    
    if len(filtered_df) > 0:
        # Display results
        display_cols = ['track_name', 'track_artist', 'track_album_name', 'playlist_genre',
                       'track_popularity', 'energy', 'danceability', 'valence']
        
        # Sort options
        sort_by = st.selectbox("Sort by:", ['track_popularity', 'energy', 'danceability', 'valence'], 
                              index=0)
        sort_order = st.radio("Sort order:", ['Descending', 'Ascending'])
        
        sorted_df = filtered_df.sort_values(sort_by, ascending=(sort_order == 'Ascending'))
        
        st.dataframe(sorted_df[display_cols].head(20), use_container_width=True)
        
        # Audio features visualization for filtered results
        if len(filtered_df) > 1:
            st.subheader("📊 Audio Features of Filtered Results")
            audio_features = ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
            avg_features = filtered_df[audio_features].mean()
            
            fig = go.Figure(data=go.Scatterpolar(
                r=avg_features.values,
                theta=avg_features.index,
                fill='toself',
                name='Average Features'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Average Audio Features Profile"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tracks found matching your criteria. Try adjusting the filters.")

def show_advanced_analytics(df):
    """Advanced analytics including clustering and PCA"""
    st.header("📈 Advanced Analytics")
    
    # Prepare data for analysis
    feature_data, feature_data_scaled, audio_features = prepare_features(df)
    
    # Check if feature preparation was successful
    if feature_data is None:
        st.error("Unable to prepare audio features for analysis.")
        return
    
    tab1, tab2, tab3 = st.tabs(["🎯 Clustering Analysis", "📊 PCA Analysis", "🔬 Statistical Insights"])
    
    with tab1:
        st.subheader("K-Means Clustering of Music Tracks")
        
        # Clustering parameters
        n_clusters = st.slider("Number of clusters:", 2, 10, 5)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_data_scaled)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        title="Cluster Size Distribution")
            fig.update_layout(xaxis_title="Cluster", yaxis_title="Number of Tracks")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Genre distribution by cluster
            cluster_genre = df_clustered.groupby(['cluster', 'playlist_genre']).size().unstack(fill_value=0)
            fig = px.imshow(cluster_genre.values, 
                           x=cluster_genre.columns, 
                           y=[f"Cluster {i}" for i in cluster_genre.index],
                           title="Genre Distribution by Cluster")
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        st.subheader("🎨 Cluster Characteristics")
        cluster_means = df_clustered.groupby('cluster')[audio_features].mean()
        st.dataframe(cluster_means.round(3), use_container_width=True)
        
        # Sample tracks from each cluster
        st.subheader("🎵 Sample Tracks by Cluster")
        selected_cluster = st.selectbox("Select cluster to explore:", range(n_clusters))
        cluster_tracks = df_clustered[df_clustered['cluster'] == selected_cluster]
        sample_tracks = cluster_tracks.sample(min(10, len(cluster_tracks)))[
            ['track_name', 'track_artist', 'playlist_genre', 'track_popularity']
        ]
        st.dataframe(sample_tracks, use_container_width=True)
    
    with tab2:
        st.subheader("Principal Component Analysis (PCA)")
        
        # Perform PCA
        pca = PCA()
        pca_data = pca.fit_transform(feature_data_scaled)
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scree plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(explained_var)+1)), 
                                   y=explained_var, mode='lines+markers',
                                   name='Individual'))
            fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_var)+1)), 
                                   y=cumulative_var, mode='lines+markers',
                                   name='Cumulative'))
            fig.update_layout(title="PCA Explained Variance",
                            xaxis_title="Principal Component",
                            yaxis_title="Explained Variance Ratio")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # PCA scatter plot
            pca_df = pd.DataFrame(pca_data[:, :2], columns=['PC1', 'PC2'])
            pca_df['genre'] = df['playlist_genre']
            
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='genre',
                           title="PCA Projection (First 2 Components)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature contributions
        st.subheader("🎯 Feature Contributions to Principal Components")
        n_components = st.slider("Number of components to show:", 1, min(5, len(audio_features)), 2)
        
        components_df = pd.DataFrame(
            pca.components_[:n_components],
            columns=audio_features,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        
        fig = px.imshow(components_df.values, 
                       x=components_df.columns, 
                       y=components_df.index,
                       color_continuous_scale='RdBu',
                       title="Feature Loadings for Principal Components")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔬 Statistical Insights")
        
        # Feature statistics by genre
        st.write("**Audio Features Statistics by Genre**")
        genre_stats = df.groupby('playlist_genre')[audio_features].agg(['mean', 'std']).round(3)
        st.dataframe(genre_stats, use_container_width=True)
        
        # Popularity insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Popularity by Genre**")
            pop_by_genre = df.groupby('playlist_genre')['track_popularity'].agg(['mean', 'median', 'std']).round(2)
            st.dataframe(pop_by_genre)
        
        with col2:
            st.write("**Top Features Correlated with Popularity**")
            pop_corr = df[audio_features + ['track_popularity']].corr()['track_popularity'].drop('track_popularity')
            pop_corr_sorted = pop_corr.abs().sort_values(ascending=False)
            st.dataframe(pop_corr_sorted.round(3))

def show_recommendations(df):
    """Music recommendation system"""
    st.header("🎯 Music Recommendations")
    
    st.info("🎵 Get personalized music recommendations based on your preferences!")
    
    # Recommendation methods
    method = st.radio("Choose recommendation method:", 
                     ["🎨 Audio Features Based", "🎤 Artist Similarity", "🎭 Genre Exploration"])
    
    if method == "🎨 Audio Features Based":
        st.subheader("Customize Your Audio Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_energy = st.slider("⚡ Energy (0=Calm, 1=High Energy):", 0.0, 1.0, 0.5)
            target_dance = st.slider("💃 Danceability (0=Not Danceable, 1=Very Danceable):", 0.0, 1.0, 0.5)
            target_valence = st.slider("😊 Valence (0=Sad, 1=Happy):", 0.0, 1.0, 0.5)
        
        with col2:
            target_acoustic = st.slider("🎸 Acousticness (0=Electronic, 1=Acoustic):", 0.0, 1.0, 0.5)
            target_instrumental = st.slider("🎼 Instrumentalness (0=Vocal, 1=Instrumental):", 0.0, 1.0, 0.5)
            min_popularity = st.slider("⭐ Minimum Popularity:", 0, 100, 20)
        
        if st.button("🎵 Get Recommendations"):
            # Create target profile
            target_features = np.array([target_energy, target_dance, target_valence, 
                                      target_acoustic, target_instrumental])
            
            # Calculate similarity
            audio_cols = ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
            
            # Handle NaN values in track features
            track_features_clean = df[audio_cols].fillna(0)
            track_features = track_features_clean.values
            
            # Normalize features
            scaler = StandardScaler()
            track_features_scaled = scaler.fit_transform(track_features)
            target_scaled = scaler.transform(target_features.reshape(1, -1))
            
            # Calculate cosine similarity
            similarities = cosine_similarity(target_scaled, track_features_scaled)[0]
            
            # Filter by popularity and get top recommendations
            popular_tracks = df[df['track_popularity'] >= min_popularity].copy()
            if len(popular_tracks) > 0:
                popular_indices = popular_tracks.index
                popular_similarities = similarities[popular_indices]
                
                # Get top 10 recommendations
                top_indices = popular_indices[np.argsort(popular_similarities)[-10:]][::-1]
                recommendations = df.loc[top_indices]
                
                st.subheader("🎵 Your Personalized Recommendations")
                display_cols = ['track_name', 'track_artist', 'playlist_genre', 'track_popularity'] + audio_cols
                st.dataframe(recommendations[display_cols], use_container_width=True)
                
                # Show recommendation profile
                rec_profile = recommendations[audio_cols].mean()
                target_profile = pd.Series(target_features, index=audio_cols)
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=target_profile.values,
                    theta=target_profile.index,
                    fill='toself',
                    name='Your Preferences',
                    opacity=0.6
                ))
                fig.add_trace(go.Scatterpolar(
                    r=rec_profile.values,
                    theta=rec_profile.index,
                    fill='toself',
                    name='Recommendations Profile',
                    opacity=0.6
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Your Preferences vs Recommendations Profile"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No tracks found with the specified minimum popularity. Try lowering the threshold.")
    
    elif method == "🎤 Artist Similarity":
        st.subheader("Discover Similar Artists")
        
        # Artist selection
        artists = sorted(df['track_artist'].unique())
        selected_artist = st.selectbox("Choose an artist you like:", artists)
        
        if st.button("🔍 Find Similar Artists"):
            # Get artist's audio profile
            artist_tracks = df[df['track_artist'] == selected_artist]
            if len(artist_tracks) > 0:
                audio_cols = ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
                
                # Handle NaN values by filling with column means or 0
                artist_tracks_clean = artist_tracks[audio_cols].fillna(0)
                artist_profile = artist_tracks_clean.mean()
                
                # Check if artist profile has valid values
                if artist_profile.isna().any():
                    st.error(f"Unable to calculate audio profile for {selected_artist}. Missing audio data.")
                    return
                
                # Get selected artist's most common genre
                selected_artist_genres = artist_tracks['playlist_genre'].value_counts()
                primary_genre = selected_artist_genres.index[0] if len(selected_artist_genres) > 0 else None
                
                st.info(f"🎭 Primary genre for {selected_artist}: {primary_genre}")
                
                # Add genre preference option
                genre_weight = st.slider("🎭 Genre matching importance (0=ignore genre, 1=prioritize same genre):", 0.0, 1.0, 0.3)
                
                # Calculate similarity with all other artists
                other_artists = df[df['track_artist'] != selected_artist]
                artist_similarities = []
                
                for artist in other_artists['track_artist'].unique():
                    artist_data = other_artists[other_artists['track_artist'] == artist]
                    if len(artist_data) > 0:
                        # Handle NaN values for other artists too
                        artist_data_clean = artist_data[audio_cols].fillna(0)
                        other_profile = artist_data_clean.mean()
                        
                        # Skip artists with invalid profiles
                        if not other_profile.isna().any():
                            try:
                                # Calculate audio similarity
                                audio_similarity = cosine_similarity([artist_profile.values], [other_profile.values])[0][0]
                                
                                # Calculate genre similarity
                                other_artist_genres = artist_data['playlist_genre'].value_counts()
                                other_primary_genre = other_artist_genres.index[0] if len(other_artist_genres) > 0 else None
                                
                                # Genre bonus: 1.0 if same genre, 0.5 if different
                                genre_bonus = 1.0 if (primary_genre and other_primary_genre == primary_genre) else 0.5
                                
                                # Combined similarity with genre weighting
                                combined_similarity = (1 - genre_weight) * audio_similarity + genre_weight * genre_bonus
                                
                                # Check if similarity is valid (not NaN)
                                if not np.isnan(combined_similarity):
                                    artist_similarities.append({
                                        'artist': artist,
                                        'audio_similarity': audio_similarity,
                                        'genre_bonus': genre_bonus,
                                        'combined_similarity': combined_similarity,
                                        'primary_genre': other_primary_genre,
                                        'track_count': len(artist_data),
                                        'avg_popularity': artist_data['track_popularity'].mean()
                                    })
                            except ValueError:
                                # Skip this artist if similarity calculation fails
                                continue
                
                # Sort by combined similarity and show top 10
                similar_artists = sorted(artist_similarities, key=lambda x: x['combined_similarity'], reverse=True)[:10]
                
                st.subheader(f"🎤 Artists Similar to {selected_artist}")
                
                # Create display dataframe with relevant columns
                display_artists = []
                for artist in similar_artists:
                    display_artists.append({
                        'Artist': artist['artist'],
                        'Primary Genre': artist['primary_genre'],
                        'Combined Score': round(artist['combined_similarity'], 3),
                        'Audio Score': round(artist['audio_similarity'], 3),
                        'Genre Match': '✅ Same Genre' if artist['genre_bonus'] == 1.0 else '⚠️ Different Genre',
                        'Track Count': artist['track_count'],
                        'Avg Popularity': round(artist['avg_popularity'], 1)
                    })
                
                similar_df = pd.DataFrame(display_artists)
                st.dataframe(similar_df, use_container_width=True)
                
                # Add explanation
                st.info(f"""
                🎯 **How similarity is calculated:**
                - **Audio Score**: Based on musical features (energy, danceability, etc.)
                - **Genre Match**: Bonus for artists in the same genre as {selected_artist}
                - **Combined Score**: Weighted combination based on your genre importance setting
                
                💡 **Tip**: Increase 'Genre matching importance' to get more culturally similar artists!
                """)
                
                # Show some tracks from top similar artists
                if len(similar_artists) > 0:
                    top_similar = similar_artists[0]['artist']
                    similar_tracks = df[df['track_artist'] == top_similar].head(5)
                    
                    st.subheader(f"🎵 Sample tracks from {top_similar}")
                    display_cols = ['track_name', 'track_album_name', 'track_popularity']
                    st.dataframe(similar_tracks[display_cols], use_container_width=True)
    
    elif method == "🎭 Genre Exploration":
        st.subheader("Explore Music by Genre")
        
        # Genre selection
        genres = sorted(df['playlist_genre'].unique())
        selected_genres = st.multiselect("Select genres to explore:", genres, default=genres[:2])
        
        if selected_genres and st.button("🎵 Get Genre Recommendations"):
            # Get tracks from selected genres
            genre_tracks = df[df['playlist_genre'].isin(selected_genres)]
            
            # Sort by popularity and get top tracks
            top_tracks = genre_tracks.nlargest(20, 'track_popularity')
            
            st.subheader(f"🎵 Top Tracks from Selected Genres")
            display_cols = ['track_name', 'track_artist', 'playlist_genre', 'track_popularity']
            st.dataframe(top_tracks[display_cols], use_container_width=True)
            
            # Show genre characteristics
            st.subheader("🎨 Genre Audio Characteristics")
            audio_cols = ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
            
            # Handle NaN values in genre tracks
            genre_tracks_clean = genre_tracks[audio_cols].fillna(0)
            genre_tracks_for_profile = genre_tracks.copy()
            genre_tracks_for_profile[audio_cols] = genre_tracks_clean
            genre_profiles = genre_tracks_for_profile.groupby('playlist_genre')[audio_cols].mean()
            
            fig = go.Figure()
            for genre in selected_genres:
                if genre in genre_profiles.index:
                    profile = genre_profiles.loc[genre]
                    fig.add_trace(go.Scatterpolar(
                        r=profile.values,
                        theta=profile.index,
                        fill='toself',
                        name=genre,
                        opacity=0.6
                    ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Audio Characteristics by Genre"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 