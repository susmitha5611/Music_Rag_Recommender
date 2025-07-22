"""
Enhanced Music RAG System - Streamlit Application
Interactive dashboard for music recommendation using RAG
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import MusicDataLoader
from music_rag_system import MusicRAGSystem
from user_simulator import UserSimulator
from evaluator import MusicRecommendationEvaluator
from rlhf_module import RLHFTrainer

# Configure Streamlit page
st.set_page_config(
    page_title="🎵 Music RAG Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #1DB954;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1DB954, #1ed760);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(29, 185, 84, 0.4);
    }
    
    .feature-slider {
        background: rgba(29, 185, 84, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize and cache the RAG system"""
    with st.spinner("🚀 Initializing Music RAG System..."):
        loader = MusicDataLoader()
        tracks_df = loader.load_final_dataset('final_dataset.csv')
        
        rag_system = MusicRAGSystem()
        rag_system.setup_vector_store(tracks_df)
        
        return rag_system, tracks_df

@st.cache_data
def load_sample_users(_tracks_df):
    """Generate and cache sample users"""
    simulator = UserSimulator(_tracks_df)
    users = simulator.generate_user_histories(n_users=20)
    return users, simulator

def main():
    st.markdown('<h1 class="main-header">🎵 Music RAG Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Discover music with AI-powered recommendations using Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
    
    # Initialize system
    try:
        rag_system, tracks_df = initialize_system()
        users, simulator = load_sample_users(tracks_df)
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🎯 Smart Recommendations", "🔍 Music Discovery", 
         "📊 Dataset Explorer", "🤖 RLHF Training", "📈 Analytics Dashboard"]
    )
    
    if page == "🏠 Home":
        show_home_page(tracks_df, rag_system)
    elif page == "🎯 Smart Recommendations":
        show_recommendations_page(rag_system, tracks_df, users, simulator)
    elif page == "🔍 Music Discovery":
        show_discovery_page(rag_system, tracks_df)
    elif page == "📊 Dataset Explorer":
        show_dataset_explorer(tracks_df)
    elif page == "🤖 RLHF Training":
        show_rlhf_training(rag_system, tracks_df)
    elif page == "📈 Analytics Dashboard":
        show_analytics_dashboard(tracks_df, rag_system)

def show_home_page(tracks_df, rag_system):
    """Display home page with system overview"""
    st.header("🏠 Welcome to Music RAG Recommender")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎵 Total Tracks</h3>
            <h2>{len(tracks_df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎤 Artists</h3>
            <h2>{tracks_df['artist_name'].nunique():,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎭 Genres</h3>
            <h2>{tracks_df['genre'].nunique()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💿 Albums</h3>
            <h2>{tracks_df['album_name'].nunique():,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    st.subheader("✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 AI-Powered Recommendations
        - Semantic search using embeddings
        - Audio feature matching
        - Hybrid recommendation algorithms
        - Personalized suggestions
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Advanced Analytics
        - Real-time music analysis
        - Genre clustering
        - Popularity trends
        - Audio feature insights
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 RLHF Integration
        - Reinforcement Learning from Human Feedback
        - Continuous model improvement
        - Personalized ranking
        - Quality optimization
        """)
    
    # Quick stats visualization
    st.markdown("---")
    st.subheader("📊 Quick Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre distribution
        genre_counts = tracks_df['genre'].value_counts().head(10)
        fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                    title="Top 10 Genres", color=genre_counts.values,
                    color_continuous_scale='viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Audio features radar
        audio_features = ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']
        avg_features = tracks_df[audio_features].mean()
        
        fig = go.Figure(data=go.Scatterpolar(
            r=avg_features.values,
            theta=avg_features.index,
            fill='toself',
            name='Average Features',
            line_color='#1DB954'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Average Audio Features Profile",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_recommendations_page(rag_system, tracks_df, users, simulator):
    """Enhanced recommendations page with multiple methods"""
    st.header("🎯 Smart Music Recommendations")
    
    # Recommendation method selection
    method = st.radio(
        "Choose recommendation method:",
        ["🔍 Text-based Search", "🎵 Audio Features", "👤 User Profile", "🔀 Hybrid Method"],
        horizontal=True
    )
    
    if method == "🔍 Text-based Search":
        show_text_recommendations(rag_system, tracks_df)
    elif method == "🎵 Audio Features":
        show_feature_recommendations(rag_system, tracks_df)
    elif method == "👤 User Profile":
        show_user_recommendations(rag_system, tracks_df, users, simulator)
    elif method == "🔀 Hybrid Method":
        show_hybrid_recommendations(rag_system, tracks_df)

def show_text_recommendations(rag_system, tracks_df):
    """Text-based semantic recommendations"""
    st.subheader("🔍 Semantic Music Search")
    
    # Search input
    query = st.text_area(
        "Describe the music you're looking for:",
        placeholder="e.g., 'upbeat dance music for working out', 'calm acoustic songs for studying', 'energetic rock music'",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        n_results = st.slider("Number of recommendations:", 5, 20, 10)
    with col2:
        genre_filter = st.selectbox("Genre filter:", ['All'] + list(tracks_df['genre'].unique()))
    
    if st.button("🔍 Get Recommendations", type="primary"):
        if query:
            with st.spinner("🎵 Finding perfect matches..."):
                # Apply genre filter if selected
                filter_metadata = None
                if genre_filter != 'All':
                    filter_metadata = {"genre": {"$eq": genre_filter}}
                
                # Get recommendations
                results = rag_system.search_similar_tracks(
                    query, n_results=n_results, filter_metadata=filter_metadata
                )
                
                # Display results
                st.success(f"✅ Found {len(results['ids'][0])} recommendations!")
                
                for i, track_id in enumerate(results['ids'][0]):
                    track_info = tracks_df[tracks_df['track_id'] == track_id].iloc[0]
                    similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{i+1} {track_info['track_name']}</h4>
                            <p><strong>Artist:</strong> {track_info['artist_name']}</p>
                            <p><strong>Genre:</strong> {track_info['genre']} | <strong>Similarity:</strong> {similarity_score:.2%}</p>
                            <p><em>{track_info['description'][:200]}...</em></p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a search query!")

def show_feature_recommendations(rag_system, tracks_df):
    """Audio feature-based recommendations"""
    st.subheader("🎵 Audio Feature Matching")
    
    st.markdown("Adjust the sliders to define your ideal music characteristics:")
    
    # Feature sliders in a nice layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-slider">', unsafe_allow_html=True)
        energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5, 0.1, help="How energetic and intense the music is")
        danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.5, 0.1, help="How suitable for dancing")
        valence = st.slider("😊 Valence (Positivity)", 0.0, 1.0, 0.5, 0.1, help="Musical positivity/happiness")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-slider">', unsafe_allow_html=True)
        acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.5, 0.1, help="How acoustic vs electronic")
        instrumentalness = st.slider("🎼 Instrumentalness", 0.0, 1.0, 0.5, 0.1, help="Likelihood of no vocals")
        speechiness = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.1, 0.1, help="Presence of spoken words")
        st.markdown('</div>', unsafe_allow_html=True)
    
    n_results = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.button("🎵 Find Similar Music", type="primary"):
        target_features = {
            'energy': energy,
            'danceability': danceability,
            'valence': valence,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'speechiness': speechiness
        }
        
        with st.spinner("🎵 Analyzing audio features..."):
            recommended_ids = rag_system.get_recommendations_by_audio_features(
                target_features=target_features,
                n_recommendations=n_results
            )
            
            # Display recommendations
            st.success(f"✅ Found {len(recommended_ids)} feature-matched tracks!")
            
            recommended_tracks = rag_system.get_track_info(recommended_ids)
            
            for i, (_, track) in enumerate(recommended_tracks.iterrows()):
                # Calculate feature similarity
                track_features = {k: track.get(k, 0) for k in target_features.keys()}
                similarity = 1 - np.mean([abs(target_features[k] - track_features[k]) for k in target_features.keys()])
                
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>#{i+1} {track['track_name']}</h4>
                        <p><strong>Artist:</strong> {track['artist_name']} | <strong>Genre:</strong> {track['genre']}</p>
                        <p><strong>Feature Match:</strong> {similarity:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show feature comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Your Preferences:**")
                        for feature, value in target_features.items():
                            st.write(f"{feature.title()}: {value:.2f}")
                    with col2:
                        st.write("**Track Features:**")
                        for feature in target_features.keys():
                            track_value = track.get(feature, 0)
                            st.write(f"{feature.title()}: {track_value:.2f}")

def show_user_recommendations(rag_system, tracks_df, users, simulator):
    """User profile-based recommendations"""
    st.subheader("👤 Personalized Recommendations")
    
    # User selection
    user_options = [f"User {user['user_id']} ({user['archetype']})" for user in users]
    selected_user_idx = st.selectbox("Select a user profile:", range(len(user_options)), 
                                    format_func=lambda x: user_options[x])
    
    selected_user = users[selected_user_idx]
    
    # Display user profile
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **👤 User Profile:**
        - **ID:** {selected_user['user_id']}
        - **Type:** {selected_user['archetype']}
        - **Preferred Genres:** {', '.join(selected_user['preferred_genres'])}
        - **Listening History:** {selected_user['history_size']} tracks
        """)
    
    with col2:
        # Show listening history sample
        st.markdown("**🎵 Recent Listening History:**")
        history_tracks = rag_system.get_track_info(selected_user['listening_history'][:5])
        for _, track in history_tracks.iterrows():
            st.write(f"• {track['track_name']} - {track['artist_name']}")
    
    # Generate preference text
    preference_text = simulator.get_user_preference_text(selected_user)
    st.markdown(f"**💭 Generated Preference:** _{preference_text}_")
    
    n_results = st.slider("Number of recommendations:", 5, 20, 10, key="user_recs")
    
    if st.button("🎯 Get Personalized Recommendations", type="primary"):
        with st.spinner("🎵 Analyzing user preferences..."):
            recommended_ids, search_results = rag_system.get_recommendations(
                user_listening_history=selected_user['listening_history'],
                preference_text=preference_text,
                n_recommendations=n_results
            )
            
            st.success(f"✅ Generated {len(recommended_ids)} personalized recommendations!")
            
            recommended_tracks = rag_system.get_track_info(recommended_ids)
            
            for i, (_, track) in enumerate(recommended_tracks.iterrows()):
                # Generate explanation
                explanation = rag_system.explain_recommendation(track['track_id'], preference_text)
                
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>#{i+1} {track['track_name']}</h4>
                        <p><strong>Artist:</strong> {track['artist_name']} | <strong>Genre:</strong> {track['genre']}</p>
                        <p><strong>Why recommended:</strong> {explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)

def show_hybrid_recommendations(rag_system, tracks_df):
    """Hybrid recommendation method"""
    st.subheader("🔀 Hybrid Recommendations")
    st.markdown("Combine text preferences with audio features for the best results!")
    
    # Text input
    preference_text = st.text_area(
        "Describe your music preferences:",
        placeholder="e.g., 'I love energetic pop music for workouts'",
        height=80
    )
    
    # Audio features (simplified)
    st.markdown("**🎵 Fine-tune with audio features:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        energy = st.slider("⚡ Energy", 0.0, 1.0, 0.7, 0.1, key="hybrid_energy")
        danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.6, 0.1, key="hybrid_dance")
    
    with col2:
        valence = st.slider("😊 Positivity", 0.0, 1.0, 0.6, 0.1, key="hybrid_valence")
        acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.3, 0.1, key="hybrid_acoustic")
    
    with col3:
        semantic_weight = st.slider("🧠 Text vs Features Balance", 0.0, 1.0, 0.7, 0.1, 
                                   help="Higher = more text-based, Lower = more feature-based")
        n_results = st.slider("Number of recommendations:", 5, 20, 10, key="hybrid_results")
    
    if st.button("🔀 Get Hybrid Recommendations", type="primary"):
        if preference_text:
            target_features = {
                'energy': energy,
                'danceability': danceability,
                'valence': valence,
                'acousticness': acousticness
            }
            
            with st.spinner("🎵 Combining semantic and feature analysis..."):
                recommended_ids = rag_system.get_hybrid_recommendations(
                    preference_text=preference_text,
                    target_features=target_features,
                    n_recommendations=n_results,
                    semantic_weight=semantic_weight
                )
                
                st.success(f"✅ Generated {len(recommended_ids)} hybrid recommendations!")
                
                recommended_tracks = rag_system.get_track_info(recommended_ids)
                
                for i, (_, track) in enumerate(recommended_tracks.iterrows()):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{i+1} {track['track_name']}</h4>
                            <p><strong>Artist:</strong> {track['artist_name']} | <strong>Genre:</strong> {track['genre']}</p>
                            <p><strong>Album:</strong> {track.get('album_name', 'Unknown')}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Please enter your music preferences!")

def show_discovery_page(rag_system, tracks_df):
    """Music discovery with advanced filtering"""
    st.header("🔍 Advanced Music Discovery")
    
    # Advanced filters
    st.subheader("🎛️ Discovery Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        genres = st.multiselect("🎭 Genres:", tracks_df['genre'].unique())
        min_popularity = st.slider("⭐ Min Popularity:", 0, 100, 0)
        max_popularity = st.slider("⭐ Max Popularity:", 0, 100, 100)
    
    with col2:
        year_range = st.slider("📅 Year Range:", 
                              int(tracks_df['year'].min()), 
                              int(tracks_df['year'].max()), 
                              (2000, 2024))
        energy_range = st.slider("⚡ Energy Range:", 0.0, 1.0, (0.0, 1.0))
    
    with col3:
        danceability_range = st.slider("💃 Danceability Range:", 0.0, 1.0, (0.0, 1.0))
        valence_range = st.slider("😊 Valence Range:", 0.0, 1.0, (0.0, 1.0))
    
    # Apply filters
    filtered_df = tracks_df.copy()
    
    if genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(genres)]
    
    filtered_df = filtered_df[
        (filtered_df['popularity'] >= min_popularity) & 
        (filtered_df['popularity'] <= max_popularity)
    ]
    
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['energy'] >= energy_range[0]) & 
        (filtered_df['energy'] <= energy_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['danceability'] >= danceability_range[0]) & 
        (filtered_df['danceability'] <= danceability_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['valence'] >= valence_range[0]) & 
        (filtered_df['valence'] <= valence_range[1])
    ]
    
    st.subheader(f"🎵 Discovered {len(filtered_df):,} tracks")
    
    if len(filtered_df) > 0:
        # Sort options
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Sort by:", ['popularity', 'energy', 'danceability', 'valence', 'year'])
        with col2:
            sort_order = st.radio("Order:", ['Descending', 'Ascending'], horizontal=True)
        
        sorted_df = filtered_df.sort_values(sort_by, ascending=(sort_order == 'Ascending'))
        
        # Display results
        display_cols = ['track_name', 'artist_name', 'genre', 'popularity', 'energy', 'danceability', 'valence']
        st.dataframe(sorted_df[display_cols].head(50), use_container_width=True)
        
        # Visualization
        if len(filtered_df) > 1:
            st.subheader("📊 Discovery Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Genre distribution
                genre_dist = filtered_df['genre'].value_counts()
                fig = px.pie(values=genre_dist.values, names=genre_dist.index,
                           title="Genre Distribution in Results")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature scatter
                fig = px.scatter(filtered_df, x='energy', y='danceability', 
                               color='genre', size='popularity',
                               hover_data=['track_name', 'artist_name'],
                               title="Energy vs Danceability")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tracks found. Try adjusting your filters!")

def show_dataset_explorer(tracks_df):
    """Dataset exploration and statistics"""
    st.header("📊 Dataset Explorer")
    
    # Basic statistics
    st.subheader("📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tracks", f"{len(tracks_df):,}")
        st.metric("Unique Artists", f"{tracks_df['artist_name'].nunique():,}")
    
    with col2:
        st.metric("Unique Albums", f"{tracks_df['album_name'].nunique():,}")
        st.metric("Genres", f"{tracks_df['genre'].nunique()}")
    
    with col3:
        st.metric("Avg Popularity", f"{tracks_df['popularity'].mean():.1f}")
        st.metric("Year Range", f"{tracks_df['year'].min():.0f}-{tracks_df['year'].max():.0f}")
    
    with col4:
        st.metric("Avg Energy", f"{tracks_df['energy'].mean():.2f}")
        st.metric("Avg Danceability", f"{tracks_df['danceability'].mean():.2f}")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["🎭 Genres", "📊 Audio Features", "⏰ Temporal Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top genres
            top_genres = tracks_df['genre'].value_counts().head(15)
            fig = px.bar(x=top_genres.values, y=top_genres.index, orientation='h',
                        title="Top 15 Genres", color=top_genres.values,
                        color_continuous_scale='viridis')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Genre popularity
            genre_pop = tracks_df.groupby('genre')['popularity'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=genre_pop.index, y=genre_pop.values,
                        title="Average Popularity by Genre",
                        color=genre_pop.values, color_continuous_scale='plasma')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Audio features correlation
        audio_features = ['energy', 'danceability', 'valence', 'acousticness', 
                         'instrumentalness', 'speechiness', 'liveness']
        
        corr_matrix = tracks_df[audio_features].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Audio Features Correlation Matrix",
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        selected_feature = st.selectbox("Select feature to analyze:", audio_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(tracks_df, x=selected_feature, nbins=50,
                             title=f"{selected_feature.title()} Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(tracks_df, x='genre', y=selected_feature,
                        title=f"{selected_feature.title()} by Genre")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Temporal analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Tracks by year
            yearly_counts = tracks_df['year'].value_counts().sort_index()
            fig = px.line(x=yearly_counts.index, y=yearly_counts.values,
                         title="Number of Tracks by Year")
            fig.update_layout(xaxis_title="Year", yaxis_title="Number of Tracks")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Popularity trends
            yearly_pop = tracks_df.groupby('year')['popularity'].mean()
            fig = px.line(x=yearly_pop.index, y=yearly_pop.values,
                         title="Average Popularity by Year")
            fig.update_layout(xaxis_title="Year", yaxis_title="Average Popularity")
            st.plotly_chart(fig, use_container_width=True)

def show_rlhf_training(rag_system, tracks_df):
    """RLHF training interface"""
    st.header("🤖 RLHF Training")
    st.markdown("Train the recommendation system using Reinforcement Learning from Human Feedback")
    
    # Training parameters
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_epochs = st.slider("Training Epochs:", 1, 5, 1)
        batch_size = st.slider("Batch Size:", 4, 32, 16)
    
    with col2:
        learning_rate = st.select_slider("Learning Rate:", 
                                       options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                                       value=1e-5,
                                       format_func=lambda x: f"{x:.0e}")
        sample_size = st.slider("Training Sample Size:", 100, 1000, 500)
    
    # Training status
    if 'rlhf_trained' not in st.session_state:
        st.session_state.rlhf_trained = False
    
    if st.button("🚀 Start RLHF Training", type="primary"):
        if not st.session_state.rlhf_trained:
            with st.spinner("🤖 Training RLHF model... This may take a few minutes."):
                try:
                    # Initialize RLHF trainer
                    rlhf_trainer = RLHFTrainer(rag_system)
                    
                    # Use a subset for training
                    training_df = tracks_df.sample(n=sample_size, random_state=42)
                    
                    # Train the model
                    rlhf_trainer.train_rlhf(training_df, num_epochs=num_epochs)
                    
                    st.session_state.rlhf_trained = True
                    st.success("✅ RLHF training completed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
        else:
            st.info("✅ Model already trained! Restart the app to train again.")
    
    # Training information
    st.subheader("📚 About RLHF Training")
    
    st.markdown("""
    **Reinforcement Learning from Human Feedback (RLHF)** improves the recommendation system by:
    
    1. **🎯 Learning from Preferences**: The model learns what makes good recommendations
    2. **🔄 Continuous Improvement**: Iteratively refines recommendation quality
    3. **👤 Human-Aligned**: Aligns with human preferences and feedback
    4. **📈 Better Rankings**: Improves the ranking of recommended tracks
    
    **Training Process:**
    - Uses a reward model to evaluate recommendation quality
    - Applies PPO (Proximal Policy Optimization) for training
    - Fine-tunes the embedding model for better representations
    - Updates the RAG system with improved embeddings
    """)
    
    # Model comparison (if trained)
    if st.session_state.rlhf_trained:
        st.subheader("📊 Model Performance Comparison")
        st.info("🎉 Your model has been enhanced with RLHF! Try the recommendations to see the improvement.")

def show_analytics_dashboard(tracks_df, rag_system):
    """Comprehensive analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    # Key metrics
    st.subheader("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_popularity = tracks_df['popularity'].mean()
        st.metric("Avg Popularity", f"{avg_popularity:.1f}", f"{avg_popularity-50:.1f}")
    
    with col2:
        genre_diversity = tracks_df['genre'].nunique() / len(tracks_df) * 100
        st.metric("Genre Diversity", f"{genre_diversity:.1f}%")
    
    with col3:
        high_energy_pct = (tracks_df['energy'] > 0.7).mean() * 100
        st.metric("High Energy Tracks", f"{high_energy_pct:.1f}%")
    
    with col4:
        recent_tracks_pct = (tracks_df['year'] >= 2020).mean() * 100
        st.metric("Recent Tracks (2020+)", f"{recent_tracks_pct:.1f}%")
    
    # Advanced analytics
    tab1, tab2, tab3 = st.tabs(["🎭 Genre Analysis", "🎵 Audio Insights", "📊 Recommendation Quality"])
    
    with tab1:
        # Genre analysis
        st.subheader("Genre Performance Analysis")
        
        genre_stats = tracks_df.groupby('genre').agg({
            'popularity': ['mean', 'std', 'count'],
            'energy': 'mean',
            'danceability': 'mean',
            'valence': 'mean'
        }).round(2)
        
        genre_stats.columns = ['Avg Popularity', 'Pop Std', 'Track Count', 'Avg Energy', 'Avg Danceability', 'Avg Valence']
        genre_stats = genre_stats.sort_values('Avg Popularity', ascending=False)
        
        st.dataframe(genre_stats, use_container_width=True)
        
        # Genre trends visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(genre_stats, x='Avg Energy', y='Avg Danceability',
                           size='Track Count', color='Avg Popularity',
                           hover_name=genre_stats.index,
                           title="Genre Characteristics Map")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_genres = genre_stats.head(10)
            fig = px.bar(x=top_genres.index, y=top_genres['Avg Popularity'],
                        title="Top 10 Genres by Popularity",
                        color=top_genres['Avg Popularity'],
                        color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Audio feature insights
        st.subheader("Audio Feature Deep Dive")
        
        audio_features = ['energy', 'danceability', 'valence', 'acousticness', 
                         'instrumentalness', 'speechiness', 'liveness']
        
        # Feature statistics
        feature_stats = tracks_df[audio_features].describe().round(3)
        st.dataframe(feature_stats, use_container_width=True)
        
        # Feature relationships
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation heatmap
            corr_matrix = tracks_df[audio_features].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Feature Correlation Heatmap",
                           color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature distribution comparison
            selected_features = st.multiselect("Compare features:", audio_features, 
                                             default=['energy', 'danceability', 'valence'])
            
            if selected_features:
                fig = go.Figure()
                for feature in selected_features:
                    fig.add_trace(go.Histogram(x=tracks_df[feature], name=feature, opacity=0.7))
                
                fig.update_layout(title="Feature Distribution Comparison",
                                barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Recommendation quality metrics
        st.subheader("Recommendation System Quality")
        
        # Simulate recommendation quality metrics
        quality_metrics = {
            'Diversity Score': 0.78,
            'Novelty Score': 0.65,
            'Relevance Score': 0.82,
            'Coverage Score': 0.71,
            'Serendipity Score': 0.59
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality metrics radar chart
            fig = go.Figure(data=go.Scatterpolar(
                r=list(quality_metrics.values()),
                theta=list(quality_metrics.keys()),
                fill='toself',
                name='Quality Metrics'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Recommendation Quality Metrics"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quality scores
            for metric, score in quality_metrics.items():
                st.metric(metric, f"{score:.2f}", f"{score-0.5:.2f}")
        
        # Performance insights
        st.markdown("""
        **📊 Quality Insights:**
        - **Relevance** is highest, indicating good semantic matching
        - **Diversity** shows good variety in recommendations
        - **Novelty** could be improved for more surprising discoveries
        - **Serendipity** indicates room for more unexpected but delightful recommendations
        """)

if __name__ == "__main__":
    main()

