# Enhanced Music RAG System

## Overview
This is an enhanced version of the Music RAG (Retrieval-Augmented Generation) recommendation system with the following improvements:

### ✅ Fixed Issues
- **NaN Values Error**: Fixed data preprocessing pipeline to handle missing values properly
- **Data Type Consistency**: Ensured all audio features are properly normalized and within valid ranges
- **Error Handling**: Added robust error handling throughout the system

### 🚀 New Features
- **RLHF Integration**: Added Reinforcement Learning from Human Feedback for improved recommendations
- **Enhanced UI**: Modern, responsive Streamlit interface with multiple recommendation methods
- **Hybrid Recommendations**: Combines semantic search with audio feature matching
- **Advanced Analytics**: Comprehensive dashboard with visualizations and insights

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the enhanced Streamlit app:
```bash
streamlit run enhanced_app.py
```

## Features

### 🎯 Smart Recommendations
- **Text-based Search**: Semantic music search using natural language
- **Audio Features**: Recommendation based on musical characteristics
- **User Profiles**: Personalized recommendations based on listening history
- **Hybrid Method**: Combines multiple recommendation approaches

### 🤖 RLHF Training
- Train the system using Reinforcement Learning from Human Feedback
- Improves recommendation quality over time
- Uses reward models to align with human preferences

### 📊 Analytics Dashboard
- Dataset exploration and statistics
- Genre analysis and trends
- Audio feature insights
- Recommendation quality metrics

### 🔍 Advanced Discovery
- Multi-dimensional filtering
- Real-time search and exploration
- Interactive visualizations

## Usage

### Basic Recommendations
```python
from music_rag_system import MusicRAGSystem
from data_loader import MusicDataLoader

# Load data and initialize system
loader = MusicDataLoader()
tracks_df = loader.load_final_dataset('final_dataset.csv')

rag_system = MusicRAGSystem()
rag_system.setup_vector_store(tracks_df)

# Get recommendations
recommendations, _ = rag_system.get_recommendations(
    preference_text="upbeat dance music for workouts",
    n_recommendations=10
)
```

### RLHF Training
```python
from rlhf_module import RLHFTrainer

# Initialize and train RLHF
rlhf_trainer = RLHFTrainer(rag_system)
rlhf_trainer.train_rlhf(tracks_df, num_epochs=1)
```

## Files Structure

- `enhanced_app.py` - Main Streamlit application with modern UI
- `music_rag_system.py` - Core RAG recommendation engine
- `data_loader.py` - Data loading and preprocessing (fixed NaN issues)
- `rlhf_module.py` - RLHF training implementation
- `user_simulator.py` - User behavior simulation
- `evaluator.py` - Recommendation evaluation metrics
- `visualization.py` - Data visualization utilities
- `main.py` - Command-line interface with RLHF support

## Deployment

The system is ready for Streamlit deployment. All dependencies are included in `requirements.txt`.

For production deployment:
1. Ensure `final_dataset.csv` is in the project directory
2. Deploy using Streamlit Cloud or your preferred platform
3. The system will automatically initialize and cache the RAG components

## Performance Improvements

- **Caching**: Streamlit caching for faster loading
- **Batch Processing**: Efficient embedding generation
- **Memory Optimization**: Reduced memory usage for large datasets
- **Error Recovery**: Graceful handling of missing data

## Technical Details

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: ChromaDB for similarity search
- **RLHF Framework**: TRL (Transformers Reinforcement Learning)
- **UI Framework**: Streamlit with custom CSS styling
- **Data Processing**: Pandas with robust NaN handling

## License

This project is open source and available under the MIT License.

