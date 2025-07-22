# Deployment Guide for Enhanced Music RAG System

## Quick Start

1. **Extract the zip file** to your desired directory
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run enhanced_app.py
   ```

## Streamlit Cloud Deployment

1. Upload the extracted files to a GitHub repository
2. Connect your GitHub repo to Streamlit Cloud
3. Set the main file as `enhanced_app.py`
4. Deploy!

## Local Development

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for embedding generation)
- Internet connection (for downloading models)

### Installation Steps
```bash
# Clone or extract the project
cd Music_RAG_Enhanced

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run enhanced_app.py
```

## Features Available

### 🏠 Home Page
- System overview and statistics
- Quick dataset insights
- Feature highlights

### 🎯 Smart Recommendations
- **Text-based Search**: Natural language music queries
- **Audio Features**: Slider-based feature matching
- **User Profiles**: Simulated user recommendation testing
- **Hybrid Method**: Combined approach for best results

### 🔍 Music Discovery
- Advanced filtering by genre, year, popularity
- Audio feature range selection
- Real-time search results

### 📊 Dataset Explorer
- Comprehensive dataset statistics
- Genre analysis and trends
- Audio feature correlations
- Temporal analysis

### 🤖 RLHF Training
- Train the recommendation system
- Improve model performance
- Human feedback integration

### 📈 Analytics Dashboard
- Performance metrics
- Quality insights
- Recommendation analysis

## Configuration

### Dataset
- The system uses `final_dataset.csv` by default
- Place your dataset in the same directory as the app
- Supported formats: CSV with music metadata and audio features

### Model Settings
- Default embedding model: `all-MiniLM-L6-v2`
- Vector database: ChromaDB (in-memory)
- RLHF: TRL framework with PPO

### Performance Tuning
- Adjust batch sizes in `music_rag_system.py`
- Modify caching settings in `enhanced_app.py`
- Configure RLHF parameters in `rlhf_module.py`

## Troubleshooting

### Common Issues

1. **Memory Error during initialization**
   - Reduce dataset size or increase system RAM
   - Modify batch_size in embedding generation

2. **Missing dependencies**
   - Ensure all packages in requirements.txt are installed
   - Use Python 3.8+ for compatibility

3. **Slow loading**
   - First run downloads embedding models (~90MB)
   - Subsequent runs use cached models

4. **RLHF training fails**
   - Requires significant computational resources
   - Consider using smaller dataset samples

### Performance Tips
- Use SSD storage for faster model loading
- Close other applications to free up RAM
- Use GPU if available (automatic detection)

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all dependencies are correctly installed
3. Ensure the dataset file is properly formatted

## System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space

**Recommended:**
- Python 3.9+
- 8GB+ RAM
- SSD storage
- GPU (optional, for faster processing)

Enjoy exploring music with AI-powered recommendations! 🎵

