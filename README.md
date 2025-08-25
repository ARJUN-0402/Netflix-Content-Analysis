# Netflix Content Analysis

An advanced Netflix content analysis tool that provides comprehensive insights into Netflix's content library through data analysis, visualization, and interactive features.

## Features

### Data Analysis
- Content type analysis (Movies vs TV Shows)
- Geographic distribution of content
- Release year trends
- Rating distribution
- Genre analysis
- Director and cast analysis
- Duration analysis for movies and TV shows
- Keyword analysis in descriptions

### Advanced Analysis
- Content type distribution over years
- Country-content type analysis
- Rating trends over time
- Genre popularity trends

### Unique Visualizations
- Content type distribution pie charts and bar charts
- Genre analysis with horizontal bar charts
- Rating distribution visualizations
- Duration analysis with histograms and box plots
- Word cloud visualization of descriptions
- Comprehensive dashboard with multiple visualizations

### Interactive Components
- Interactive dashboard for exploring content
- Content recommender based on user preferences
- Search functionality for specific content

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/netflix-content-analysis.git
   ```

2. Navigate to the project directory:
   ```bash
   cd netflix-content-analysis
   ```

3. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud
   ```

## Usage

1. Run the analysis script:
   ```bash
   python netflix_analysis.py
   ```

2. The script will automatically:
   - Load and preprocess the Netflix data
   - Perform all analyses
   - Generate visualization files
   - Launch the interactive dashboard

## Data

The project uses a sample Netflix dataset (`netflix_data.csv`) that includes:
- Show ID
- Type (Movie/TV Show)
- Title
- Director
- Cast
- Country
- Date added
- Release year
- Rating
- Duration
- Listed in (genres)
- Description

## Visualizations Generated

- `content_types_analysis.png`: Content type distribution
- `genre_analysis.png`: Genre analysis
- `rating_analysis.png`: Rating distribution
- `duration_analysis.png`: Duration analysis
- `wordcloud_analysis.png`: Word cloud of descriptions
- `netflix_dashboard.png`: Comprehensive dashboard

## Interactive Features

### Dashboard
Navigate through different analysis sections with the interactive dashboard menu.

### Content Recommender
Get personalized content recommendations based on:
- Content type preference (Movies/TV Shows)
- Genre preference
- Rating preference

### Search Functionality
Search for content by:
- Title
- Actor
- Director
- Keywords in descriptions

## Project Structure

```
netflix-content-analysis/
├── netflix_analysis.py     # Main analysis script
├── netflix_data.csv        # Sample Netflix dataset
├── README.md               # This file
├── *.png                   # Generated visualization files
```

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud (optional, for word cloud visualization)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is for educational purposes
- The sample dataset is fictional and created for demonstration purposes