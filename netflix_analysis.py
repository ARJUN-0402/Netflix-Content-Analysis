import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class NetflixAnalyzer:
    def __init__(self, data_path='netflix_data.csv'):
        """
        Initialize the Netflix Analyzer with data loading and preprocessing
        """
        self.df = None
        self.load_data(data_path)
        self.preprocess_data()
        
    def load_data(self, data_path):
        """
        Load Netflix data from CSV file
        """
        try:
            self.df = pd.read_csv(data_path)
            print(f"Successfully loaded {len(self.df)} Netflix titles")
        except FileNotFoundError:
            print(f"Data file {data_path} not found. Please check the file path.")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def preprocess_data(self):
        """
        Preprocess and clean the data
        """
        if self.df is None:
            print("No data to preprocess")
            return
            
        # Convert date_added to datetime
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], 
                                               format='%B %d, %Y', 
                                               errors='coerce')
        
        # Extract year, month, day from date_added
        self.df['year_added'] = self.df['date_added'].dt.year
        self.df['month_added'] = self.df['date_added'].dt.month_name()
        self.df['day_added'] = self.df['date_added'].dt.day_name()
        
        # Clean duration column for movies (convert to minutes)
        self.df['duration_min'] = self.df['duration'].apply(
            lambda x: int(x.split(' ')[0]) if 'min' in str(x) else np.nan
        )
        
        # Clean duration for TV shows (convert to seasons)
        self.df['seasons'] = self.df['duration'].apply(
            lambda x: int(x.split(' ')[0]) if 'Season' in str(x) else np.nan
        )
        
        # Create content type column
        self.df['content_type'] = self.df['type'].apply(
            lambda x: 'Movie' if x == 'Movie' else 'TV Show'
        )
        
        print("Data preprocessing completed")
        
    def content_type_analysis(self):
        """
        Analyze content types (Movies vs TV Shows)
        """
        content_counts = self.df['content_type'].value_counts()
        print("\nContent Type Analysis:")
        print(content_counts)
        return content_counts
    
    def country_analysis(self):
        """
        Analyze content by country
        """
        country_counts = self.df['country'].value_counts()
        print("\nTop Countries by Content Count:")
        print(country_counts.head(10))
        return country_counts
    
    def release_year_analysis(self):
        """
        Analyze content by release year
        """
        year_counts = self.df['release_year'].value_counts().sort_index()
        print("\nContent by Release Year:")
        print(year_counts.tail(10))
        return year_counts
    
    def rating_analysis(self):
        """
        Analyze content by rating
        """
        rating_counts = self.df['rating'].value_counts()
        print("\nContent by Rating:")
        print(rating_counts)
        return rating_counts
    
    def genre_analysis(self):
        """
        Analyze content by genre
        """
        # Split genres and count occurrences
        all_genres = self.df['listed_in'].str.split(', ').explode()
        genre_counts = all_genres.value_counts()
        print("\nTop Genres:")
        print(genre_counts.head(10))
        return genre_counts
    
    def time_series_analysis(self):
        """
        Analyze content added over time
        """
        if 'year_added' in self.df.columns:
            yearly_added = self.df['year_added'].value_counts().sort_index()
            print("\nContent Added by Year:")
            print(yearly_added)
            return yearly_added
        else:
            print("Year added data not available")
            return None
    
    def advanced_content_analysis(self):
        """
        Perform advanced content analysis including:
        - Content type distribution over years
        - Country-content type analysis
        - Rating trends over time
        """
        print("\n=== Advanced Content Analysis ===")
        
        # Content type distribution over years
        content_by_year = self.df.groupby(['release_year', 'content_type']).size().unstack(fill_value=0)
        print("\nContent Type Distribution by Release Year:")
        print(content_by_year.tail(5))
        
        # Country-content type analysis
        country_content = self.df.groupby(['country', 'content_type']).size().unstack(fill_value=0)
        print("\nContent Type Distribution by Country (Top 5):")
        print(country_content.head(5))
        
        # Rating trends over time
        rating_by_year = self.df.groupby(['release_year', 'rating']).size().unstack(fill_value=0)
        print("\nRating Trends by Year:")
        print(rating_by_year.tail(5))
        
        return content_by_year, country_content, rating_by_year
     
    def director_analysis(self):
        """
        Analyze directors and their content
        """
        print("\n=== Director Analysis ===")
        
        # Split directors and count occurrences
        all_directors = self.df['director'].str.split(', ').explode()
        director_counts = all_directors.value_counts()
        
        print("\nTop Directors:")
        print(director_counts.head(10))
        
        # Directors by content type
        if 'director' in self.df.columns:
            director_type = self.df.groupby(['director', 'content_type']).size().unstack(fill_value=0)
            print("\nDirectors by Content Type (Top 5):")
            print(director_type.head(5))
            
        return director_counts
     
    def cast_analysis(self):
        """
        Analyze cast members and their content
        """
        print("\n=== Cast Analysis ===")
        
        # Split cast and count occurrences
        all_cast = self.df['cast'].str.split(', ').explode()
        cast_counts = all_cast.value_counts()
        
        print("\nTop Cast Members:")
        print(cast_counts.head(10))
        
        return cast_counts
     
    def duration_analysis(self):
        """
        Analyze content duration for movies and TV shows
        """
        print("\n=== Duration Analysis ===")
        
        # Movies duration analysis
        movies = self.df[self.df['content_type'] == 'Movie']
        if not movies['duration_min'].isnull().all():
            print(f"\nAverage Movie Duration: {movies['duration_min'].mean():.2f} minutes")
            print(f"Shortest Movie: {movies['duration_min'].min()} minutes")
            print(f"Longest Movie: {movies['duration_min'].max()} minutes")
        
        # TV Shows seasons analysis
        tv_shows = self.df[self.df['content_type'] == 'TV Show']
        if not tv_shows['seasons'].isnull().all():
            print(f"\nAverage TV Show Seasons: {tv_shows['seasons'].mean():.2f}")
            print(f"Minimum Seasons: {tv_shows['seasons'].min()}")
            print(f"Maximum Seasons: {tv_shows['seasons'].max()}")
     
    def keyword_analysis(self):
        """
        Analyze keywords in descriptions
        """
        print("\n=== Keyword Analysis ===")
        
        # Convert descriptions to lowercase and split into words
        all_descriptions = ' '.join(self.df['description'].fillna('').str.lower())
        words = all_descriptions.split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        print("\nTop Keywords in Descriptions:")
        for word, count in word_counts.most_common(10):
            print(f"{word}: {count}")
        
        return word_counts
    
    def visualize_content_types(self):
        """
        Create unique visualizations for content types
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Netflix Content Analysis - Content Types', fontsize=16, fontweight='bold')
        
        # 1. Content Type Distribution Pie Chart
        content_counts = self.df['content_type'].value_counts()
        axes[0, 0].pie(content_counts.values, labels=content_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Content Type Distribution')
        
        # 2. Content Type by Year Stacked Bar Chart
        content_by_year = self.df.groupby(['release_year', 'content_type']).size().unstack(fill_value=0)
        content_by_year.plot(kind='bar', stacked=True, ax=axes[0, 1])
        axes[0, 1].set_title('Content Type Distribution by Year')
        axes[0, 1].set_xlabel('Release Year')
        axes[0, 1].set_ylabel('Number of Titles')
        axes[0, 1].legend(title='Content Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Content Type by Rating Heatmap
        content_rating = self.df.groupby(['content_type', 'rating']).size().unstack(fill_value=0)
        sns.heatmap(content_rating, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Content Type vs Rating')
        axes[1, 0].set_xlabel('Rating')
        axes[1, 0].set_ylabel('Content Type')
        
        # 4. Content Type by Country (Top 10)
        country_content = self.df.groupby(['country', 'content_type']).size().unstack(fill_value=0)
        top_countries = country_content.sum(axis=1).nlargest(10).index
        country_content.loc[top_countries].plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title('Content Type Distribution by Country (Top 10)')
        axes[1, 1].set_xlabel('Country')
        axes[1, 1].set_ylabel('Number of Titles')
        axes[1, 1].legend(title='Content Type')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('netflix_content_types.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_trends(self):
        """
        Create unique visualizations for trends over time
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Netflix Content Analysis - Trends Over Time', fontsize=16, fontweight='bold')
        
        # 1. Content Added Over Time Line Chart
        if 'year_added' in self.df.columns:
            yearly_added = self.df['year_added'].value_counts().sort_index()
            axes[0, 0].plot(yearly_added.index, yearly_added.values, marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_title('Content Added Over Time')
            axes[0, 0].set_xlabel('Year Added')
            axes[0, 0].set_ylabel('Number of Titles')
            axes[0, 0].grid(True)
        
        # 2. Release Year Distribution Histogram
        axes[0, 1].hist(self.df['release_year'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Distribution of Release Years')
        axes[0, 1].set_xlabel('Release Year')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Rating Trends Over Time
        rating_by_year = self.df.groupby(['release_year', 'rating']).size().unstack(fill_value=0)
        rating_by_year.plot(kind='area', stacked=True, ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('Rating Trends Over Time')
        axes[1, 0].set_xlabel('Release Year')
        axes[1, 0].set_ylabel('Number of Titles')
        axes[1, 0].legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Genre Popularity Over Time (Top 5 Genres)
        all_genres = self.df['listed_in'].str.split(', ').explode()
        top_genres = all_genres.value_counts().head(5).index
        
        genre_year_data = []
        for genre in top_genres:
            genre_mask = self.df['listed_in'].str.contains(genre, na=False)
            genre_by_year = self.df[genre_mask].groupby('release_year').size()
            genre_year_data.append((genre, genre_by_year))
        
        for genre, data in genre_year_data:
            axes[1, 1].plot(data.index, data.values, marker='o', label=genre, linewidth=2)
        axes[1, 1].set_title('Top 5 Genre Popularity Over Time')
        axes[1, 1].set_xlabel('Release Year')
        axes[1, 1].set_ylabel('Number of Titles')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('netflix_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_duration(self):
        """
        Create unique visualizations for content duration
        """
        # Filter data for movies and TV shows with valid duration
        movies = self.df[(self.df['content_type'] == 'Movie') & (self.df['duration_min'].notnull())]
        tv_shows = self.df[(self.df['content_type'] == 'TV Show') & (self.df['seasons'].notnull())]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Netflix Content Analysis - Duration Insights', fontsize=16, fontweight='bold')
        
        # 1. Movie Duration Distribution
        if not movies.empty:
            axes[0, 0].hist(movies['duration_min'], bins=20, color='lightcoral', edgecolor='black')
            axes[0, 0].set_title('Movie Duration Distribution')
            axes[0, 0].set_xlabel('Duration (minutes)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(movies['duration_min'].mean(), color='red', linestyle='--',
                              label=f'Mean: {movies["duration_min"].mean():.1f} min')
            axes[0, 0].legend()
        
        # 2. TV Show Seasons Distribution
        if not tv_shows.empty:
            axes[0, 1].hist(tv_shows['seasons'], bins=10, color='lightblue', edgecolor='black')
            axes[0, 1].set_title('TV Show Seasons Distribution')
            axes[0, 1].set_xlabel('Number of Seasons')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(tv_shows['seasons'].mean(), color='blue', linestyle='--',
                              label=f'Mean: {tv_shows["seasons"].mean():.1f} seasons')
            axes[0, 1].legend()
        
        # 3. Duration by Content Type Box Plot
        duration_data = [movies['duration_min'].dropna(), tv_shows['seasons'].dropna()]
        axes[1, 0].boxplot(duration_data, labels=['Movies (min)', 'TV Shows (seasons)'])
        axes[1, 0].set_title('Duration Comparison: Movies vs TV Shows')
        axes[1, 0].set_ylabel('Duration')
        
        # 4. Duration by Rating (Movies only)
        if not movies.empty:
            rating_duration = movies.groupby('rating')['duration_min'].mean().sort_values(ascending=False)
            axes[1, 1].bar(rating_duration.index, rating_duration.values, color='orange')
            axes[1, 1].set_title('Average Movie Duration by Rating')
            axes[1, 1].set_xlabel('Rating')
            axes[1, 1].set_ylabel('Average Duration (minutes)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('netflix_duration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_genres(self):
        """
        Create unique visualizations for genre analysis
        """
        # Split genres and count occurrences
        all_genres = self.df['listed_in'].str.split(', ').explode()
        genre_counts = all_genres.value_counts().head(15)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Netflix Content Analysis - Genre Insights', fontsize=16, fontweight='bold')
        
        # 1. Top Genres Bar Chart
        axes[0, 0].barh(genre_counts.index, genre_counts.values, color='purple')
        axes[0, 0].set_title('Top 15 Genres')
        axes[0, 0].set_xlabel('Number of Titles')
        
        # 2. Genre Distribution by Content Type
        genre_content = self.df.groupby(['listed_in', 'content_type']).size().unstack(fill_value=0)
        # Get top 10 genres by total count
        top_genres = all_genres.value_counts().head(10).index
        genre_content_top = genre_content.loc[genre_content.index.isin(top_genres)]
        genre_content_top.plot(kind='bar', stacked=True, ax=axes[0, 1])
        axes[0, 1].set_title('Genre Distribution by Content Type (Top 10)')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Number of Titles')
        axes[0, 1].legend(title='Content Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Genre Network Graph (simplified)
        # For visualization purposes, we'll create a simple network of top genres
        from collections import Counter
        genre_pairs = []
        for genres in self.df['listed_in'].str.split(', '):
            if isinstance(genres, list) and len(genres) > 1:
                for i in range(len(genres)):
                    for j in range(i+1, len(genres)):
                        genre_pairs.append((genres[i].strip(), genres[j].strip()))
        
        # Count co-occurrences
        pair_counts = Counter(genre_pairs)
        top_pairs = pair_counts.most_common(10)
        
        # Create a simple bar chart of genre co-occurrences
        if top_pairs:
            pairs, counts = zip(*top_pairs)
            pair_labels = [f"{p[0]} & {p[1]}" for p in pairs]
            axes[1, 0].barh(pair_labels, counts, color='green')
            axes[1, 0].set_title('Top Genre Combinations')
            axes[1, 0].set_xlabel('Co-occurrence Count')
        
        # 4. Genre Popularity Over Time
        # Get top 5 genres
        top_5_genres = all_genres.value_counts().head(5).index
        
        for genre in top_5_genres:
            genre_mask = self.df['listed_in'].str.contains(genre, na=False)
            genre_by_year = self.df[genre_mask].groupby('release_year').size()
            axes[1, 1].plot(genre_by_year.index, genre_by_year.values, marker='o', label=genre, linewidth=2)
        
        axes[1, 1].set_title('Top 5 Genre Popularity Over Time')
        axes[1, 1].set_xlabel('Release Year')
        axes[1, 1].set_ylabel('Number of Titles')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('netflix_genres.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def interactive_dashboard(self):
        """
        Create an interactive dashboard for exploring Netflix content
        """
        print("\n=== Interactive Netflix Dashboard ===")
        print("Welcome to the Netflix Content Analysis Dashboard!")
        print("Choose an option to explore:")
        print("1. Content Type Analysis")
        print("2. Genre Analysis")
        print("3. Rating Analysis")
        print("4. Duration Analysis")
        print("5. Search Content")
        print("6. Get Recommendations")
        print("7. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-7): ").strip()
                
                if choice == '1':
                    self.content_type_analysis()
                    self.visualize_content_types()
                elif choice == '2':
                    self.genre_analysis()
                    self.visualize_genres()
                elif choice == '3':
                    self.rating_analysis()
                    self.visualize_trends()
                elif choice == '4':
                    self.duration_analysis()
                    self.visualize_duration()
                elif choice == '5':
                    self.search_content()
                elif choice == '6':
                    self.content_recommender()
                elif choice == '7':
                    print("Thank you for using the Netflix Content Analysis Dashboard!")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 7.")
                    
            except KeyboardInterrupt:
                print("\n\nDashboard interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
    
    def content_recommender(self):
        """
        Recommend content based on user preferences
        """
        print("\n=== Content Recommender ===")
        print("Let's find some content you might like!")
        
        # Get user preferences
        print("\nWhat type of content are you interested in?")
        print("1. Movies")
        print("2. TV Shows")
        print("3. Both")
        
        content_type_choice = input("Enter your choice (1-3): ").strip()
        content_type = None
        if content_type_choice == '1':
            content_type = 'Movie'
        elif content_type_choice == '2':
            content_type = 'TV Show'
        elif content_type_choice != '3':
            print("Invalid choice. Showing both types.")
        
        # Get genre preference
        all_genres = self.df['listed_in'].str.split(', ').explode().unique()
        print(f"\nAvailable genres: {', '.join(all_genres[:10])}...")
        genre = input("Enter a genre you're interested in (or press Enter to skip): ").strip()
        
        # Get rating preference
        ratings = self.df['rating'].unique()
        print(f"\nAvailable ratings: {', '.join(ratings)}")
        rating = input("Enter a rating you prefer (or press Enter to skip): ").strip()
        
        # Filter content based on preferences
        filtered_df = self.df.copy()
        
        if content_type:
            filtered_df = filtered_df[filtered_df['content_type'] == content_type]
        
        if genre:
            filtered_df = filtered_df[filtered_df['listed_in'].str.contains(genre, case=False, na=False)]
        
        if rating:
            filtered_df = filtered_df[filtered_df['rating'] == rating.upper()]
        
        # Show recommendations
        if not filtered_df.empty:
            print(f"\nFound {len(filtered_df)} recommendations:")
            recommendations = filtered_df.sample(min(5, len(filtered_df)))
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                print(f"\n{idx}. {row['title']} ({row['release_year']})")
                print(f"   Type: {row['content_type']}")
                print(f"   Genre: {row['listed_in']}")
                print(f"   Rating: {row['rating']}")
                print(f"   Description: {row['description'][:100]}...")
        else:
            print("\nSorry, no content matches your preferences. Try different criteria.")
    
    def search_content(self):
        """
        Search for specific content
        """
        print("\n=== Content Search ===")
        query = input("Enter a title, actor, director, or keyword to search for: ").strip()
        
        if not query:
            print("No search query provided.")
            return
        
        # Search in multiple columns
        title_matches = self.df[self.df['title'].str.contains(query, case=False, na=False)]
        cast_matches = self.df[self.df['cast'].str.contains(query, case=False, na=False)]
        director_matches = self.df[self.df['director'].str.contains(query, case=False, na=False)]
        description_matches = self.df[self.df['description'].str.contains(query, case=False, na=False)]
        
        # Combine all matches
        all_matches = pd.concat([title_matches, cast_matches, director_matches, description_matches]).drop_duplicates()
        
        if not all_matches.empty:
            print(f"\nFound {len(all_matches)} results for '{query}':")
            for idx, (_, row) in enumerate(all_matches.iterrows(), 1):
                print(f"\n{idx}. {row['title']} ({row['release_year']})")
                print(f"   Type: {row['content_type']}")
                print(f"   Genre: {row['listed_in']}")
                print(f"   Rating: {row['rating']}")
                print(f"   Description: {row['description'][:100]}...")
                if idx >= 10:  # Limit to 10 results
                    print(f"\n... and {len(all_matches) - 10} more results")
                    break
        else:
            print(f"\nNo results found for '{query}'. Try a different search term.")
 
# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NetflixAnalyzer()
     
    # Run basic analyses
    analyzer.content_type_analysis()
    analyzer.country_analysis()
    analyzer.release_year_analysis()
    analyzer.rating_analysis()
    analyzer.genre_analysis()
    analyzer.time_series_analysis()
     
    # Run advanced analyses
    analyzer.advanced_content_analysis()
    analyzer.director_analysis()
    analyzer.cast_analysis()
    analyzer.duration_analysis()
    analyzer.keyword_analysis()
    
    # Run visualizations
    analyzer.visualize_content_types()
    analyzer.visualize_trends()
    analyzer.visualize_duration()
    analyzer.visualize_genres()

    # Run interactive components
    analyzer.interactive_dashboard()
    analyzer.content_recommender()
    analyzer.search_content()