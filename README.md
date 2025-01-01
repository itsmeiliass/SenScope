# SenScope

## Overview

SenScope is an innovative application that applies Natural Language Processing (NLP) techniques to analyze and classify social media feedback, specifically Twitter comments. The project utilizes machine learning models to determine the sentiment of user feedback, categorizing it into three distinct classes: **positive**, **negative**, and **neutral**.

This classification provides valuable insights for businesses, organizations, and individuals to gauge public opinion, understand customer sentiment, and improve their services or products.

### Key Features:
- Sentiment classification for Twitter comments.
- Visual representation of sentiment analysis results.
- User-friendly landing page for easy interaction.

## Project Structure

- **`/graphs`**: Contains the generated graphs and visualizations of the sentiment analysis data.
- **`/templates`**: Contains the HTML files for rendering the Flask app, including the index page and results page.
- **`app.py`**: The main Flask application that handles routing and sentiment analysis.
- **`main.py`**: the full data processing and model deployement .


## Prerequisites

Before running the app, make sure you have the following installed on your system:

- Python (3.x)
- pip (Python package installer)

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/itsmeiliass/SenScope.git
   cd SenScope
   python app.py
   
