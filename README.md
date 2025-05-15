Course Project
________________________________________
Introduction

With the rapid development of society and the continuous improvement of material living standards, individuals are increasingly prioritizing spiritual and cultural consumption. Music plays a significant role in daily life and entertainment. In advertising, music symbolizes brand identity and product style, influencing audience awareness, brand impression, and purchase intention. However, English songs often face challenges in resonating effectively within the Chinese market due to language barriers and cultural differences in emotional interpretation.
This project aims to analyze the emotional tone of English lyrics within a Chinese cultural context to select suitable advertising music for the Chinese market. By developing a sentiment analysis framework, we classify the emotional tone (positive, negative, romantic, peaceful, excited) of 759 English lyrics, adapting them to align with Chinese cultural preferences. Three tools—VADER, Hugging Face Transformers, and a customized Chinese-adapted VADER—were utilized, along with a supervised binary classification approach to address data limitations.

Key Objectives

The primary objective of this project is to analyze the emotional tone of English lyrics within the Chinese cultural context and verify whether the Chinese cultural background impacts the understanding of English lyrics. This analysis is crucial for screening music for advertisements that align with brand identity and audience preferences. By understanding the emotional nuances that resonate with Chinese audiences, advertisers can craft more effective and culturally sensitive campaigns.

Methodology

•	Dataset Collection
The dataset collection process involved leveraging APIs to gather a comprehensive set of English song lyrics suitable for sentiment analysis. Two main APIs were used:
- Spotify API:
Good at music metadata and playback features, but limited lyrics support (only available in all regions, and requires partner permissions)
- Genius API: 
Get complete song lyrics based on Spotify's song metadata, including structured passages (such as chorus, lead song). 
Spotify API is the industry standard for music metadata and playback, suitable for underlying data; Genius API is a premium source of lyrics that complements Spotify's shortcomings.
•	Preprocessing
•	Sentiment Analysis Tools
- VADER
It is a rule-based sentiment analysis tool, which generates the scores of sentiments by the intensity of lexical features and semantic meanings of the text. Vader returns four components with associated intensity scores.
VADER is particularly good at processing sentiments expressed through colloquial language because of its rule-based architecture and pre-trained lexicon.
- Hugging Face
It is a platform that houses a variety of cutting-edge natural language processing (NLP) models, including transformer-based models like BERT, GPT, and many others, even if it isn’t a dedicated sentiment analysis tool. These models are available via the Hugging Face Transformers library.
- Logistic Regression
	Used for multi-class sentiment classification attempts, aiming to filter more emotions to match various advertising styles. Logistic regression provided a statistical approach to sentiment classification, complementing the rule-based and deep learning models.
•	Sentiment Analysis of Lyrics
________________________________________
Data Collection

Data Sources and Acquisition

•	APIs Used: 
o	Spotify API (spotipy) with queries like "year: 2005-2014, genre: pop" to fetch 1049 songs. 
Authenticated through the Spotipy library, client_id and client_secret are used. Then, through Spotipy's search method, the song information is obtained, such as the song name, artist, year and other metadata, and stored in the DataFrame.
Functions and parameters:
Search query: Search by year and genre in batches (such as year:2005-2014 genre:pop).
Pagination processing: Get batch_size=50 songs per batch, and limit the total offset offset + limit ≤ 1000.
Data extraction: Extract the song's unique identifier track_id, title title, artist artist, and release year year from the returned results, and store them in the all_tracks list without duplication.

Error handling:

Catch exceptions and print error information, such as request failure or rate limit triggering. Mitigate rate limit issues with time.sleep(1).
o	Genius API (lyricsgenius) to retrieve lyrics, ensuring English content via langdetect.
Use the LyricsGenius library and authenticate it through the GENIUS_API_TOKEN provided. Search for lyrics by metadata such as song name and artist fetched from Spotify, and detect the language of the lyrics, skipping if it's non-English. For example, when searching for Chinese or Korean songs, if it detects that the lyrics are not in English, those songs are skipped.
Functions and parameters:
Lyrics search: Search lyrics by song title and artist through genius_api.search_song(title, artist).
Language filtering: Use langdetect to detect the language of lyrics. If it is not English (such as Chinese, Korean, Japanese, etc.), skip storage.
Deduplication and storage: Only store unique lyrics in English to the lyrics_data list.

Common Problems:

Lyrics not found (e.g. Lyrics not found): The search may fail due to the lack of content in the Genius database, mismatch of song name/artist name, or special characters.
No lyrics (e.g. Specified song does not contain lyrics): Some songs (e.g. instrumental or Intro) are marked as not containing lyrics.
Non-English lyrics (e.g. Skipping non-English song): Filter non-English content, such as Chinese, Korean, and Japanese songs.
•	Optimization and considerations
Rate limit: Use time.sleep(1) to reduce the request frequency to avoid triggering API limits.
Error retry: Some codes implement retry mechanism (such as Another Love by Tom Odell, which succeeds after the first failure).
Language detection accuracy: Depends on langdetect library, which may have misjudgment (such as mixed language lyrics).
Special character processing: Some song names contain brackets or symbols (such as Ni**as In Paris), so make sure the search parameters are escaped correctly.
•	Initial Dataset: 1049 songs with columns: id, title, artist, year, and lyrics.
The dataset comprised 1049 songs with necessary metadata, serving as a comprehensive foundation for analysis.
________________________________________
Data Cleaning

The cleaning process was essential to refine the dataset and ensure its suitability for sentiment analysis.
•	Excel Cleaning: The initial dataset underwent a cleaning process using Excel, where duplicate entries and non-English songs were removed. After this refinement, a final dataset of 759 English songs was retained, with the following columns:
o	id: Unique identifier for each song
o	title: Title of the song
o	artist: Artist of the song
o	lyrics: Full lyrics of the song
o	label: Emotional label assigned to the song
Considering the emotional impact of the amount of English lyrics, need to ensure that the vast majority of the lyrics are in English.
•	Text Processing: 
Removed:
o	Whitespace and empty rows
o	Contributor information
o	Advertisements (e.g., "See Coldplay LiveGet tickets as low as $154You might also like")
o	Lyrics structure tags (e.g., [Chorus])
o	Embedded information
Preprocess (lowercasing, tokenization, lemmatization) for supervised learning.
•	Result: Dataset contained 759 English song lyrics, ready for sentiment analysis with columns id, title, artist, lyrics.

Labeling Process

•	Manual Labeling: Randomly selected 166 songs and were labeled by 3 Chinese students into five mutually exclusive categories: excited, negative, romantic, peaceful, and positive, with each song assigned exactly one label.
•	Processing: Applied a majority voting approach to determine the final label for each song. 
If all three classmates assigned the same label, that label was adopted. 
If all three labels differed, a fourth person was brought in to label the song, and their label was used as the final decision.
•	Results: Label data for the 166 songs with the following distribution based on initial counts: 43 excited, 32 negative, 31 romantic, 30 peaceful, and 30 positive.
•	Challenges: Due to time and resource constraints, the amount of data is too small to capture complex emotional patterns, is prone to overfitting, and is difficult to meet the training requirements of five categories of classification.
•	Final Dataset: 759 songs with columns: id, title, artist, year, lyrics and label, ready for machine learning.
________________________________________
Limitations

Due to time and resource constraints, collecting a robust dataset with single sentiment labeling can be more challenging than allowing multiple sentiments. Also, advertising music often aims to evoke a range of emotions to connect with diverse audiences. By restricting sentiment labeling to one emotion, you may miss the complex emotional layers that make the music effective in conveying the brand message. And might not fully reflect the emotional engagement intended by the music, thus limiting the effectiveness of the advertising campaign.

Individual focus and interpretation can vary widely. This subjectivity poses challenges in both developing tools that accurately emulate human judgment and ensuring uniformity among human evaluators. Emphasizing the necessity for clear guidelines and training for judges, as well as the ongoing enhancement of sentiment analysis algorithms to better match human interpretations.[10]

Men and women have different perspectives on music (Zander, 2006), and research shows that women are generally better at detecting nonverbal cues than men (Hall, Carter & Horgan, 2000). Therefore, having both male and female participants label song lyrics could help uncover gender differences in lyric interpretation. This approach can enhance the alignment of advertising products with their target audiences. 
________________________________________
Prospectives

Hybrid Approach for Sentiment Classification
Combining dictionary-based methods with machine learning approaches for binary text sentiment classification can address the challenge of limited emotional vocabulary expansion inherent in dictionary-based methods, as well as the "curse of dimensionality" often encountered in feature selection within machine learning methods.

Gender-Inclusive Lyric Analysis
Having both male and female participants label song lyrics could help uncover gender differences in lyric interpretation. This approach can enhance the alignment of advertising products with their target audiences.

Multimodal Analysis for Cross-Cultural Sentiment Recognition
Integrating audio features (such as rhythm and melody) with lyrical sentiment to enhance the accuracy of sentiment recognition. For instance, slower-paced songs might be perceived as "soothing" by Chinese users. 
Additionally, leveraging sentiment from Chinese social media comments, which reflect Chinese users' evaluations after listening to the music, can provide insights into the combined impact of rhythm and lyrics. This approach offers promising potential for more precise sentiment analysis across diverse cultural contexts.

________________________________________
Conclusion:

This study demonstrates the critical role of cultural adaptation in analyzing English lyrics for advertising music targeting Chinese audiences. By comparing three sentiment analysis frameworks—VADER, Hugging Face Transformers, and a culturally adapted VADER—the project revealed significant discrepancies in emotional interpretation. The original VADER, trained on Western data, skewed toward positivity (500+ songs labeled positive), while the Chinese-adapted VADER emphasized neutrality (~250 neutral), aligning with cultural norms of restraint. Hugging Face Transformers, leveraging contextual understanding, identified more negative sentiments (400+), highlighting its sensitivity to implicit metaphors.
Facing limited labeled data (166 songs), binary classifiers achieved over 90% accuracy in filtering non-sentiment categories (e.g., "non-excited"), enabling practical screening of unsuitable tracks. Temporal analysis of unlabeled songs (593) revealed evolving Chinese perceptions, shifting from emotional sensitivity (2005–2013) to cultural adaptation (2014–2019) and stabilization (2020–2024).
While constrained by data scarcity and subjective labeling, this work underscores the necessity of cultural context in cross-cultural advertising. Future integration of audio features and dynamic lexicon updates could refine alignment. For advertisers, prioritizing subtlety and neutrality—as reflected in the adapted VADER—offers a strategic edge in resonating with Chinese audiences. This framework bridges Western content with Eastern interpretation, advancing globalized yet localized advertising strategies.
