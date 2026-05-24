Task 1: News Topic Classifier (BERT)
This project is an AI-powered web application that automatically reads a news headline and figures out its topic category.

The Goal: To sort text into four distinct groups: World, Sports, Business, or Sci/Tech.

How it works: It uses a powerful, pre-trained language model called BERT. When you type a headline into the Streamlit web interface, the model processes the words, analyzes the sentence structure, and outputs a prediction matching the most likely news category.

Task 2: End-to-End Machine Learning Pipeline
This project builds a clean, automated system to clean data and predict whether a customer is likely to leave a service (Customer Churn).

The Goal: To combine data cleaning, feature scaling, and machine learning training into one single, reusable pipeline structure using Scikit-learn.

How it works: Instead of handling missing data, converting category text to numbers, and scaling numbers separately, a unified Pipeline executes these steps automatically. It uses GridSearchCV to test different settings for a Random Forest classifier, finds the setup with the highest accuracy, and saves the final system into a reusable file.

Task 4: Context-Aware Chatbot (RAG)
This project builds an intelligent chatbot that can read a custom text document and use that specific information to answer your questions accurately without forgetting previous messages.

The Goal: To connect a Large Language Model to a private knowledge base (Retrieval-Augmented Generation) while keeping track of the conversation history.

How it works: It reads your custom data file, breaks it into small chunks, and saves them into a searchable database (Vector Store). When you chat with the app via Streamlit, the system retrieves the exact sentences from the document that match your question, reviews your previous chat messages for context, and generates a personalized, factual response.
