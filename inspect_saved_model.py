import pickle

# Load the saved model dictionary
with open('./models/enhanced_lda_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

n_topics = model_data['n_topics']
document_names = model_data['document_names']
topic_distributions = model_data['topic_distributions']

# Find dominant topic for each document
dominant_topics = []
for dist in topic_distributions:
    dominant_topics.append(dist.index(max(dist)))

# Group documents by topic
topic_to_docs = {i: [] for i in range(n_topics)}
for doc_name, topic_id in zip(document_names, dominant_topics):
    topic_to_docs[topic_id].append(doc_name)

# Generate HTML
html = "<html><head><title>Topics and Documents</title></head><body>"
html += "<h1>Topics and Their Documents</h1>"
for topic_id in range(n_topics):
    html += f"<h2>Topic {topic_id}</h2><ul>"
    for doc in topic_to_docs[topic_id]:
        html += f"<li>{doc}</li>"
    html += "</ul>"
html += "</body></html>"

# Save HTML
with open('./models/visualizations/topics_documents_note.html', 'w') as f:
    f.write(html)

print("HTML page saved to ./models/visualizations/topics_documents_note.html")
