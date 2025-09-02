import pickle
import os

# Load the saved model dictionary
with open('./models/enhanced_lda_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

document_names = model_data['document_names']
topic_distributions = model_data['topic_distributions']
n_topics = model_data['n_topics']

# Load cluster results (assuming you saved as a DataFrame pickle)
with open('./models/visualizations/cluster_results.pkl', 'rb') as f:
    cluster_results = pickle.load(f)

# Group documents by cluster
cluster_to_docs = {}
for idx, row in cluster_results.iterrows():
    cluster = row['cluster']
    doc_name = row['document']
    cluster_to_docs.setdefault(cluster, []).append(doc_name)

# Generate HTML
html = "<html><head><title>Document Clusters</title></head><body>"
html += "<h1>Documents Grouped by Cluster</h1>"
for cluster_id in sorted(cluster_to_docs.keys()):
    html += f"<h2>Cluster {cluster_id}</h2><ul>"
    for doc in cluster_to_docs[cluster_id]:
        
        pdf_name = os.path.splitext(doc)[0] + ".docx"
        pdf_path = f"/Users/ethanqiu/Documents/projects/SmallRAG/data/{pdf_name}"
        html += f'<li><a href="{pdf_path}">{doc}</a></li>'
    html += "</ul>"
html += "</body></html>"

# Save HTML
os.makedirs('./models/visualizations', exist_ok=True)
with open('./models/visualizations/document_clusters_links.html', 'w') as f:
    f.write(html)

print("HTML page with document cluster links saved to ./models/visualizations/document_clusters_links.html")