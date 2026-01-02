import matplotlib.pyplot as plt

class Ploter:
    def __init__(self):
        pass
    def bar_ploter(self,product_counts):
        product_counts.plot(
            kind='bar',
            figsize=(12,6),
            title='Complaints by Product',
            color='#ad12a3'
        )
        plt.xlabel("Product")
        plt.ylabel("Number of Complaints")
        plt.xticks(rotation=45, ha='right')  # rotate labels for readability
        plt.tight_layout()
        plt.show()
    
    def hist_ploter(self, df):
        plt.figure(figsize=(12,6))
        plt.hist(df, bins=200, color="#dd7112")
        plt.title('Distribution of Complaint Narrative Lengths')
        plt.xlabel('narrative_word_count')
        plt.ylabel('Frequency')
        plt.show()
    def pie_with(self, with_narrative, without_narrative):
        plt.figure(figsize=(6,6))
        plt.pie(
            [with_narrative, without_narrative],
            labels=['With Narrative', 'Without Narrative'],
            autopct='%1.1f%%',  # fix: use double %%
            colors=['#ad12a3', '#12a3ad']
        )
        plt.title("Complaints With vs Without Narratives")
        plt.show()