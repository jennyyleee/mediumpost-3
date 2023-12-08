import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

animals_df = pd.read_csv("zoo2.csv")
print(animals_df)

#removing name and type
attributes_df = animals_df.drop(columns=['animal_name', 'class_type'])

#changing categorical lables to unique numbers
for column in attributes_df.columns:
    attributes_df[column] = attributes_df[column].astype('category').cat.codes

# Calculating cosine similarity
similarity_matrix = cosine_similarity(attributes_df)

#The function finds and returns N similar animals to the animal_name provided 
#by the user the function will find 10 similar animals as default
def find_similar_animals(animal_name, n=10):
   
    try:
        animal_index = animals_df.index[animals_df['animal_name'] == animal_name].tolist()[0]
        similar_animals_indices = similarity_matrix[animal_index].argsort()[-n-1:-1][::-1]

        similar_animals = animals_df.loc[similar_animals_indices, 'animal_name'].tolist()
        similarity_values = similarity_matrix[animal_index][similar_animals_indices].tolist()
        
        similar_animals_with_similarity = {animal: score for animal, score in zip(similar_animals, similarity_values)}
        return similar_animals_with_similarity
    except IndexError:
        return f"There were no similar animals to '{animal_name}'."

#Query 1
print("first query:")
query1= 'turtle'
similar_animals = find_similar_animals(query1)
print(similar_animals)

#Query 2
print("second query:")
query2= 'butterfly'
similar_animals = find_similar_animals(query2)
print(similar_animals)

#Query 3
print("third query:")
query3= 'jellyfish'
similar_animals = find_similar_animals(query3)
print(similar_animals)


