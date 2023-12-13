# import all necessary libraries
import pandas as pd
from wordcloud import STOPWORDS
import nltk
from nltk.corpus import stopwords
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#nltk.download('stopwords')
#nltk.download('punkt')


# function to plot histograms for top words in each genre
def plot_top_words(word_counts, top_n):
    # Cceate a Tkinter window
    root = tk.Tk()
    root.title("Histograms by Genre")

    # initialize a notebook to have multiple tabs of histograms
    notebook = ttk.Notebook(root)

     # get the genres of each category, excluding 'total' 
    genres = [category for category in word_counts.keys() if category != 'total']

    # iterate thorugh the genres
    for genre in genres:
        # get the top n words and their occurrances for the current genre
        top_words = sorted(word_counts[genre].items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_words, frequencies = zip(*top_words)

        # create a new histogram for the current genre
        hist, ax = plt.subplots(figsize=(6, 4))

        # plot a histogram for top words and their frequencies
        ax.bar(top_words, frequencies, color='skyblue')
        ax.set_title(f"Top {top_n} Words in {genre} Genre")
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        # create a new Tkinter canvas 
        canvas = FigureCanvasTkAgg(hist, master=root)
        canvas.draw()

        # put the canvas into the GUI window
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # add the current canvas as a tab in the notebook
        notebook.add(canvas.get_tk_widget(), text=genre)

    # pack the notebook and display all histogram tabs within the GUI window
    notebook.pack(expand=True, fill='both')
    root.mainloop()

#source for understanding how to use dictionaries https://www.w3schools.com/python/python_dictionaries.asp

def filt(strin):
    # replace difficult to deal with characters in the string 
    strin = strin.replace("'", '')
    strin = strin.replace('1', ' ')
    strin = strin.replace('2', ' ')
    strin = strin.replace('3', ' ')
    strin = strin.replace('4', ' ')
    strin = strin.replace('5', ' ')
    strin = strin.replace('6', ' ')
    strin = strin.replace('7', ' ')
    strin = strin.replace('8', ' ')
    strin = strin.replace('9', ' ')
    strin = strin.replace('0', ' ')
    
    # declare list to hold the tokenized string with no stopwords
    filtered_list = []
    # convert string into a list of only words, with no special characters
    words = re.findall(r'\b\w+\b', strin.lower())
    for w in words:
        if w.lower() not in stopwords:
            filtered_list.append(w)

    return filtered_list

# read the dataset from the CSV file into a dataframe
newDF = pd.read_csv('HuluRaw.csv')

# select the necessary columns from the dataset
newDF = newDF[['show/canonical_name', 'show/description', 'show/genre']].copy()
# drop duplicate columns and reset indexes
newDF = newDF.drop_duplicates('show/canonical_name', keep='last', ignore_index =True)
#newDF.reset_index()

# rename columns
newDF.columns = ['Show Name', 'Description', 'Genre']


# separate out the training (80%) and testing (20%) data
# calculate 20% of the dataset
twenPerc = int(len(newDF.index) * 0.2)

# calculate 80% of the dataset
eightyPerc = len(newDF.index) - twenPerc

# create a new dataframe with the first 20% of the data
testData = newDF.head(twenPerc)

# create a new dataframe with the last 80% of the data
trainData = newDF.tail(eightyPerc)

#reset the indexes of the testing dataset
trainData.reset_index(drop=True, inplace=True)

# read in stopwords
f = open("stopwords.txt", encoding="utf8")
stri= ""    
for line in f:
    stri+=line 
# split the string of stopwords and convert it into list
x = stri.split()   
f.close()
# declare set of stopwords and update it with the txt file stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords = set(STOPWORDS)
stopwords.update(x)

### how many shows in the genre that the word occurred in!!!! so 2/5 would be 2 shows out of 5 had this specific word :|
#Reference for Dictionaries: https://www.geeksforgeeks.org/defaultdict-in-python/

# STEP 1: 


word_counts = defaultdict(lambda: defaultdict(int))
category_counts = defaultdict(int)

# iterate through the training data to find word counts  
for index, row in trainData.iterrows():
    # declare a single string to hold the Show Name and Description
    strin = (row['Show Name']  + ' ' + row['Description']).lower()
    # declare a string to hold the genre of the show
    category = row['Genre']
    # filter the string using a method to get the list of the tokenized version of strin, without stopwords
    filtered = filt(strin)
   
   # for each word in the filtered list, add 1 to the count for the word in both the word counts for all categories and the current category
    for word in filtered:
        word_counts[category][word] += 1
        word_counts['total'][word] += 1
    
    # add 1 to the count for the current category, along with all categories
    category_counts[category] += 1
    category_counts['total'] += 1



# calculate word probabilities for each category: 

# laplace smoothing factor
alpha = 1  

# declare dictionary to hold the probabilities of each word in each category
word_probs = defaultdict(lambda: defaultdict(float)) 

# iterate through the category_counts dictionary
for category in category_counts:

    # skip over the total category because we do not need the probability of the word over all categories
    if (category == 'total'):
        continue
    
    # calculate the probability of the category by dividing the occurances of the category by the total occurrances of all categories 
    prob_cat = category_counts[category] / category_counts['total']
    
    # find the total count of words that occurred in the current category
    total_words = sum(word_counts[category].values())
    
    # iterate through all words in the current category to calculate their probabilities
    for word in word_counts[category]:
        #word probability for the current word in the current category = total occurrances of word + alpha(1) / (total words in category + all possible unique words)
        word_probs[category][word] = (word_counts[category][word] + alpha) / (total_words + (len(word_counts['total']))                                                                     
        )

             
# Step 2: Calculate probabilities for each record in the testing data
# declare dictionary to hold the actual and predicted categories
results = []

# iterate through the testing data
for index, row in testData.iterrows():
    # declare a single string to hold the Show Name and Description
    strin = (row['Show Name']  + ' ' + row['Description']).lower()
    # declare a string to hold the genre of the show
    category = row['Genre']
    # filter the string using a method to get the list of the tokenized version of strin, without stopwords
    filtered = filt(strin)
    # declare empty dictionary to hold the probabilities of the current show belonging to each category
    record_probs = defaultdict(float)
    
    # iterate through the categories
    for category in category_counts:
        # find the total count of words that occurred in the current category
        total_words = sum(word_counts[category].values())

        # skip over the total category
        if (category == 'total'):
                continue
    
        # calculate the probability of the category by dividing the occurances of the category by the total occurrances of all categories 
        prob_cat = category_counts[category] / category_counts['total']
        
        # initialize with the probability of the category 
        record_probs[category] = prob_cat 
        
        # iterate through all words in the filtered list
        for word in filtered:
            # if the word did not occur in the training data use laplace smoothing so the probability is 1/ (total words in category + all possible unique words)
            if(word_probs[category][word] == 0):
                # multiply the probability in record_probs and the probability of the current word
                record_probs[category] *= (1 /(total_words + (len(word_counts['total']))))

            # if the word did occur in the training data   
            else:
                # multiply the probability in record_probs and the probability of the current word
                record_probs[category] *= word_probs[category][word]

    # append the actual Genre and the predicted categories to the results dictionary
    results.append((row['Genre'], max(record_probs, key=record_probs.get)))


        
# Step 3: Compare probabilities and assign category tags
# hold total amount of correct predictions
correct = 0
# dictionary holds the correct number of predictions for each category
category_correct = defaultdict(int)
# dictionary holds both correct and incorrect number of predictions for each category
category_total = defaultdict(int)

# iterate through the results
for actual, predicted in results:
    #add to the category total
    category_total[actual] += 1

    # if the prediction is correct
    if actual == predicted:
        # add to the total correct and the category correct
        correct += 1
        category_correct[actual] += 1

        
# calculate overall accuracy percentage by dividing total correct by the length of results (all predictions), then multiply 100
overall_accuracy = correct / len(results) * 100

# declare dictionary to hold the category accuracies
category_accuracy = {}

# iterate through all categories
for category in category_counts:
    # skip the total category
    if (category == 'total'):
        continue

    # if the category was in the testing data
    if category_total[category] > 0:
        #calculate accuracy by dividing category correct by the category total, then multiply 100
        category_accuracy[category] = (category_correct[category] / category_total[category]) * 100

        # if the categoruy did not occur in the testing data
    else:
        # set accuracy to 0 if no records in that category in testing data
        category_accuracy[category] = 0  # Set accuracy to 0 if no records in that category in testing data


# https://www.askpython.com/python/examples/print-a-percentage-value-in-python      
# print the results
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
print("Category-wise Accuracy:")
for category, accuracy in category_accuracy.items():
    print(f"{category}: {accuracy:.2f}%")


plot_top_words(word_counts, 10)
