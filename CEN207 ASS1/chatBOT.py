import nltk, json, _pickle, numpy, tflearn, tensorflow, _random
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("prompts.json") as file:
    data = json.load(file)

#blank lists to store lits of our words and a pickle file to hold our code
try:
    with open("data.pickle", "rb") as d:
        info, labels, training, output = _pickle.load(d)
except:
    info = []
    labels = []
    i_log = []
    tag_log = []
    
#Loops through Json file.
#Learns how to classify the patterns with the corresponding tag
#Tokenizes and stemms words.
for prompt in data['prompts']:
    for pattern in prompt['patterns']:
        i = nltk.word_tokenize(pattern)
        info.extend(i)
        i_log.append(i)
        tag_log.append(prompt["tag"])
        
    if prompt['tag'] not in labels:
        labels.append(prompt['tag'])

#Organises and cleans the list. 
#Turns everyword into lowercase and gets rid of punctuation 
info = [stemmer.stem(i.lower()) for i in info if i != "?"]
info = sorted(list(set(info)))
labels = sorted(labels)

#Bag of words model utlised. Code gets One-hot encoded.
training = []
output = []

blank_out = [0 for _ in range(len(labels))]

for i, log in enumerate(i_log):
    B_o_W = []

    i = [stemmer.stem(i.lower()) for i in log]

    for i in info:
        if i in i:
            B_o_W.append(1)
        else:
            B_o_W.append(0)

    output_row = blank_out[:]
    output_row[labels.index(tag_log [i])] = 1
    training.append(B_o_W)
    output.append(output_row)

#change lists into arrays so tflearn can read
training = numpy.array(training)
output = numpy.array(output)

with open("data.pickle", "wb") as d:
        _pickle.dump((info, labels, training, output), d) 

tensorflow.reset_default_graph()

#Builds neural network. 
#Predicting chatbot response
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#number of times it sees the data the more it learns higher accruacy. 
#Allows it to only run once and be stored. 
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
 
#Using Bag of Words model to transform the the user input so model can read 

def user_input(u, words):
    B_o_W = [0 for _ in range(len(words))]

    u_words = nltk.word_tokenize(u)
    u_words = [stemmer.stem(word.lower()) for word in u_words]

    for se in u_words:
        for i, w in enumerate(words):
            if w == se:
                B_o_W[i] = 1
            
    return numpy.array(B_o_W)

#Chatting with the user. 
def chat():
    print("Hello, I'm bot (type quit to stop)!")
    while True:
        user = input("Me: ")
        if user.lower() == "quit":
            break

        outcome = model.predict([user_input(user, info)])
        outcome_index = numpy.argmax(outcome)
        tag = labels[outcome_index]

        if outcome[outcome_index] > 0.7:
            for tgs in data["prompts"]:
                if tgs["tag"] == tag:
                    responses = tgs["responses"]

        print(_random.choice(responses))
    else: print("I'm sorry, I don't understand. Try again.")

chat()

#References: Tim, https://www.techwithtim.net/tutorials/ai-chatbot/