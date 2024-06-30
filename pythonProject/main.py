import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import preprocess
from transformers import BertModel
import json
import emoji


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,matthews_corrcoef




from tqdm import tqdm, trange,tnrange,tqdm_notebook
import random
import os
import io
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

class BERT_14(torch.nn.Module):

    def __init__(self, bert):
        super(BERT_14, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = torch.nn.Dropout(0.1)

        # relu activation function
        self.relu = torch.nn.ReLU()

        # dense layer 1
        self.fc1 = torch.nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = torch.nn.Linear(512, 6)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        output = self.bert(sent_id, attention_mask=mask)
        x = output[0][:, 0, :]
        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        #       # apply softmax activation
        #       x = F.softmax(x, dim = 1)

        return x

def train(model, train_dataloader, val_dataloader, epochs):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch in range(epochs):
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()
        best = 0
        total_steps = len(train_dataloader)        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            logits = model(b_input_ids, b_attn_mask)

            loss = criteria(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            if step % 50 == 0:
                # Calculate time elapsed for 20 batches

                # Print training results
                print("Train:Epoch ", epoch, ": ", batch_loss/batch_counts," Progress:{}/{}".format(step,total_steps))
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        val_loss, val_accuracy = evaluate(model, val_dataloader)
        if val_accuracy > best:
            best = val_accuracy
            model_save_name = 'fineTuneModel.bin'
            path = path_model = './{model_save_name}'
            torch.save(model.state_dict(),path)

        print("Val loss: ",val_loss,"; Val accuracy: ", val_accuracy)

def evaluate(model, val_dataloader):
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = criteria(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def preprocessing(text): #预测时的preprocess
    input_id = tokenizer.encode(text, add_special_tokens=True,padding='longest',truncation=True,return_token_type_ids=False)
    attention = [float(i>0) for i in input_id]
    input_id = torch.tensor([input_id]).to(device)
    attention = torch.tensor([attention]).to(device)
    return input_id, attention

def predict(model,text):
        input_ids, attention_mask = preprocessing(text)

        with torch.no_grad():
            probabilities = F.softmax(model(input_ids, attention_mask).detach())
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            config["CLASS_NAMES"][predicted_class],
            confidence,
            dict(zip(config["CLASS_NAMES"], probabilities)),
        )
def test(val_dataloader,model):
    save = []
    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        predict = logits.to("cpu").numpy()
        predict = np.argmax(predict, axis=1).flatten()
        labels_flat = b_labels.to("cpu").numpy().flatten()
        temp = pd.DataFrame({'Actual_class': labels_flat, 'Predicted_class': predict})
        save.append(temp)

    df_metrics = pd.concat(save, axis=0)
    df_metrics = df_metrics.reset_index()
    emotions_dict={emotion: None for emotion in config["CLASS_NAMES"]}
    print(classification_report(df_metrics['Actual_class'].values, df_metrics['Predicted_class'].values,
                                target_names=emotions_dict.keys(), digits=len(config["CLASS_NAMES"])))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open("config.json") as json_file:
        config = json.load(json_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(SEED)

    #preprocess
    train_text, val_text, train_labels, val_labels= preprocess.preprocess()

    #Tokenizer
    MAX_LEN = 256
    model_directory = './bert-base-uncased/'
    tokenizer = BertTokenizer.from_pretrained(model_directory)
    # tokenize and encode sequences in the training set (for Bertbase fine tuning)
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=MAX_LEN,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=MAX_LEN,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    # Create iterator of data with DataLoader
    # for train set
    train_inputs = torch.tensor(tokens_train['input_ids'])
    train_y = torch.tensor(train_labels.tolist())
    train_masks = torch.tensor(tokens_train['attention_mask'])

    # for validation set
    val_inputs = torch.tensor(tokens_val['input_ids'])
    val_y = torch.tensor(val_labels.tolist())
    val_masks = torch.tensor(tokens_val['attention_mask'])

    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_y)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # Load model
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    model = BERT_14(bert)
    model.to(device)

    #Instantiate optimizer and scheduler
    epochs = 4
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )
    num_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=num_steps)
    criteria = torch.nn.CrossEntropyLoss()

    #train
    #train(model, train_dataloader, val_dataloader, epochs)
    #model_save_name = 'fineTuneModel.bin'
    #torch.save(model.state_dict(), model_save_name)

    model.load_state_dict(torch.load("./fineTuneModel.bin"))
    model.to(device)
    test(val_dataloader,model)

    '''  
    #predict
    text = "she is happy"
    model.load_state_dict(torch.load("./fineTuneModel.bin"))
    model.to(device)
    result= predict(model,text)

    # Define a mapping from emotions to emojis
    emotion_to_emoji = {
        'anger': emoji.emojize(':angry_face:'),
        'fear': emoji.emojize(':fearful_face:'),
        'joy': emoji.emojize(':smiling_face_with_smiling_eyes:'),
        'love': emoji.emojize(':red_heart:'),
        'sadness': emoji.emojize(':crying_face:'),
        'surprise': emoji.emojize(':face_with_open_mouth:')
    }

    # Get the corresponding emoji for "sadness"
    emoji = emotion_to_emoji[result[0]]
    print(emoji)
    '''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/





