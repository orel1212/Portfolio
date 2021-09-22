
from nltk.tokenize import TweetTokenizer
import time

class Preprocessor:
    def __init__(self, datasets, sequence_list, misc_categories):
        self.tokenizer = TweetTokenizer()
        datasets["test_set"] = self.preprocess_dataset(datasets["test_set"],sequence_list, misc_categories)
        datasets["valid_set"] = self.preprocess_dataset(datasets["valid_set"],sequence_list, misc_categories)
        datasets["train_set"] = self.preprocess_dataset(datasets["train_set"],sequence_list, misc_categories)
        print("finished preprocessing the dataset - wnut")

    def preprocess_dataset(self, dataset, sequence_list, misc_categories):
            def slice_helper(text, timestep):
                for i,word in enumerate(text):
                    text[i] = ''.join(word.split())
                text = text[:timestep]
                text = " ".join(text)
                return text

            dataset_preprocessed = []
            for line in dataset:
                text = self.tokenizer.tokenize(line[1])
                text = slice_helper(text,sequence_list[0])
                if line[2] is None:
                    desc = self.tokenizer.tokenize("")
                else:
                    desc = self.tokenizer.tokenize(line[2])
                desc = slice_helper(desc,sequence_list[1])
                username = self.tokenizer.tokenize(line[3])
                username = slice_helper(username,sequence_list[2])
                if line[4] is None:
                	profile_location = self.tokenizer.tokenize("")
                else:
                	profile_location = self.tokenizer.tokenize(line[4])
                profile_location = slice_helper(profile_location,sequence_list[3])

                tweet_lang = misc_categories["tweet_lang"].index(line[5])
                user_lang = misc_categories["user_lang"].index(line[6])
                time_zone = misc_categories["time_zone"].index(line[7])

                try:
                    hours = int(time.strftime('%H', time.strptime(line[8],'%a %b %d %H:%M:%S +0000 %Y')))
                    first_minute_digit = int(time.strftime('%M', time.strptime(line[8],'%a %b %d %H:%M:%S +0000 %Y'))[0])
                    posting_time = hours*6 + first_minute_digit
                except:
                    print("Error preprocessing posting time of tweet exiting program")
                    exit()
                country_label = misc_categories["country"].index(line[0])
                data = [country_label, text, desc, username, profile_location, tweet_lang, 
                        user_lang, time_zone, posting_time]
                dataset_preprocessed.append(data)
            return dataset_preprocessed


