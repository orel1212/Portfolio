import regex as re

class Preprocessor:
    def __init__(self, datasets, timesteps = 30):
        self.train_set_preprocessed = self.preprocess_dataset(datasets["train_set"],timesteps)
        datasets["train_set"] = self.train_set_preprocessed
        self.valid_set_preprocessed = self.preprocess_dataset(datasets["valid_set"],timesteps)
        datasets["valid_set"] = self.valid_set_preprocessed
        self.test_set_preprocessed = self.preprocess_dataset(datasets["test_set"],timesteps)
        datasets["test_set"] = self.test_set_preprocessed
        print("finished preprocessing the dataset - sentiment140")

    def preprocess_dataset(self, dataset, timesteps):
            dataset_preprocessed = []
            for line in dataset:
                text = self.preprocess_tweets(line[1])
                text = text.split()
                text = text[:timesteps]
                text = " ".join(text)
                dataset_preprocessed.append([line[0],text])
            return dataset_preprocessed


    def preprocess_tweets(self, text):

        FLAGS = re.MULTILINE | re.DOTALL

        def hashtag(text):
            text = text.group()
            hashtag_body = text[1:]
            if hashtag_body.isupper():
                result = "{}".format(hashtag_body.lower())
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
            return result

        def allcaps(text):
            text = text.group()
            return text.lower() + " <allcaps>"


        def tokenize(text):
            # Different regex parts for smiley faces
            eyes = r"[8:=;]"
            nose = r"['`\-]?"

            # function so code less repetitive
            def re_sub(pattern, repl):
                return re.sub(pattern, repl, text, flags=FLAGS)

            text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
            text = re_sub(r"@\w+", "<user>")
            text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
            text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
            text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
            text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
            #text = re_sub(r"/"," / ")
            text = re_sub(r"<3","<heart>")
            text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
            text = re_sub(r"#\S+", hashtag)
            text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
            text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
            text = re_sub(r"([A-Z]){2,}", allcaps)
            return text.lower()
        return tokenize(text)