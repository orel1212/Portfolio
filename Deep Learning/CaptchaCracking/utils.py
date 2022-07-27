import torch
import torch.nn.functional as F
import numpy as np

import os
import matplotlib.pyplot as plt

import fnmatch


def train_valid_test_split(dataset, train_ratio, test_ratio, batch_size, seed=42):
    total_count = len(dataset)
    train_count = int(train_ratio * total_count)
    test_count = int(test_ratio * total_count)
    valid_count = total_count - train_count - test_count
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_count, valid_count, test_count), generator=generator)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader, valid_loader, test_loader


def get_num_curr_models(path, file_name, file_ext):
    files = fnmatch.filter((f for f in os.listdir(path)), file_name + '*' + file_ext)  # target filename
    return len(files)


def show_error_images(errors_images_dict, end_sentence_char, req_size, to_rand=True):
    errors_pred_image_lst = []
    for _, vals in errors_images_dict.items():
        _, char_preds, char_labels, images = vals
        for i in range(len(images)):
            label_str = ''.join(char_labels[i])
            pred_str = ''.join(char_preds[i])

            num_errors = sum(map(lambda x, y: x != y, label_str, pred_str))
            label_str = label_str.replace(end_sentence_char, "")
            pred_str = pred_str.replace(end_sentence_char, "")
            num_errors = min(num_errors, len(label_str))
            str_title = pred_str.upper() + '( Mistakes: ' + str(num_errors) + ')'
            errors_pred_image_lst.append([str_title, images[i]])

    if to_rand is True:
        size_to_rand = min(req_size, len(errors_pred_image_lst))
        errors_pred_image_idx_lst = np.random.choice(range(len(errors_pred_image_lst)), size=size_to_rand,
                                                     replace=False)
    else:
        errors_pred_image_idx_lst = range(len(errors_pred_image_lst))

    dims = np.ceil(np.sqrt(req_size)).astype(int)
    fig = plt.figure("Captcha Wrong Predictions", figsize=(16, 16))

    for fig_idx, pred_id in enumerate(errors_pred_image_idx_lst):
        str_title, img = errors_pred_image_lst[pred_id]
        fig.add_subplot(dims, dims, fig_idx + 1)
        plt.axis("off")
        plt.title(str_title, fontsize=16, fontweight="bold")
        plt.imshow(img)
    plt.show()


class GreedyCTCDecoder():
    def __init__(self, blank=0, end_sentence=1):
        self.blank = blank
        self.end_sentence = end_sentence

    def _remove_duplicates_and_zeros(self, indices):
        len_indices = len(indices)
        derivative_mask = np.insert(np.diff(indices).astype(bool), 0, True)  # evaluate duplicate values by np.diff
        indices = indices[np.ix_(derivative_mask)]
        indices = indices[indices > 0]
        append_end_sentence_len = len_indices - len(indices)
        indices = np.concatenate((indices, np.array(append_end_sentence_len * [self.end_sentence])), axis=0)
        return indices

    def forward(self, preds, classes_lst):
        log_probs = F.log_softmax(preds, dim=-1)  # dim:[seq_len, batch, num_classes]
        indices = torch.argmax(log_probs, dim=-1).transpose(0, 1)  # dim:[batch, seq_len]
        np_indices = np.copy(indices.numpy())

        np_indices = np.apply_along_axis(self._remove_duplicates_and_zeros, 1, np_indices)
        np_chars_lst = np.choose(np_indices, classes_lst)

        indices = torch.LongTensor(np_indices)
        return indices, np_chars_lst
