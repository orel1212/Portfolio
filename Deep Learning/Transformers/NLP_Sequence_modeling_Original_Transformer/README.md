# Question-answering system based on Orignal Transformer with BytePair GPT2 tokenizer

## Dataset
Cifar10 dataset
<a> https://huggingface.co/datasets/cifar10 </a>
<a> https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html </a>
A corpus that contains a large metadata-rich collection of fictional conversations extracted from raw movie scripts
## Models
Transformer - original
<a> https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need </a>
Bytepair GPT2 Tokenizer
<a> https://huggingface.co/docs/transformers/model_doc/gpt2 </a>

## Files
### question_answering_train.ipynb
running notebook of the code of the train
### question_answering_test.ipynb
running notebook of the code of the test
### transformer.py
full implementation of the transformer with GPT2 Byte-pair tokenizer

## Method - ODM
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}&space;ODM_{SSM}&space;=&space;ODM(\alpha&space;\cdot&space;IMG_{input}&space;&plus;&space;(1-\alpha)&space;\cdot&space;SSM_{output})&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{equation}&space;ODM_{SSM}&space;=&space;ODM(\alpha&space;\cdot&space;IMG_{input}&space;&plus;&space;(1-\alpha)&space;\cdot&space;SSM_{output})&space;\end{equation}" title="\begin{equation} ODM_{SSM} = ODM(\alpha \cdot IMG_{input} + (1-\alpha) \cdot SSM_{output}) \end{equation}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}&space;ODM_{IMG}&space;=&space;ODM(IMG_{input})&space;\label{formulas:img_odm_equation}&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{equation}&space;ODM_{IMG}&space;=&space;ODM(IMG_{input})&space;\label{formulas:img_odm_equation}&space;\end{equation}" title="\begin{equation} ODM_{IMG} = ODM(IMG_{input}) \label{formulas:img_odm_equation} \end{equation}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}&space;SIM(O_{o},O_{s})&space;=&space;(1&space;-&space;|confidence_{o}&space;-&space;confidence_{s}|)&space;\cdot&space;IoU_{o,s}&space;\label{formulas:objects_sim_equation}&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{equation}&space;SIM(O_{o},O_{s})&space;=&space;(1&space;-&space;|confidence_{o}&space;-&space;confidence_{s}|)&space;\cdot&space;IoU_{o,s}&space;\label{formulas:objects_sim_equation}&space;\end{equation}" title="\begin{equation} SIM(O_{o},O_{s}) = (1 - |confidence_{o} - confidence_{s}|) \cdot IoU_{o,s} \label{formulas:objects_sim_equation} \end{equation}" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}&space;SIM(ODM_{IMG},ODM_{SSM})&space;=&space;\frac{1}{K}&space;\cdot&space;\sum_{\substack{&space;i&space;=&space;1&space;\\&space;O_{o_{i}}\in&space;ODM_{IMG}\\O_{s_{i}}\in&space;ODM_{SSM}}}^{K}&space;C&space;\cdot&space;SIM(O_{o_{i}},O_{s_{i}})&space;\label{formulas:images_sim_equation}&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{equation}&space;SIM(ODM_{IMG},ODM_{SSM})&space;=&space;\frac{1}{K}&space;\cdot&space;\sum_{\substack{&space;i&space;=&space;1&space;\\&space;O_{o_{i}}\in&space;ODM_{IMG}\\O_{s_{i}}\in&space;ODM_{SSM}}}^{K}&space;C&space;\cdot&space;SIM(O_{o_{i}},O_{s_{i}})&space;\label{formulas:images_sim_equation}&space;\end{equation}" title="\begin{equation} SIM(ODM_{IMG},ODM_{SSM}) = \frac{1}{K} \cdot \sum_{\substack{ i = 1 \\ O_{o_{i}}\in ODM_{IMG}\\O_{s_{i}}\in ODM_{SSM}}}^{K} C \cdot SIM(O_{o_{i}},O_{s_{i}}) \label{formulas:images_sim_equation} \end{equation}" /></a><br>

![alt tag](https://github.com/orel1212/MyWorks/blob/main/Deep%20Learning/SSMODM/%E2%80%8F%E2%80%8Falpha.PNG)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Deep%20Learning/SSMODM/%E2%80%8F%E2%80%8Fod_on_image.PNG)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Deep%20Learning/SSMODM/%E2%80%8F%E2%80%8Fexamples.PNG)
